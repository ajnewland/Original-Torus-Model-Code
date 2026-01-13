#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-geometry probe for a torus ensemble:
- Builds N randomized copies of a base torus slice (subpixel shifts; optional small rotations)
- Places them on a larger periodic canvas; forms an ensemble-averaged curvature map
- Measures curvature correlation vs inter-torus separation C_RR(d)
- Builds a graph over torus centers and estimates meta spectral dimension via heat trace

Inputs:
    --h5_base:       path to a single torus fields .h5 (e.g. *_t0_fields.h5)
    --dataset_lnOm:  HDF5 key for lnOmega (default: auto; if 'Omega' present, will use log(Omega))
    --dataset_Teff:  HDF5 key for Teff (unused here; reserved)
    --dataset_rho_m: HDF5 key for rho_m (unused here; reserved)
    --N  :           number of tori in ensemble (e.g. 32, 64, 100)
    --alpha:         conformal coupling (e.g. 4.9) [only used if reconstructing from lnΩ/Ω]
    --smooth_px:     Gaussian smoothing sigma [pixels] for presentation (e.g. 72)
    --mask_margin:   crop border in pixels to avoid wrap artefacts (e.g. 8)
    --operator:      laplacian operator: spectral | nine  (default: spectral)
    --canvas_factor: size multiplier for the big canvas vs single torus (e.g. 3.0 => 3×Nx by 3×Ny)
    --rot_deg:       max absolute rotation in degrees (small, e.g. 5). If scipy absent, rotations are skipped.
    --seed:          RNG seed
    --sigma_graph:   graph length-scale for center-center edges (in base-pixel units; e.g. 0.75*min(Nx,Ny))
    --wsize:         side length of local windows (pixels) used in correlation
    --outdir:        output directory

Outputs (in outdir):
    meta_summary.json
    centers.csv                      (torus centers & phases)
    corr_vs_distance.csv             (binned C_RR(d))
    meta_heat_trace.csv              (t, K(t), ds_meta(t))
    figs/ensemble_canvas.png
    figs/corr_vs_d.png
    figs/heat_trace.png
    figs/ds_meta.png
"""

import os, json, argparse, math
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Optional rotation support
try:
    from scipy.ndimage import rotate as sci_rotate
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ---------- Utilities ----------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def gaussian_blur_fft(img, sigma_px):
    if sigma_px is None or sigma_px <= 0:
        return img
    nx, ny = img.shape
    kx = np.fft.fftfreq(nx)[:, None]
    ky = np.fft.fftfreq(ny)[None, :]
    # Gaussian in freq domain: exp(-2*pi^2*sigma^2 * (kx^2+ky^2))
    G = np.exp(-2*(np.pi**2)*(sigma_px**2)*(kx**2+ky**2))
    F = np.fft.fftn(img)
    return np.fft.ifftn(F*G).real

def laplacian_spectral(f):
    nx, ny = f.shape
    kx = (2*np.pi)*np.fft.fftfreq(nx)[:, None]
    ky = (2*np.pi)*np.fft.fftfreq(ny)[None, :]
    F = np.fft.fftn(f)
    lapF = -(kx*kx + ky*ky) * F
    return np.fft.ifftn(lapF).real

def laplacian_nine(f):
    # 9-point stencil (periodic)
    kern = np.array([[1,  4,  1],
                     [4, -20, 4],
                     [1,  4,  1]]) / 6.0
    # periodic pad via roll
    out = (kern[0,0]*np.roll(np.roll(f, -1, 0), -1, 1) +
           kern[0,1]*np.roll(f, -1, 0) +
           kern[0,2]*np.roll(np.roll(f, -1, 0),  1, 1) +
           kern[1,0]*np.roll(f, -1, 1) +
           kern[1,1]*f +
           kern[1,2]*np.roll(f,  1, 1) +
           kern[2,0]*np.roll(np.roll(f,  1, 0), -1, 1) +
           kern[2,1]*np.roll(f,  1, 0) +
           kern[2,2]*np.roll(np.roll(f,  1, 0),  1, 1))
    return out

def grad_xy_spectral(f):
    nx, ny = f.shape
    kx = (2*np.pi)*np.fft.fftfreq(nx)[:, None]
    ky = (2*np.pi)*np.fft.fftfreq(ny)[None, :]
    F = np.fft.fftn(f)
    dfdx = np.fft.ifftn((1j*kx)*F).real
    dfdy = np.fft.ifftn((1j*ky)*F).real
    return dfdx, dfdy

def frac_shift_fft(img, sx, sy):
    """Subpixel shift by (sx, sy) pixels, periodic, via Fourier phase."""
    nx, ny = img.shape
    kx = np.fft.fftfreq(nx)[:, None]
    ky = np.fft.fftfreq(ny)[None, :]
    phase = np.exp(-2j*np.pi*(kx*sx + ky*sy))
    return np.fft.ifftn(np.fft.fftn(img)*phase).real

def load_R_from_h5(h5, key_candidates=('R', 'R2', 'R_eff')):
    """Try to read a curvature field directly."""
    for k in key_candidates:
        if k in h5:
            R = h5[k][:]
            return R, k
    return None, None

def curvature_from_lnOm(lnOm, operator='spectral', alpha=4.9, smooth_px=0.0):
    lnOm_s = gaussian_blur_fft(lnOm, smooth_px)
    Om = np.exp(lnOm_s)
    Om /= (Om.mean() + 1e-30)
    if operator == 'nine':
        lap_ln = laplacian_nine(lnOm_s)
    else:
        lap_ln = laplacian_spectral(lnOm_s)
    # R in conformal 2D toy: R ~ -2 ∇² lnΩ / Ω²
    R = -2.0*lap_ln / (Om*Om + 1e-30)
    return R, Om

# ---------- Ensemble builder ----------

def build_ensemble_fields(lnOm_base=None, R_base=None, N=64, alpha=4.9, smooth_px=72.0,
                          operator='spectral', canvas_factor=3.0, rot_deg=5.0, seed=0):
    """
    Build an ensemble-averaged curvature canvas from either lnΩ or a base R field.
    """
    rng = np.random.default_rng(seed)
    if lnOm_base is None and R_base is None:
        raise ValueError("Provide lnOm_base or R_base")
    if lnOm_base is not None:
        nx, ny = lnOm_base.shape
    else:
        nx, ny = R_base.shape

    NX = int(round(canvas_factor*nx))
    NY = int(round(canvas_factor*ny))

    centers = []
    instances = []
    for _ in range(N):
        sx = rng.uniform(-nx/2, nx/2)
        sy = rng.uniform(-ny/2, ny/2)

        if lnOm_base is not None:
            ln_i = frac_shift_fft(lnOm_base, sx, sy)
            theta = rng.uniform(-rot_deg, rot_deg) if HAS_SCIPY and rot_deg>0 else 0.0
            if HAS_SCIPY and rot_deg>0:
                ln_i = sci_rotate(ln_i, theta, reshape=False, order=1, mode='wrap')
            R_i, _ = curvature_from_lnOm(ln_i, operator=operator, alpha=alpha, smooth_px=smooth_px)
        else:
            R_i = frac_shift_fft(R_base, sx, sy)
            theta = rng.uniform(-rot_deg, rot_deg) if HAS_SCIPY and rot_deg>0 else 0.0
            if HAS_SCIPY and rot_deg>0:
                R_i = sci_rotate(R_i, theta, reshape=False, order=1, mode='wrap')

        cx = rng.uniform(0, NX); cy = rng.uniform(0, NY)
        centers.append((cx, cy, sx, sy, theta))
        instances.append({'R': R_i})

    centers = np.array(centers)

    # Periodic stamping onto the canvas (vectorized across y for speed)
    R_canvas = np.zeros((NX, NY), dtype=float)
    W_canvas = np.zeros((NX, NY), dtype=float)
    for i in range(N):
        R_i = instances[i]['R']; w_i = 1.0
        ix0 = int(np.floor(centers[i,0])) % NX
        iy0 = int(np.floor(centers[i,1])) % NY
        for dx in range(nx):
            x = (ix0 + dx) % NX
            ys = (iy0 + np.arange(ny)) % NY
            R_canvas[x, ys] += w_i * R_i[dx, :]
            W_canvas[x, ys] += w_i
    W_canvas = np.where(W_canvas>0, W_canvas, 1.0)
    R_eff = R_canvas / W_canvas
    return R_eff, centers

# ---------- Correlations & spectral dimension ----------

def build_windows_from_canvas(R_eff, centers, wsize=32):
    """Extract aligned square windows of size wsize×wsize around each center (periodic)."""
    NX, NY = R_eff.shape
    half = wsize//2
    wins = []
    for (cx,cy,_,_,_) in centers:
        ix = int(np.floor(cx)) % NX
        iy = int(np.floor(cy)) % NY
        xs = [(ix+dx) % NX for dx in range(-half, half)]
        ys = [(iy+dy) % NY for dy in range(-half, half)]
        patch = R_eff[np.ix_(xs, ys)].copy()
        wins.append(patch)
    return wins

def corr_vs_distance(centers, win_Rs, NX, NY, nbins=24):
    """
    Curvature correlation vs inter-center distance.
    Pearson correlation between windowed patches, binned by separation d (periodic metric).
    """
    N = len(win_Rs)
    ds, cs = [], []
    for i in range(N):
        for j in range(i+1, N):
            dx = centers[i,0]-centers[j,0]
            dy = centers[i,1]-centers[j,1]
            dx = min(abs(dx), NX-abs(dx))
            dy = min(abs(dy), NY-abs(dy))
            d = math.hypot(dx, dy)
            ri = win_Rs[i].ravel()
            rj = win_Rs[j].ravel()
            ri_m = ri - ri.mean()
            rj_m = rj - rj.mean()
            denom = (ri_m.std()+1e-30)*(rj_m.std()+1e-30)
            c = float((ri_m*rj_m).sum() / (len(ri)*denom))
            ds.append(d); cs.append(c)
    ds = np.array(ds); cs = np.array(cs)
    dmax = 0.5*min(NX, NY)
    bins = np.linspace(0, dmax, 24+1) if nbins is None else np.linspace(0, dmax, nbins+1)
    mids = 0.5*(bins[:-1]+bins[1:])
    C = np.zeros(len(mids))
    Ncnt = np.zeros(len(mids), dtype=int)
    for k in range(len(mids)):
        mask = (ds>=bins[k]) & (ds<bins[k+1])
        if mask.any():
            C[k] = cs[mask].mean()
            Ncnt[k] = mask.sum()
        else:
            C[k] = np.nan
    return mids, C, Ncnt

def meta_graph_spectral_dimension(centers, sigma_graph, t_vals):
    """
    Fully-connected Gaussian-weighted graph on torus centers:
        w_ij = exp(-d^2 / (2 sigma_graph^2)) for i != j; w_ii = 0
    Normalized Laplacian L = I - D^{-1/2} W D^{-1/2}.
    Heat trace K(t) = Tr[ exp(-t L) ], and d_s(t) = -2 d ln K / d ln t.
    """
    pos = centers[:, :2]
    N = pos.shape[0]
    DX = pos[:,0,None]-pos[None,:,0]
    DY = pos[:,1,None]-pos[None,:,1]
    D2 = DX*DX + DY*DY
    W = np.exp(-0.5*D2/(sigma_graph*sigma_graph))
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)
    d_safe = np.where(d>1e-30, d, 1.0)
    Dmh = np.diag(1.0/np.sqrt(d_safe))
    L = np.eye(N) - Dmh @ W @ Dmh
    evals = np.linalg.eigvalsh(L)  # symmetric
    evals = np.clip(evals, 0, None)
    K = np.array([np.sum(np.exp(-t*evals)) for t in t_vals])
    lnt = np.log(t_vals); lnK = np.log(K)
    ds = np.zeros_like(K)
    for i in range(1, len(t_vals)-1):
        slope = (lnK[i+1]-lnK[i-1])/(lnt[i+1]-lnt[i-1])
        ds[i] = -2.0*slope
    ds[0] = ds[1]; ds[-1] = ds[-2]
    return K, ds

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Meta-geometry probe for a torus ensemble")
    ap.add_argument('--h5_base', required=True)
    ap.add_argument('--dataset_lnOm', default=None)
    ap.add_argument('--dataset_Teff', default=None)
    ap.add_argument('--dataset_rho_m', default=None)
    ap.add_argument('--N', type=int, default=64)
    ap.add_argument('--alpha', type=float, default=4.9)
    ap.add_argument('--smooth_px', type=float, default=72.0)
    ap.add_argument('--mask_margin', type=int, default=8)
    ap.add_argument('--operator', choices=['spectral','nine'], default='spectral')
    ap.add_argument('--canvas_factor', type=float, default=3.0)
    ap.add_argument('--rot_deg', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sigma_graph', type=float, default=None,
                   help='graph length scale in base pixels (default: 0.75*min(nx,ny))')
    ap.add_argument('--wsize', type=int, default=32, help='window size for local R patches')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir,'figs'))

    # Load base torus fields
    with h5py.File(args.h5_base,'r') as h5:
        # Try curvature directly first
        R_direct, R_key = load_R_from_h5(h5)
        if R_direct is not None:
            R_base = R_direct
            lnOm = None
            nx, ny = R_base.shape
            base_mode = f"R-from-H5[{R_key}]"
        else:
            keys = list(h5.keys())
            # explicit dataset override
            if args.dataset_lnOm is not None and args.dataset_lnOm in keys:
                if args.dataset_lnOm.lower() in ('omega', 'om', 'Omega'):
                    Om = h5[args.dataset_lnOm][:]
                    lnOm = np.log(np.clip(Om, 1e-30, None))
                else:
                    lnOm = h5[args.dataset_lnOm][:]
            else:
                # auto-detect common keys
                if 'lnOm' in keys:
                    lnOm = h5['lnOm'][:]
                elif 'lnOmega' in keys:
                    lnOm = h5['lnOmega'][:]
                elif 'Omega' in keys:
                    Om = h5['Omega'][:]
                    lnOm = np.log(np.clip(Om, 1e-30, None))
                else:
                    raise RuntimeError(f"Cannot find lnΩ or Ω in {keys}")
            nx, ny = lnOm.shape
            R_base = None
            base_mode = "lnOmega-path"

    # Build ensemble canvas & centers
    if base_mode.startswith("R-from-H5"):
        R_eff, centers = build_ensemble_fields(
            lnOm_base=None, R_base=R_base, N=args.N, alpha=args.alpha, smooth_px=args.smooth_px,
            operator=args.operator, canvas_factor=args.canvas_factor, rot_deg=args.rot_deg, seed=args.seed
        )
    else:
        R_eff, centers = build_ensemble_fields(
            lnOm_base=lnOm, R_base=None, N=args.N, alpha=args.alpha, smooth_px=args.smooth_px,
            operator=args.operator, canvas_factor=args.canvas_factor, rot_deg=args.rot_deg, seed=args.seed
        )

    NX, NY = R_eff.shape

    # Figure: ensemble curvature canvas (masked borders for display)
    mm = args.mask_margin
    i0, i1 = mm, max(mm, NX-mm)
    j0, j1 = mm, max(mm, NY-mm)
    R_show = R_eff[i0:i1, j0:j1]
    v = np.nanpercentile(R_show, [5,95])
    plt.figure(figsize=(7,6))
    plt.imshow(R_show.T, origin='lower', cmap='coolwarm', vmin=v[0], vmax=v[1], aspect='equal')
    plt.title(f'Ensemble curvature R (N={args.N}, {args.operator}, smooth_px={args.smooth_px})')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir,'figs','ensemble_canvas.png'), dpi=150)
    plt.close()

    # Local windows around centers on the canvas
    win_Rs = build_windows_from_canvas(R_eff, centers, wsize=args.wsize)
    mids, C, Ncnt = corr_vs_distance(centers, win_Rs, NX, NY, nbins=24)

    # Plot correlation vs distance
    plt.figure(figsize=(6,4))
    plt.plot(mids, C, '-o', ms=3)
    plt.xlabel('Inter-torus distance d (pixels on canvas)')
    plt.ylabel('Correlation C_RR(d)')
    plt.title('Curvature correlation vs separation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir,'figs','corr_vs_d.png'), dpi=150)
    plt.close()

    # Meta-graph spectral dimension
    base_min = min(nx, ny)
    if args.sigma_graph is None:
        args.sigma_graph = 0.75*base_min
    t_vals = np.logspace(-3, 1, 60)  # graph diffusion times
    K, ds = meta_graph_spectral_dimension(centers, args.sigma_graph, t_vals)

    # Plots: heat trace and ds_meta
    plt.figure(figsize=(6,4))
    plt.loglog(t_vals, K, '-')
    plt.xlabel('t'); plt.ylabel('K(t) = Tr[e^{-tL}]')
    plt.title('Meta-graph heat trace')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir,'figs','heat_trace.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.semilogx(t_vals, ds, '-')
    plt.xlabel('t'); plt.ylabel(r'$d_s^{\rm meta}(t)$')
    plt.ylim(0,6)
    plt.title('Meta spectral dimension')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir,'figs','ds_meta.png'), dpi=150)
    plt.close()

    # Save CSVs & summary
    np.savetxt(os.path.join(args.outdir,'centers.csv'),
               centers, delimiter=',',
               header='cx,cy,shift_x,shift_y,theta', comments='')
    np.savetxt(os.path.join(args.outdir,'corr_vs_distance.csv'),
               np.c_[mids, C, Ncnt], delimiter=',',
               header='d_mid,C_RR,Npairs', comments='')
    np.savetxt(os.path.join(args.outdir,'meta_heat_trace.csv'),
               np.c_[t_vals, K, ds], delimiter=',',
               header='t,K(t),ds_meta', comments='')

    summary = dict(
        N=args.N, alpha=args.alpha, smooth_px=args.smooth_px, operator=args.operator,
        canvas_factor=args.canvas_factor, rot_deg=args.rot_deg, seed=args.seed,
        sigma_graph=args.sigma_graph, wsize=args.wsize, NX=NX, NY=NY,
        base_mode=base_mode,
        notes="Look for a long-range tail in C_RR(d) and a ds_meta plateau near ~4 over a finite t-band."
    )
    with open(os.path.join(args.outdir,'meta_summary.json'),'w') as f:
        json.dump(summary, f, indent=2)

    print("=== Meta-geometry probe ===")
    print(f"Base: {base_mode} | N={args.N} | operator={args.operator} | smooth_px={args.smooth_px}")
    print(f"Canvas: {NX}x{NY} | windowsize={args.wsize} | sigma_graph={args.sigma_graph:.2f}")
    print(f"Saved to: {args.outdir}")

if __name__ == "__main__":
    main()