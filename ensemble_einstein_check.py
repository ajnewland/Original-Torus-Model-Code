#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble Einstein + Momentum Check (stable, with augmentation)

- Loads a series of torus slices: <series_prefix>_t0_fields.h5 ... _tK_fields.h5
- (Optionally) augments each slice by integer or sub-pixel shifts (+ tiny phase jitter)
- Averages to form an ensemble, then evaluates:
    R2 = A*rho_phi + B*rho_m + C   (robust fit)
    Momentum constraint (with and without source subtraction)
- Reports ein_rms, ein_rel, rms_R2, mom_rel, mom_rel_src, etc.
"""

import os, sys, argparse
import numpy as np, pandas as pd, h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------- numerics ----------
def laplacian_spectral_unit(F):
    Ny, Nx = F.shape
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=1.0)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=1.0)
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX*KX + KY*KY
    Fhat = np.fft.fft2(F)
    lap = np.fft.ifft2(-k2 * Fhat).real
    lap[0,0] = 0.0
    return lap

def central_grad_unit(F):
    dFx = 0.5*(np.roll(F, -1, axis=1) - np.roll(F, 1, axis=1))
    dFy = 0.5*(np.roll(F, -1, axis=0) - np.roll(F, 1, axis=0))
    return dFx, dFy

def gaussian1d_kernel(sig, radius=None):
    if sig <= 0: return None
    if radius is None: radius = int(max(3, round(3*sig)))
    v = np.arange(-radius, radius+1, dtype=float)
    k = np.exp(-0.5*(v/sig)**2); k /= k.sum()
    return k

def gaussian_smooth2d(F, sig):
    k = gaussian1d_kernel(sig)
    if k is None: return F
    pad = len(k)//2
    G = F.copy()
    # rows
    for j in range(F.shape[0]):
        row = np.r_[F[j, -pad:], F[j], F[j, :pad]]
        conv = np.convolve(row, k, mode="same")
        G[j, :] = conv[pad:pad+F.shape[1]]
    # cols
    H = G.copy()
    for i in range(F.shape[1]):
        col = np.r_[G[-pad:, i], G[:, i], G[:pad, i]]
        conv = np.convolve(col, k, mode="same")
        H[:, i] = conv[pad:pad+F.shape[0]]
    return H

def sanitize(A, fill=0.0):
    B = A.copy()
    bad = ~np.isfinite(B)
    if bad.any(): B[bad] = fill
    return B

def safe_lnOmega(T, alpha, smooth_px, lnom_clip):
    lnOm = alpha * T
    if smooth_px > 0: lnOm = gaussian_smooth2d(lnOm, smooth_px)
    lnOm = lnOm - float(np.mean(lnOm))
    if lnom_clip and lnom_clip > 0:
        lnOm = np.clip(lnOm, -lnom_clip, lnom_clip)
    return lnOm

def lnOmega_to_Omega(lnOm, omega_floor):
    Om = np.exp(lnOm)
    m = Om.mean()
    if not np.isfinite(m) or m <= 0:
        lnOm = np.clip(lnOm - float(np.mean(lnOm)), -8.0, 8.0)
        Om = np.exp(lnOm); m = Om.mean() if np.isfinite(Om.mean()) and Om.mean() > 0 else 1.0
    Om = Om / m
    Om = np.maximum(Om, omega_floor)
    return sanitize(Om, fill=1.0)

# ---------- robust fit ----------
def direct_fit_robust(R2, rho_phi, rho_m, mask, ridge=1e-8):
    r = R2[mask].ravel()
    a = rho_phi[mask].ravel()
    b = rho_m[mask].ravel()
    finite = np.isfinite(r) & np.isfinite(a) & np.isfinite(b)
    r, a, b = r[finite], a[finite], b[finite]
    if r.size < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    rms_R2 = float(np.sqrt(np.mean(r**2)) + 1e-30)

    A = (a - a.mean()) / (a.std() + 1e-12)
    B = (b - b.mean()) / (b.std() + 1e-12)
    X = np.vstack([A, B, np.ones_like(A)]).T
    y = r - r.mean()

    XtX = X.T @ X
    lam = ridge * (y.var() + 1e-30)
    R = np.diag([lam, lam, 0.0])
    coef = np.linalg.solve(XtX + R, X.T @ y)
    Ahat_s, Bhat_s, Chat = coef.tolist()

    Ahat = Ahat_s / (a.std() + 1e-12)
    Bhat = Bhat_s / (b.std() + 1e-12)
    Chat = Chat + r.mean() - Ahat*a.mean() - Bhat*b.mean()

    fit = Ahat*rho_phi + Bhat*rho_m + Chat
    r_mask = R2[mask].ravel()
    f_mask = fit[mask].ravel()
    finite2 = np.isfinite(r_mask) & np.isfinite(f_mask)
    r_mask, f_mask = r_mask[finite2], f_mask[finite2]
    ss_res = float(np.sum((r_mask - f_mask)**2))
    ss_tot = float(np.sum((r_mask - np.mean(r_mask))**2)) + 1e-30
    R2_score = 1.0 - ss_res/ss_tot
    ein_rms = float(np.sqrt(np.mean((r_mask - f_mask)**2)))
    ein_rel = float(ein_rms / rms_R2)
    return Ahat, Bhat, Chat, ein_rms, ein_rel, R2_score, rms_R2

# ---------- sub-pixel augmentation ----------
def fourier_shift2d(A, dy, dx):
    """Sub-pixel periodic shift by (dy, dx) using Fourier phase ramps."""
    Ny, Nx = A.shape
    ky = 2*np.pi*np.fft.fftfreq(Ny)
    kx = 2*np.pi*np.fft.fftfreq(Nx)
    KX, KY = np.meshgrid(kx, ky)
    phase = np.exp(-1j*(dx*KX + dy*KY))
    return np.fft.ifft2(np.fft.fft2(A)*phase).real

def parse_shifts(spec: str):
    """
    spec example: "(0,0);(0,1);(1,0);(-1,0);(0,-1)" or fractional "(0.33,0)".
    Returns list of (dy, dx) as floats.
    """
    if not spec: return [(0.0, 0.0)]
    items = []
    for part in spec.split(";"):
        part = part.strip()
        if not part: continue
        assert part[0]=="(" and part[-1]==")", "shift must be like (dy,dx)"
        dy,dx = part[1:-1].split(",")
        items.append((float(dy), float(dx)))
    return items

def augment_and_stack(arr_list, shifts, frac=False, jitter_phase=0.0, rng=None):
    """
    Apply each (dy,dx) shift to each array (periodic), stack along new axis.
    If frac=False -> integer np.roll; if frac=True -> Fourier sub-pixel shift.
    jitter_phase: std dev of random Fourier-phase noise (radians), 0=off.
    """
    outs=[]
    if rng is None: rng = np.random.default_rng(42)
    for A in arr_list:
        for (dy,dx) in shifts:
            if frac:
                B = fourier_shift2d(A, dy, dx)
            else:
                B = np.roll(np.roll(A, int(round(dy)), axis=0), int(round(dx)), axis=1)
            if jitter_phase > 0:
                F = np.fft.fft2(B)
                noise = rng.normal(0.0, jitter_phase, size=B.shape)
                F = F * np.exp(1j*noise)
                B = np.fft.ifft2(F).real
            outs.append(B)
    return np.stack(outs, axis=0)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Ensemble Einstein + momentum check (stable, augmented)")
    ap.add_argument("--series_prefix", required=True)
    ap.add_argument("--k_max", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=4.9)
    ap.add_argument("--smooth_px", type=float, default=2.0)
    ap.add_argument("--lnom_clip", type=float, default=3.0)
    ap.add_argument("--omega_floor", type=float, default=1e-3)
    ap.add_argument("--mask_margin", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--lapse", type=float, default=1.0)
    ap.add_argument("--augment_shifts", type=str, default="(0,0);(0,1);(1,0);(-1,0);(0,-1)",
                    help='semicolon-separated list, e.g. "(0,0);(0.33,0);(0,0.33)"')
    ap.add_argument("--augment_frac", action="store_true",
                    help="interpret shifts as fractional pixels via Fourier shift")
    ap.add_argument("--jitter_phase", type=float, default=0.0,
                    help="random Fourier-phase jitter std (radians), e.g. 0.02")
    ap.add_argument("--outdir", default="ensemble_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    shifts = parse_shifts(args.augment_shifts)

    # Load slices
    Ts, rhos_m = [], []
    for k in range(0, args.k_max+1):
        path = f"{args.series_prefix}_t{k}_fields.h5"
        if not os.path.exists(path):
            print("Missing:", path); continue
        with h5py.File(path, "r") as h:
            Ts.append(h["Teff"][:])
            rhos_m.append(h["rho_m"][:])
    if len(Ts) < 2:
        print("Not enough slices."); return

    # Augment & average
    T_stack    = augment_and_stack(Ts, shifts, frac=args.augment_frac, jitter_phase=args.jitter_phase)
    rhom_stack = augment_and_stack(rhos_m, shifts, frac=args.augment_frac, jitter_phase=0.0)
    T_ens      = np.mean(T_stack, axis=0)
    rho_m_ens  = np.mean(rhom_stack, axis=0)

    Ny, Nx = T_ens.shape
    mask = np.ones((Ny, Nx), bool)
    mm = int(args.mask_margin)
    mask[:mm,:] = mask[-mm:,:] = mask[:,:mm] = mask[:,-mm:] = False

    # Averaged fields
    lnOm = safe_lnOmega(T_ens, args.alpha, args.smooth_px, args.lnom_clip)
    Om   = lnOmega_to_Omega(lnOm, args.omega_floor)
    Om2  = Om*Om
    lap_ln = laplacian_spectral_unit(lnOm)
    R2 = sanitize(-2.0 * lap_ln / Om2)
    dTdx, dTdy = central_grad_unit(T_ens)
    rho_phi = sanitize(0.5 * (dTdx*dTdx + dTdy*dTdy) / Om2)

    # Hamiltonian fit
    A,B,C,ein_rms,ein_rel,R2_score,rms_R2 = direct_fit_robust(R2, rho_phi, rho_m_ens, mask)

    # Momentum (stationary proxy from ensemble Ω)
    Kxx = -(Om * ((Om - Om.mean()) / args.dt)) / args.lapse
    Kyy = Kxx.copy()
    ginv = 1.0 / (Om2 + 1e-30)
    K = ginv * (Kxx + Kyy)
    Kxx_c = (ginv*ginv) * Kxx
    Kyy_c = (ginv*ginv) * Kyy
    Sxx = Kxx_c - ginv * K
    Syy = Kyy_c - ginv * K
    Sxy = np.zeros_like(Sxx); Syx = np.zeros_like(Sxx)
    dTxx_dx = 0.5*(np.roll(Sxx, -1, 1) - np.roll(Sxx, 1, 1))
    dTxy_dy = 0.5*(np.roll(Sxy, -1, 0) - np.roll(Sxy, 1, 0))
    dTyx_dx = 0.5*(np.roll(Syx, -1, 1) - np.roll(Syx, 1, 1))
    dTyy_dy = 0.5*(np.roll(Syy, -1, 0) - np.roll(Syy, 1, 0))
    Cx = dTxx_dx + dTxy_dy
    Cy = dTyx_dx + dTyy_dy
    Cnorm = (Om2)*(Cx*Cx + Cy*Cy)
    dKdx, dKdy = central_grad_unit(K)
    gradK_norm = (Om2)*(dKdx*dKdx + dKdy*dKdy)
    mfin = mask & np.isfinite(Cnorm) & np.isfinite(gradK_norm)
    mom_rms_abs = float(np.sqrt(np.mean(Cnorm[mfin])))
    mom_rms_ref = float(np.sqrt(np.mean(gradK_norm[mfin])) + 1e-12)
    mom_rel = float(mom_rms_abs / mom_rms_ref)

    # Source subtraction via first two raw slices (if available)
    Tdot = (Ts[1] - Ts[0]) / args.dt if len(Ts) >= 2 else np.zeros_like(T_ens)
    jx = ginv * Tdot * dTdx
    jy = ginv * Tdot * dTdy
    num = float(np.nansum(Cx[mfin]*jx[mfin] + Cy[mfin]*jy[mfin]))
    den = float(np.nansum(jx[mfin]*jx[mfin] + jy[mfin]*jy[mfin]) + 1e-30)
    c_phi = num / den
    Cx_s = Cx - c_phi * jx
    Cy_s = Cy - c_phi * jy
    Cnorm_s = (Om2)*(Cx_s*Cx_s + Cy_s*Cy_s)
    mom_rms_abs_src = float(np.sqrt(np.nanmean(Cnorm_s[mfin])))
    mom_rel_src = float(mom_rms_abs_src / (np.sqrt(np.nanmean(gradK_norm[mfin])) + 1e-12))

    # Output
    row = dict(A=A,B=B,C=C,ein_rms=ein_rms,ein_rel=ein_rel,rms_R2=rms_R2,
               R2_score=R2_score,mom_rel=mom_rel,c_phi=c_phi,mom_rel_src=mom_rel_src,
               n_slices=len(Ts), n_aug=len(shifts), total_members=len(Ts)*len(shifts))
    df = pd.DataFrame([row])
    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "ensemble_summary.csv")
    df.to_csv(out_csv, index=False)

    with PdfPages(os.path.join(args.outdir, "ensemble_summary.pdf")) as pp:
        fig,ax = plt.subplots(figsize=(7.0,4.6))
        bars = ["ein_rel","mom_rel_src"]
        vals = [ein_rel, mom_rel_src]
        ax.bar(bars, vals)
        ax.set_ylabel("Residual")
        ax.set_title("Ensemble-averaged Einstein & Momentum Residuals")
        for i,v in enumerate(vals):
            ax.text(i, v+0.02, f"{v:.3f}", ha='center')
        plt.tight_layout(); pp.savefig(fig); plt.close(fig)

    print("=== Ensemble Einstein Check (augmented) ===")
    for k,v in row.items():
        try:
            print(f"{k:>16s} : {v:.6g}")
        except Exception:
            print(f"{k:>16s} : {v}")
    print("Saved:", out_csv, "and PDF in", args.outdir)

if __name__ == "__main__":
    main()