#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SE(2) pre-alignment + ensemble ADM Hamiltonian check.

- Detect runs in: <ensemble_dir>/run_???/series_t{0,1}_fields.h5
- For each run:
    * estimate rotation θ via small-angle scan maximizing NCC on Teff(t0)
    * estimate integer shift (dy,dx) via FFT phase correlation
    * warp BOTH T0 and T1 with (θ,dy,dx) using periodic bilinear sampling
- Average aligned fields over runs and fit the full Hamiltonian:
    H_LHS = R(γ) + K^2 - K_ij K^ij  ≈  β ρφ + κ ρm + 2Λ

Outputs in outdir:
  - align_params.csv (θ, dy, dx per run)
  - ensemble_aligned_summary.csv (β, κ, 2Λ, residuals)
  - ensemble_aligned.pdf (quick bar plot of residuals)
"""

import os, sys, glob, argparse
import numpy as np, pandas as pd, h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------- Numerics ----------
def laplacian_spectral_unit(F):
    Ny, Nx = F.shape
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=1.0)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=1.0)
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX*KX + KY*KY
    Fhat = np.fft.fft2(F)
    out = np.fft.ifft2(-k2 * Fhat).real
    out[0,0] = 0.0
    return out

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

# ---------- Periodic rotation (bilinear) ----------
def rotate_periodic_bilinear(F, theta_rad):
    """Rotate F by theta around center with periodic wrap + bilinear sampling."""
    Ny, Nx = F.shape
    y = np.arange(Ny) - Ny/2.0
    x = np.arange(Nx) - Nx/2.0
    X, Y = np.meshgrid(x, y)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    # inverse map (dst -> src)
    Xs = c*X + s*Y + Nx/2.0
    Ys = -s*X + c*Y + Ny/2.0
    # periodic wrap
    X0 = np.floor(Xs).astype(int) % Nx
    Y0 = np.floor(Ys).astype(int) % Ny
    X1 = (X0 + 1) % Nx
    Y1 = (Y0 + 1) % Ny
    ax = Xs - np.floor(Xs)
    ay = Ys - np.floor(Ys)
    # bilinear
    F00 = F[Y0, X0]
    F10 = F[Y0, X1]
    F01 = F[Y1, X0]
    F11 = F[Y1, X1]
    return ( (1-ax)*(1-ay)*F00 + ax*(1-ay)*F10 + (1-ax)*ay*F01 + ax*ay*F11 )

# ---------- Phase correlation (integer shift) ----------
def estimate_shift_int(ref, img):
    """Return (dy,dx) integer shift that best aligns img to ref via phase correlation."""
    F1 = np.fft.fft2(ref)
    F2 = np.fft.fft2(img)
    R = F1 * np.conj(F2)
    denom = np.abs(R) + 1e-30
    cps = R / denom
    corr = np.fft.ifft2(cps).real
    j, i = np.unravel_index(np.argmax(corr), corr.shape)  # (row,col)=(y,x)
    Ny, Nx = ref.shape
    dy = j if j < Ny/2 else j - Ny
    dx = i if i < Nx/2 else i - Nx
    return int(dy), int(dx)

def apply_shift_int(F, dy, dx):
    return np.roll(np.roll(F, dy, axis=0), dx, axis=1)

# ---------- Robust fit ----------
def robust_fit(lhs, rphi, rm, mask, ridge=1e-8):
    y = lhs[mask].ravel()
    a = rphi[mask].ravel()
    b = rm[mask].ravel()
    finite = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[finite], a[finite], b[finite]
    if y.size < 10:
        return (np.nan,)*7
    A = (a - a.mean())/(a.std()+1e-12)
    B = (b - b.mean())/(b.std()+1e-12)
    X = np.vstack([A, B, np.ones_like(A)]).T
    y0 = y - y.mean()
    XtX = X.T @ X
    lam = ridge * (y0.var() + 1e-30)
    R = np.diag([lam, lam, 0.0])
    coef = np.linalg.solve(XtX + R, X.T @ y0)
    Ahat_s, Bhat_s, Chat = coef.tolist()
    beta  = Ahat_s/(a.std()+1e-12)
    kappa = Bhat_s/(b.std()+1e-12)
    twoLambda = Chat + y.mean() - beta*a.mean() - kappa*b.mean()
    fit = beta*rphi + kappa*rm + twoLambda
    f = fit[mask].ravel()[finite]
    ss_res = float(np.sum((y - f)**2))
    ss_tot = float(np.sum((y - y.mean())**2) + 1e-30)
    R2 = 1.0 - ss_res/ss_tot
    rms_lhs = float(np.sqrt(np.mean(y**2)) + 1e-30)
    rms_res = float(np.sqrt(np.mean((y - f)**2)))
    rel_res = float(rms_res / rms_lhs)
    return beta, kappa, twoLambda, R2, rms_res, rel_res, rms_lhs

# ---------- Hamiltonian from aligned pair ----------
def hamiltonian_from_pair(T0, T1, rho_m, alpha, smooth_px, lnom_clip, omega_floor, dt, lapse, mask_margin):
    Ny, Nx = T0.shape
    mask = np.ones((Ny, Nx), bool)
    mm = int(mask_margin)
    mask[:mm,:] = mask[-mm:,:] = mask[:,:mm] = mask[:,-mm:] = False

    lnOm0 = safe_lnOmega(T0, alpha, smooth_px, lnom_clip)
    Om0   = lnOmega_to_Omega(lnOm0, omega_floor)
    Om02  = Om0*Om0
    R2    = sanitize(-2.0 * laplacian_spectral_unit(lnOm0) / Om02)

    dTx, dTy = central_grad_unit(T0)
    rho_phi  = sanitize(0.5 * (dTx*dTx + dTy*dTy) / Om02)

    lnOm1 = safe_lnOmega(T1, alpha, smooth_px, lnom_clip)
    Om1   = lnOmega_to_Omega(lnOm1, omega_floor)
    dOm_dt = (Om1 - Om0) / dt

    N = lapse
    Kxx = -(1.0/(2*N)) * (2*Om0*dOm_dt)  # = -(Ω \dotΩ)/N
    Kyy = Kxx.copy()

    ginv = 1.0/(Om02 + 1e-30)
    Kx_x = ginv*Kxx; Ky_y = ginv*Kyy
    K = Kx_x + Ky_y
    KijKij = (ginv*ginv)*(Kxx*Kxx + Kyy*Kyy)
    H_LHS = sanitize(R2 + K*K - KijKij)
    return H_LHS, rho_phi, rho_m, mask, R2

# ---------- Alignment helpers ----------
def ncc(a, b):
    a0 = a - a.mean(); b0 = b - b.mean()
    denom = (np.sqrt((a0*a0).mean()) * np.sqrt((b0*b0).mean()) + 1e-30)
    return float(np.mean(a0*b0)/denom)

def estimate_rotation(ref, img, deg_min, deg_max, deg_step, smooth_px):
    best_deg, best_score = 0.0, -1e9
    Fref = gaussian_smooth2d(ref, smooth_px) if smooth_px>0 else ref
    for d in np.arange(deg_min, deg_max+1e-9, deg_step):
        Rimg = rotate_periodic_bilinear(img, np.deg2rad(d))
        RimgS = gaussian_smooth2d(Rimg, smooth_px) if smooth_px>0 else Rimg
        score = ncc(Fref, RimgS)
        if score > best_score:
            best_score, best_deg = score, d
    return best_deg, best_score

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser("SE(2) pre-alignment + ensemble ADM Hamiltonian")
    ap.add_argument("--ensemble_dir", required=True, help="Folder with run_XXX/series_t{0,1}_fields.h5")
    ap.add_argument("--alpha", type=float, default=4.9)
    ap.add_argument("--smooth_px", type=float, default=2.0)
    ap.add_argument("--lnom_clip", type=float, default=3.0)
    ap.add_argument("--omega_floor", type=float, default=1e-3)
    ap.add_argument("--mask_margin", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--lapse", type=float, default=1.0)
    ap.add_argument("--deg_min", type=float, default=-5.0)
    ap.add_argument("--deg_max", type=float, default=5.0)
    ap.add_argument("--deg_step", type=float, default=0.25)
    ap.add_argument("--outdir", default="ensemble_aligned")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs = sorted(glob.glob(os.path.join(args.ensemble_dir, "run_*")))
    runs = [r for r in runs if os.path.exists(os.path.join(r,"series_t0_fields.h5")) and
                             os.path.exists(os.path.join(r,"series_t1_fields.h5"))]
    if not runs:
        print("No runs found."); return

    # Reference = first run's T0
    with h5py.File(os.path.join(runs[0], "series_t0_fields.h5"), "r") as h0:
        Tref = h0["Teff"][:]

    align_rows = []
    aligned_T0s, aligned_T1s, aligned_rhom = [], [], []

    for r in runs:
        with h5py.File(os.path.join(r, "series_t0_fields.h5"), "r") as h0, \
             h5py.File(os.path.join(r, "series_t1_fields.h5"), "r") as h1:
            T0 = h0["Teff"][:]
            T1 = h1["Teff"][:]
            rhom = h0["rho_m"][:]

        # estimate rotation
        theta_deg, score = estimate_rotation(Tref, T0, args.deg_min, args.deg_max, args.deg_step, smooth_px=args.smooth_px/2.0)
        T0r = rotate_periodic_bilinear(T0, np.deg2rad(theta_deg))
        T1r = rotate_periodic_bilinear(T1, np.deg2rad(theta_deg))

        # estimate integer shift on rotated pair
        dy, dx = estimate_shift_int(Tref, T0r)
        T0a = apply_shift_int(T0r, dy, dx)
        T1a = apply_shift_int(T1r, dy, dx)

        aligned_T0s.append(T0a)
        aligned_T1s.append(T1a)
        aligned_rhom.append(rhom)

        align_rows.append(dict(run=os.path.basename(r), theta_deg=theta_deg, dy=dy, dx=dx, ncc=score))

    # Ensemble averages (aligned)
    T0_mean = np.mean(np.stack(aligned_T0s, axis=0), axis=0)
    T1_mean = np.mean(np.stack(aligned_T1s, axis=0), axis=0)
    rm_mean = np.mean(np.stack(aligned_rhom, axis=0), axis=0)

    # Build Hamiltonian from aligned ensemble mean
    H_LHS, rphi, rm, mask, R2 = hamiltonian_from_pair(
        T0_mean, T1_mean, rm_mean,
        args.alpha, args.smooth_px, args.lnom_clip, args.omega_floor,
        args.dt, args.lapse, args.mask_margin
    )
    beta, kappa, twoLam, R2fit, rms_res, rel_res, rms_lhs = robust_fit(H_LHS, rphi, rm, mask)

    # Save outputs
    pd.DataFrame(align_rows).to_csv(os.path.join(args.outdir, "align_params.csv"), index=False)
    summ = pd.DataFrame([dict(
        beta=beta, kappa=kappa, twoLambda=twoLam,
        R2_score=R2fit, rms_res=rms_res, rel_res=rel_res, rms_LHS=rms_lhs,
        n_runs=len(runs),
        deg_scan=f"[{args.deg_min},{args.deg_max}] step {args.deg_step}"
    )])
    summ.to_csv(os.path.join(args.outdir, "ensemble_aligned_summary.csv"), index=False)

    with PdfPages(os.path.join(args.outdir, "ensemble_aligned.pdf")) as pp:
        fig, ax = plt.subplots(figsize=(6.2,3.8))
        bars = ["ein_rel (aligned)"]
        vals = [rel_res]
        ax.bar(bars, vals)
        ax.set_ylim(0, max(1.0, vals[0]+0.1))
        ax.set_ylabel("Relative residual")
        ax.set_title("ADM Hamiltonian (aligned ensemble)")
        for i,v in enumerate(vals):
            ax.text(i, v+0.02, f"{v:.3f}", ha='center')
        plt.tight_layout(); pp.savefig(fig); plt.close(fig)

    print("=== SE(2) aligned ensemble (ADM) ===")
    for k,v in dict(beta=beta, kappa=kappa, twoLambda=twoLam,
                    R2_score=R2fit, rms_res=rms_res, rel_res=rel_res, rms_LHS=rms_lhs).items():
        try: print(f"{k:>10s} : {v:.6g}")
        except: print(f"{k:>10s} : {v}")
    print("Saved CSV and PDF to:", args.outdir)

if __name__ == "__main__":
    main()