#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADM Hamiltonian on an ensemble mean with motion (shift vector β):
    K_ij = -(2N)^(-1) [ ∂_t γ_ij - L_β γ_ij ],  γ_ij = Ω^2 δ_ij

- Loads runs: <ensemble_dir>/run_???/series_t{0,1}_fields.h5
- Optional small SE(2) alignment (rotation + integer shift) for robustness
- Estimates a GLOBAL β = (βx, βy) and μ by ridge least-squares from:
      ∂_t T ≈ -β·∇T + μ ΔT
- Builds Ω from Teff via lnΩ = α T (with smoothing, clipping, floor)
- Forms full Hamiltonian LHS = R(γ) + K^2 - K_ij K^ij
- Fits: LHS ≈ β_fit ρφ + κ ρm + 2Λ   (robust ridge)
- Writes CSV + a tiny PDF summary

Author: A. J. Newland, 2025
"""

import os, sys, glob, argparse
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
    for j in range(F.shape[0]):
        row = np.r_[F[j, -pad:], F[j], F[j, :pad]]
        conv = np.convolve(row, k, mode="same")
        G[j, :] = conv[pad:pad+F.shape[1]]
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

# ---------- simple SE(2) tools ----------
def rotate_periodic_bilinear(F, theta_rad):
    Ny, Nx = F.shape
    y = np.arange(Ny) - Ny/2.0
    x = np.arange(Nx) - Nx/2.0
    X, Y = np.meshgrid(x, y)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    Xs = c*X + s*Y + Nx/2.0
    Ys = -s*X + c*Y + Ny/2.0
    X0 = np.floor(Xs).astype(int) % Nx
    Y0 = np.floor(Ys).astype(int) % Ny
    X1 = (X0 + 1) % Nx
    Y1 = (Y0 + 1) % Ny
    ax = Xs - np.floor(Xs)
    ay = Ys - np.floor(Ys)
    F00 = F[Y0, X0]; F10 = F[Y0, X1]; F01 = F[Y1, X0]; F11 = F[Y1, X1]
    return ( (1-ax)*(1-ay)*F00 + ax*(1-ay)*F10 + (1-ax)*ay*F01 + ax*ay*F11 )

def estimate_shift_int(ref, img):
    F1 = np.fft.fft2(ref)
    F2 = np.fft.fft2(img)
    R = F1 * np.conj(F2)
    cps = R / (np.abs(R) + 1e-30)
    corr = np.fft.ifft2(cps).real
    j, i = np.unravel_index(np.argmax(corr), corr.shape)
    Ny, Nx = ref.shape
    dy = j if j < Ny/2 else j - Ny
    dx = i if i < Nx/2 else i - Nx
    return int(dy), int(dx)

def apply_shift_int(F, dy, dx):
    return np.roll(np.roll(F, dy, axis=0), dx, axis=1)

def ncc(a, b):
    a0 = a - a.mean(); b0 = b - b.mean()
    den = (np.sqrt((a0*a0).mean()) * np.sqrt((b0*b0).mean()) + 1e-30)
    return float(np.mean(a0*b0)/den)

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

# ---------- fit helpers ----------
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
    f = fit
    y_fit = f[mask].ravel()[np.isfinite(f[mask].ravel())][:len(y)]  # guard
    # recompute properly
    fmask = (np.isfinite(f) & mask)
    yy = lhs[fmask].ravel()
    ff = f[fmask].ravel()
    ss_res = float(np.sum((yy - ff)**2))
    ss_tot = float(np.sum((yy - yy.mean())**2) + 1e-30)
    R2 = 1.0 - ss_res/ss_tot
    rms_lhs = float(np.sqrt(np.mean(yy**2)) + 1e-30)
    rms_res = float(np.sqrt(np.mean((yy - ff)**2)))
    rel_res = float(rms_res / rms_lhs)
    return beta, kappa, twoLambda, R2, rms_res, rel_res, rms_lhs

def fit_global_beta_mu(T0, T1, dt, ridge=1e-6):
    dTdt = (T1 - T0)/dt
    dTx, dTy = central_grad_unit(T0)
    lapT = laplacian_spectral_unit(T0)
    # y = dTdt ; features = [dTx, dTy, -lapT] with ridge
    y = dTdt.ravel()
    X = np.vstack([dTx.ravel(), dTy.ravel(), (-lapT).ravel()]).T
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[finite]; y = y[finite]
    XtX = X.T @ X
    lam = ridge * (y.var() + 1e-30)
    coef = np.linalg.solve(XtX + lam*np.eye(3), X.T @ y)
    bx, by, mu = coef.tolist()
    return bx, by, mu

# ---------- Hamiltonian with Lie derivative ----------
def hamiltonian_LHS_with_shift(T0, T1, rho_m, alpha, smooth_px, lnom_clip, omega_floor,
                               dt, lapse, mask_margin, beta_x, beta_y):
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

    # Lie derivative L_β γ_ij for γ_ij = Ω^2 δ_ij
    # ∂i β^k
    b = np.dstack([np.full_like(T0, beta_x), np.full_like(T0, beta_y)])  # constant β field
    dbx_dx, dbx_dy = central_grad_unit(b[...,0])
    dby_dx, dby_dy = central_grad_unit(b[...,1])
    # β·∇(Ω^2)
    gradOm2_x, gradOm2_y = central_grad_unit(Om02)
    beta_dot_grad_Om2 = beta_x*gradOm2_x + beta_y*gradOm2_y

    # Components of Lβ γ
    Lxx = beta_dot_grad_Om2 + 2*Om02*dbx_dx
    Lyy = beta_dot_grad_Om2 + 2*Om02*dby_dy
    Lxy = Om02*(dbx_dy + dby_dx)  # off-diagonal

    N = lapse
    # ∂t γ_ij = 2 Ω ∂tΩ δ_ij
    dgam_xx = 2*Om0*dOm_dt
    dgam_yy = dgam_xx
    dgam_xy = np.zeros_like(Om0)

    Kxx = -(1.0/(2*N)) * (dgam_xx - Lxx)
    Kyy = -(1.0/(2*N)) * (dgam_yy - Lyy)
    Kxy = -(1.0/(2*N)) * (dgam_xy - Lxy)

    ginv = 1.0/(Om02 + 1e-30)
    # traces and contractions in 2D
    K = ginv*(Kxx + Kyy)                         # trace
    KijKij = (ginv*ginv)*(Kxx*Kxx + Kyy*Kyy + 2*Kxy*Kxy)
    H_LHS = sanitize(R2 + K*K - KijKij)

    return H_LHS, rho_phi, rho_m, mask, R2

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("ADM ensemble with shift (β) on aligned ensemble mean")
    ap.add_argument("--ensemble_dir", required=True)
    ap.add_argument("--alpha", type=float, default=4.9)
    ap.add_argument("--smooth_px", type=float, default=2.0)
    ap.add_argument("--lnom_clip", type=float, default=3.0)
    ap.add_argument("--omega_floor", type=float, default=1e-3)
    ap.add_argument("--mask_margin", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--lapse", type=float, default=1.0)
    ap.add_argument("--do_align", action="store_true", help="Apply small SE(2) pre-alignment")
    ap.add_argument("--deg_min", type=float, default=-5.0)
    ap.add_argument("--deg_max", type=float, default=5.0)
    ap.add_argument("--deg_step", type=float, default=0.25)
    ap.add_argument("--outdir", default="ensemble_with_shift")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs = sorted(glob.glob(os.path.join(args.ensemble_dir, "run_*")))
    runs = [r for r in runs if os.path.exists(os.path.join(r,"series_t0_fields.h5")) and
                             os.path.exists(os.path.join(r,"series_t1_fields.h5"))]
    if not runs:
        print("No runs found."); return

    # Reference for optional alignment
    with h5py.File(os.path.join(runs[0], "series_t0_fields.h5"), "r") as h0:
        Tref = h0["Teff"][:]

    aligned_T0s, aligned_T1s, aligned_rhom = [], [], []
    align_rows = []

    for r in runs:
        with h5py.File(os.path.join(r, "series_t0_fields.h5"), "r") as h0, \
             h5py.File(os.path.join(r, "series_t1_fields.h5"), "r") as h1:
            T0 = h0["Teff"][:]
            T1 = h1["Teff"][:]
            rm = h0["rho_m"][:]

        if args.do_align:
            theta_deg, _ = estimate_rotation(Tref, T0, args.deg_min, args.deg_max, args.deg_step, smooth_px=args.smooth_px/2.0)
            T0r = rotate_periodic_bilinear(T0, np.deg2rad(theta_deg))
            T1r = rotate_periodic_bilinear(T1, np.deg2rad(theta_deg))
            dy, dx = estimate_shift_int(Tref, T0r)
            T0a = apply_shift_int(T0r, dy, dx)
            T1a = apply_shift_int(T1r, dy, dx)
            align_rows.append(dict(run=os.path.basename(r), theta_deg=theta_deg, dy=dy, dx=dx))
        else:
            T0a, T1a = T0, T1

        aligned_T0s.append(T0a)
        aligned_T1s.append(T1a)
        aligned_rhom.append(rm)

    # Ensemble means
    T0_mean = np.mean(np.stack(aligned_T0s, axis=0), axis=0)
    T1_mean = np.mean(np.stack(aligned_T1s, axis=0), axis=0)
    rm_mean = np.mean(np.stack(aligned_rhom, axis=0), axis=0)

    # --- estimate global β and μ from advection–diffusion on ensemble mean ---
    bx, by, mu = fit_global_beta_mu(T0_mean, T1_mean, args.dt, ridge=1e-6)

    # --- full Hamiltonian with Lie derivative ---
    H_LHS, rphi, rm, mask, R2 = hamiltonian_LHS_with_shift(
        T0_mean, T1_mean, rm_mean,
        args.alpha, args.smooth_px, args.lnom_clip, args.omega_floor,
        args.dt, args.lapse, args.mask_margin,
        beta_x=bx, beta_y=by
    )

    beta_fit, kappa_fit, twoLam, R2fit, rms_res, rel_res, rms_lhs = robust_fit(H_LHS, rphi, rm, mask)

    # --- save outputs ---
    if align_rows:
        pd.DataFrame(align_rows).to_csv(os.path.join(args.outdir, "align_params.csv"), index=False)

    pd.DataFrame([dict(
        beta_scalar=beta_fit, kappa=kappa_fit, twoLambda=twoLam,
        R2_score=R2fit, rms_res=rms_res, rel_res=rel_res, rms_LHS=rms_lhs,
        bx=bx, by=by, mu_est=mu, n_runs=len(runs)
    )]).to_csv(os.path.join(args.outdir, "ensemble_with_shift_summary.csv"), index=False)

    with PdfPages(os.path.join(args.outdir, "ensemble_with_shift.pdf")) as pp:
        fig, ax = plt.subplots(figsize=(6.2,3.6))
        ax.bar(["rel_res"], [rel_res])
        ax.set_ylim(0, max(1.0, rel_res+0.1))
        ax.set_ylabel("Relative residual")
        ax.set_title("ADM Hamiltonian (ensemble mean with shift β)")
        ax.text(0, rel_res+0.02, f"{rel_res:.3f}", ha="center")
        plt.tight_layout(); pp.savefig(fig); plt.close(fig)

    print("=== ADM with shift (ensemble mean) ===")
    print(f"  Estimated beta (βx, βy) = ({bx:.4g}, {by:.4g}),  mu ≈ {mu:.4g}")
    print(f"  rel_res = {rel_res:.6f}   (R2={R2fit:.4f}, rms_LHS={rms_lhs:.3e})")
    print("Saved CSV/PDF to:", args.outdir)

if __name__ == "__main__":
    main()