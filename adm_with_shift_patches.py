#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-run ADM Hamiltonian with spatially varying shift β(x,y), using t0 & t2 (central difference).

For each run folder containing:
  series_t0_fields.h5, series_t2_fields.h5  (and rho_m inside t0 h5)
we compute:
  - Ω from Teff with smoothing/clipping (NO per-slice mean subtraction; NO renormalization)
  - R(γ) from lnΩ via spectral Laplacian
  - β(x,y) from advection–diffusion fit on patches:
        (T2 - T0)/(2 dt)  ≈  -β·∇T0 + μ ΔT0
    (ridge LS), then Gaussian-smoothed
  - K_ij with Lie derivative term L_β γ_ij
  - Fit H_LHS ≈ β_fit ρφ + κ ρm + 2Λ  (ridge LS, robust)

Outputs (per run):
  adm_t0t2_patches/summary.csv   (β_fit, κ, 2Λ, rel_res, R², β stats)
  adm_t0t2_patches/beta_fields.h5 (beta_x, beta_y raw+smoothed, μ_raw)
  adm_t0t2_patches/preview.pdf    (bars + |β| image)

Author: A. J. Newland, 2025
"""

import os, sys, argparse, h5py, numpy as np, pandas as pd
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

# ---------- CRITICAL FIXES ----------
def safe_lnOmega(T, alpha, smooth_px, lnom_clip):
    """
    Build lnΩ = α T with optional smoothing and clipping.
    DO NOT subtract the per-slice mean (keep global expansion mode).
    """
    lnOm = alpha * T
    if smooth_px > 0:
        lnOm = gaussian_smooth2d(lnOm, smooth_px)
    if lnom_clip and lnom_clip > 0:
        lnOm = np.clip(lnOm, -lnom_clip, lnom_clip)
    return lnOm

def lnOmega_to_Omega(lnOm, omega_floor):
    """
    DO NOT renormalize Ω by its slice mean.
    Keep absolute scale; only apply a floor.
    """
    Om = np.exp(lnOm)
    Om = np.maximum(Om, omega_floor)
    return sanitize(Om, fill=1.0)

# ---------- patchwise β fit ----------
def fit_beta_patches(T0, T2, dt, patch, ridge=1e-6):
    Ny, Nx = T0.shape
    dTdt = (T2 - T0) / (2.0*dt)         # central difference
    dTx, dTy = central_grad_unit(T0)
    lapT = laplacian_spectral_unit(T0)

    bx = np.zeros_like(T0); by = np.zeros_like(T0); mu = np.zeros_like(T0)

    for y0 in range(0, Ny, patch):
        for x0 in range(0, Nx, patch):
            ys = slice(y0, min(y0+patch, Ny))
            xs = slice(x0, min(x0+patch, Nx))
            Y = dTdt[ys, xs].ravel()
            X = np.vstack([ dTx[ys, xs].ravel(),
                            dTy[ys, xs].ravel(),
                           -lapT[ys, xs].ravel() ]).T   # y ≈ -β·∇T + μ ΔT
            finite = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
            X = X[finite]; Y = Y[finite]
            if Y.size < 10:
                continue
            XtX = X.T @ X
            lam = ridge * (Y.var() + 1e-30)
            coef = np.linalg.solve(XtX + lam*np.eye(3), X.T @ Y)
            bxi, byi, mui = coef.tolist()
            bx[ys, xs] = bxi
            by[ys, xs] = byi
            mu[ys, xs] = mui

    return bx, by, mu

# ---------- ADM pieces ----------
def hamiltonian_with_beta_xy(T0, T2, rho_m, alpha, smooth_px, lnom_clip, omega_floor,
                             dt_eff, lapse, mask_margin, beta_x, beta_y):
    Ny, Nx = T0.shape
    mask = np.ones((Ny, Nx), bool)
    mm = int(mask_margin)
    mask[:mm,:] = mask[-mm:,:] = mask[:,:mm] = mask[:,-mm:] = False

    # build Ω(t0)
    lnOm0 = safe_lnOmega(T0, alpha, smooth_px, lnom_clip)
    Om0   = lnOmega_to_Omega(lnOm0, omega_floor)
    Om02  = Om0*Om0

    # intrinsic curvature R(γ) in 2D conformal gauge: R = -2 Δ lnΩ / Ω^2
    R2    = sanitize(-2.0 * laplacian_spectral_unit(lnOm0) / Om02)

    # field energy proxy
    dTx, dTy = central_grad_unit(T0)
    rho_phi  = sanitize(0.5 * (dTx*dTx + dTy*dTy) / Om02)

    # Ω(t2)
    lnOm2 = safe_lnOmega(T2, alpha, smooth_px, lnom_clip)
    Om2   = lnOmega_to_Omega(lnOm2, omega_floor)

    # time derivative of metric γ_ij = Ω^2 δ_ij
    # central difference over effective Δt = 2*dt: (Ω2^2 - Ω0^2)/Δt
    dgam_xx = (Om2*Om2 - Om02) / dt_eff
    dgam_yy = dgam_xx.copy()
    dgam_xy = np.zeros_like(Om0)

    # Lie derivative L_β γ with γ_ij = Ω^2 δ_ij
    gradOm2_x, gradOm2_y = central_grad_unit(Om02)
    dbx_dx, dbx_dy = central_grad_unit(beta_x)
    dby_dx, dby_dy = central_grad_unit(beta_y)
    beta_dot_grad_Om2 = beta_x*gradOm2_x + beta_y*gradOm2_y

    Lxx = beta_dot_grad_Om2 + 2*Om02*dbx_dx
    Lyy = beta_dot_grad_Om2 + 2*Om02*dby_dy
    Lxy = Om02*(dbx_dy + dby_dx)

    N = lapse
    Kxx = -(1.0/(2*N)) * (dgam_xx - Lxx)
    Kyy = -(1.0/(2*N)) * (dgam_yy - Lyy)
    Kxy = -(1.0/(2*N)) * (dgam_xy - Lxy)

    ginv = 1.0/(Om02 + 1e-30)
    K = ginv*(Kxx + Kyy)
    KijKij = (ginv*ginv)*(Kxx*Kxx + Kyy*Kyy + 2*Kxy*Kxy)
    H_LHS = sanitize(R2 + K*K - KijKij)

    return H_LHS, rho_phi, rho_m, mask, R2

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
    coef = np.linalg.solve(XtX + lam*np.eye(3), X.T @ y0)
    Ahat_s, Bhat_s, Chat = coef.tolist()
    beta  = Ahat_s/(a.std()+1e-12)
    kappa = Bhat_s/(b.std()+1e-12)
    twoLambda = Chat + y.mean() - beta*a.mean() - kappa*b.mean()

    fit = beta*rphi + kappa*rm + twoLambda
    fmask = (np.isfinite(fit) & mask)
    yy = lhs[fmask].ravel()
    ff = fit[fmask].ravel()
    ss_res = float(np.sum((yy - ff)**2))
    ss_tot = float(np.sum((yy - yy.mean())**2) + 1e-30)
    R2 = 1.0 - ss_res/ss_tot
    rms_lhs = float(np.sqrt(np.mean(yy**2)) + 1e-30)
    rms_res = float(np.sqrt(np.mean((yy - ff)**2)))
    rel_res = float(rms_res / rms_lhs)
    return beta, kappa, twoLambda, R2, rms_res, rel_res, rms_lhs

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Per-run ADM with β(x,y) patches using t0 & t2 (NO Ω renorm)")
    ap.add_argument("--run_dir", required=True, help="Folder with series_t0_fields.h5 and series_t2_fields.h5")
    ap.add_argument("--alpha", type=float, default=4.9)
    ap.add_argument("--smooth_px", type=float, default=1.4)
    ap.add_argument("--lnom_clip", type=float, default=2.5)
    ap.add_argument("--omega_floor", type=float, default=1e-3)
    ap.add_argument("--mask_margin", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--lapse", type=float, default=1.0)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--beta_ridge", type=float, default=1e-6)
    ap.add_argument("--beta_smooth_px", type=float, default=1.0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    t0_h5 = os.path.join(args.run_dir, "series_t0_fields.h5")
    t2_h5 = os.path.join(args.run_dir, "series_t2_fields.h5")
    if not (os.path.exists(t0_h5) and os.path.exists(t2_h5)):
        print("Missing t0/t2 H5 in:", args.run_dir); sys.exit(1)

    if args.outdir is None:
        args.outdir = os.path.join(args.run_dir, "adm_t0t2_patches")
    os.makedirs(args.outdir, exist_ok=True)

    with h5py.File(t0_h5, "r") as h0, h5py.File(t2_h5, "r") as h2:
        T0 = h0["Teff"][:]
        T2 = h2["Teff"][:]
        rho_m = h0["rho_m"][:]

    # Estimate β(x,y) on patches, then smooth
    bx_raw, by_raw, mu_raw = fit_beta_patches(T0, T2, args.dt, args.patch, ridge=args.beta_ridge)
    bx = gaussian_smooth2d(bx_raw, args.beta_smooth_px) if args.beta_smooth_px>0 else bx_raw
    by = gaussian_smooth2d(by_raw, args.beta_smooth_px) if args.beta_smooth_px>0 else by_raw

    # Hamiltonian with Lie derivative using β(x,y)
    dt_eff = 2.0*args.dt
    H_LHS, rphi, rm, mask, R2_field = hamiltonian_with_beta_xy(
        T0, T2, rho_m,
        args.alpha, args.smooth_px, args.lnom_clip, args.omega_floor,
        dt_eff, args.lapse, args.mask_margin,
        bx, by
    )

    beta_fit, kappa_fit, twoLam, R2fit, rms_res, rel_res, rms_lhs = robust_fit(H_LHS, rphi, rm, mask)

    # save outputs
    with h5py.File(os.path.join(args.outdir, "beta_fields.h5"), "w") as w:
        w.create_dataset("beta_x", data=bx)
        w.create_dataset("beta_y", data=by)
        w.create_dataset("beta_x_raw", data=bx_raw)
        w.create_dataset("beta_y_raw", data=by_raw)
        w.create_dataset("mu_raw", data=mu_raw)

    pd.DataFrame([dict(
        beta_scalar=beta_fit, kappa=kappa_fit, twoLambda=twoLam,
        R2_score=R2fit, rms_res=rms_res, rel_res=rel_res, rms_LHS=rms_lhs,
        bx_mean=float(np.mean(bx)), by_mean=float(np.mean(by)),
        bx_std=float(np.std(bx)),  by_std=float(np.std(by)),
        patch=args.patch, beta_ridge=args.beta_ridge, beta_smooth_px=args.beta_smooth_px
    )]).to_csv(os.path.join(args.outdir, "summary.csv"), index=False)

    with PdfPages(os.path.join(args.outdir, "preview.pdf")) as pp:
        fig, ax = plt.subplots(figsize=(6.2,3.6))
        ax.bar(["rel_res","R2"], [rel_res, max(0.0,R2fit)])
        ax.set_ylim(0, max(1.0, rel_res+0.1))
        ax.set_title("ADM (t0→t2) with β(x,y) patches — NO Ω renorm")
        for i,v in enumerate([rel_res, max(0.0,R2fit)]):
            ax.text(i, v+0.02, f"{v:.3f}", ha="center")
        plt.tight_layout(); pp.savefig(fig); plt.close(fig)

        mag = np.sqrt(bx*bx + by*by)
        fig, ax = plt.subplots(figsize=(5,4))
        im = ax.imshow(mag, origin="lower")
        ax.set_title("|β(x,y)| (smoothed)")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout(); pp.savefig(fig); plt.close(fig)

    print("=== ADM t0→t2 with β(x,y) patches (NO Ω renorm) ===")
    print(f" rel_res={rel_res:.6f}  R2={R2fit:.4f}  rms_LHS={rms_lhs:.3e}")
    print("Saved:", args.outdir)

if __name__ == "__main__":
    main()