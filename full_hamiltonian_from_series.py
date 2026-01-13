#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full ADM Hamiltonian check from real time pairs:
   H_LHS = R(γ) + K^2 - K_ij K^ij
fit H_LHS = β ρ_phi + κ ρ_m + 2Λ   (robust)

Consistent numerics: unit-box spectral Laplacian, clipped lnΩ, Ω floor, masks.

Usage (example):
  python full_hamiltonian_from_series.py ^
    --series_prefix "...\FINAL_RUN\series\torus_series_stable" ^
    --k_max 6 --dt 0.08 --lapse 1.0 ^
    --alpha 4.9 --smooth_px 2.0 --lnom_clip 3.0 --omega_floor 1e-3 ^
    --mask_margin 8 ^
    --outdir "...\FINAL_RUN\full_hamiltonian_pairs"
"""

import os, sys, argparse
import numpy as np, pandas as pd, h5py

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

# ---------- robust linear fit ----------
def robust_fit(lhs, rphi, rm, mask, ridge=1e-8):
    """
    Returns: beta, kappa, twoLambda, R2, rms_res, rel_res, rms_lhs, fit
    On insufficient data: returns NaNs and fit=None (keeps unpacking stable).
    """
    y = lhs[mask].ravel()
    a = rphi[mask].ravel()
    b = rm[mask].ravel()
    finite = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[finite], a[finite], b[finite]
    if y.size < 10:
        return (np.nan,)*7 + (None,)  # beta,kappa,2Λ,R2,rms_res,rel_res,rms_lhs,fit(None)

    # standardize features
    A = (a - a.mean())/(a.std()+1e-12)
    B = (b - b.mean())/(b.std()+1e-12)
    X = np.vstack([A, B, np.ones_like(A)]).T
    y0 = y - y.mean()

    XtX = X.T @ X
    lam = ridge * (y0.var() + 1e-30)
    R = np.diag([lam, lam, 0.0])
    coef = np.linalg.solve(XtX + R, X.T @ y0)
    Ahat_s, Bhat_s, Chat = coef.tolist()

    Astd = a.std() + 1e-12
    Bstd = b.std() + 1e-12
    beta  = Ahat_s / Astd
    kappa = Bhat_s / Bstd
    twoLambda = Chat + y.mean() - beta*a.mean() - kappa*b.mean()

    fit = beta*rphi + kappa*rm + twoLambda

    f = fit[mask].ravel()[finite]
    ss_res = float(np.sum((y - f)**2))
    ss_tot = float(np.sum((y - y.mean())**2)) + 1e-30
    R2 = 1.0 - ss_res/ss_tot
    rms_lhs = float(np.sqrt(np.mean(y**2)) + 1e-30)
    rms_res = float(np.sqrt(np.mean((y - f)**2)))
    rel_res = float(rms_res / rms_lhs)
    return beta, kappa, twoLambda, R2, rms_res, rel_res, rms_lhs, fit

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Full ADM Hamiltonian from real time pairs")
    ap.add_argument("--series_prefix", required=True)
    ap.add_argument("--k_max", type=int, required=True)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--lapse", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=4.9)
    ap.add_argument("--smooth_px", type=float, default=2.0)
    ap.add_argument("--lnom_clip", type=float, default=3.0)
    ap.add_argument("--omega_floor", type=float, default=1e-3)
    ap.add_argument("--mask_margin", type=int, default=8)
    ap.add_argument("--outdir", default="full_hamiltonian_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows=[]
    for k in range(args.k_max):
        f0 = f"{args.series_prefix}_t{k}_fields.h5"
        f1 = f"{args.series_prefix}_t{k+1}_fields.h5"
        if not (os.path.exists(f0) and os.path.exists(f1)):
            print("Skipping missing pair:", f0, f1); continue

        with h5py.File(f0, "r") as h0, h5py.File(f1, "r") as h1:
            T0 = h0["Teff"][:]; rho_m = h0["rho_m"][:]
            T1 = h1["Teff"][:]

        Ny, Nx = T0.shape
        mask = np.ones((Ny, Nx), bool)
        mm = int(args.mask_margin)
        mask[:mm,:] = mask[-mm:,:] = mask[:,:mm] = mask[:,-mm:] = False

        # Build Ω, R(γ) at t0
        lnOm0 = safe_lnOmega(T0, args.alpha, args.smooth_px, args.lnom_clip)
        Om0   = lnOmega_to_Omega(lnOm0, args.omega_floor)
        Om02  = Om0*Om0
        R2    = sanitize(-2.0 * laplacian_spectral_unit(lnOm0) / Om02)

        # Gradient energy ρ_phi at t0
        dTx, dTy = central_grad_unit(T0)
        rho_phi  = sanitize(0.5 * (dTx*dTx + dTy*dTy) / Om02)

        # Extrinsic curvature from γ_ij(t) = Ω^2 δ_ij
        # γ_xx = γ_yy = Ω^2 ; ∂_t γ_xx = 2 Ω ∂_t Ω
        lnOm1 = safe_lnOmega(T1, args.alpha, args.smooth_px, args.lnom_clip)
        Om1   = lnOmega_to_Omega(lnOm1, args.omega_floor)
        dOm_dt = (Om1 - Om0)/args.dt
        N = args.lapse
        Kxx = -(1.0/(2*N)) * (2*Om0*dOm_dt)   # = -(Ω \dotΩ)/N
        Kyy = Kxx.copy()
        # (we assume zero off-diagonal for now)
        # Raise indices with γ^{ij} = Ω^{-2} δ^{ij}
        ginv = 1.0/(Om02 + 1e-30)
        Kx_x = ginv*Kxx; Ky_y = ginv*Kyy
        # Scalars
        K = Kx_x + Ky_y                         # trace K = γ^{ij}K_{ij}
        KijKij = (ginv*ginv)*(Kxx*Kxx + Kyy*Kyy)
        H_LHS = sanitize(R2 + K*K - KijKij)     # full Hamiltonian left-hand side

        # Fit: H_LHS = β ρ_phi + κ ρ_m + 2Λ
        beta, kappa, twoLambda, R2fit, rms_res, rel_res, rms_lhs, _ = robust_fit(
            H_LHS, rho_phi, rho_m, mask, ridge=1e-8
        )

        rows.append(dict(pair=f"t{k}->t{k+1}",
                         beta=beta, kappa=kappa, twoLambda=twoLambda,
                         R2_score=R2fit, rms_res=rms_res, rel_res=rel_res,
                         rms_LHS=rms_lhs))
        print(f"[{k}] rel_res={rel_res:.3f}  rms_LHS={rms_lhs:.3e}  beta={beta:.3g}  kappa={kappa:.3g}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.outdir, "full_hamiltonian_pairs.csv")
    df.to_csv(out_csv, index=False)
    if len(df):
        agg = df.agg({"rel_res":["median","min"], "rms_LHS":["median","max"], "R2_score":["median","max"]})
        agg.to_csv(os.path.join(args.outdir, "summary.csv"))
    print("Saved:", out_csv, "and summary in", args.outdir)

if __name__ == "__main__":
    main()