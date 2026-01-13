#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a perturbed torus ensemble and test Einstein emergence (full ADM Hamiltonian).

- Input: a *stable* template slice H5 (e.g., torus_rich_t0_fields.h5)
  containing: Teff (T), rho_m (matter), ax, ay (for reference), etc.
- For each run i=1..N:
  * Create a perturbed Teff: T0' = T0 + noise
  * Evolve one small grad-flow step: T1' = T0' + dt * mu_i * Δ T0'
  * From (T0',T1'), build Ω, R(γ), K_ij and Hamiltonian LHS
  * Fit: H_LHS = β ρφ + κ ρm + 2Λ  (robust, ridge-stabilized)
  * Save run folder with t0/t1 H5s + per-run CSV
- After each run, update ensemble running means of H_LHS, ρφ, ρm and
  refit to report ensemble relative residual vs number of tori.

Author: A. J. Newland, 2025
"""

import os, sys, argparse
import numpy as np, pandas as pd, h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------- Numerics ----------------

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

def evolve_step_gradflow(T, mu, dt):
    # simple stable step: T_{t+dt} = T + dt * mu * ΔT
    return sanitize(T + dt * mu * laplacian_spectral_unit(T))

# ---------------- Robust fit ----------------

def robust_fit(lhs, rphi, rm, mask, ridge=1e-8):
    """
    Fit lhs ≈ β rphi + κ rm + 2Λ on masked finite pixels.
    Returns: beta, kappa, twoLambda, R2, rms_res, rel_res, rms_lhs
    """
    y = lhs[mask].ravel()
    a = rphi[mask].ravel()
    b = rm[mask].ravel()
    finite = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[finite], a[finite], b[finite]
    if y.size < 10:
        return (np.nan,)*7

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

    beta  = Ahat_s / (a.std()+1e-12)
    kappa = Bhat_s / (b.std()+1e-12)
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

# ---------------- Per-run Hamiltonian build ----------------

def hamiltonian_from_pair(T0, T1, rho_m, alpha, smooth_px, lnom_clip, omega_floor, dt, lapse, mask_margin):
    Ny, Nx = T0.shape
    mask = np.ones((Ny, Nx), bool)
    mm = int(mask_margin)
    mask[:mm,:] = mask[-mm:,:] = mask[:,:mm] = mask[:,-mm:] = False

    # Metric & curvature at t0
    lnOm0 = safe_lnOmega(T0, alpha, smooth_px, lnom_clip)
    Om0   = lnOmega_to_Omega(lnOm0, omega_floor)
    Om02  = Om0*Om0
    R2    = sanitize(-2.0 * laplacian_spectral_unit(lnOm0) / Om02)

    # Scalar gradient energy at t0
    dTx, dTy = central_grad_unit(T0)
    rho_phi  = sanitize(0.5 * (dTx*dTx + dTy*dTy) / Om02)

    # Extrinsic curvature from γ_ij = Ω^2 δ_ij
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

# ---------------- Main ensemble builder ----------------

def main():
    ap = argparse.ArgumentParser("Build a perturbed torus ensemble and test ADM Hamiltonian emergence")
    ap.add_argument("--template_h5", required=True, help="Stable template t0 fields (e.g., torus_rich_t0_fields.h5)")
    ap.add_argument("--N", type=int, default=6, help="Number of perturbed tori to build")
    ap.add_argument("--alpha0", type=float, default=4.9)
    ap.add_argument("--mu0", type=float, default=0.002)
    ap.add_argument("--dalpha", type=float, default=0.02, help="Uniform ± range on alpha")
    ap.add_argument("--dmu", type=float, default=0.001, help="Uniform ± range on mu")
    ap.add_argument("--noise_Teff_std", type=float, default=0.02, help="Teff noise as fraction of std(T)")
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--smooth_px", type=float, default=2.0)
    ap.add_argument("--lnom_clip", type=float, default=3.0)
    ap.add_argument("--omega_floor", type=float, default=1e-3)
    ap.add_argument("--mask_margin", type=int, default=8)
    ap.add_argument("--lapse", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--outdir", default="torus_ensemble")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Load template
    with h5py.File(args.template_h5, "r") as h:
        T0_base = h["Teff"][:]
        rho_m_base = h["rho_m"][:]
        # keep ax, ay for provenance if present
        ax = h["ax"][:] if "ax" in h else None
        ay = h["ay"][:] if "ay" in h else None

    Ny, Nx = T0_base.shape
    Tstd = float(np.std(T0_base))

    # Running ensemble means for H_LHS, rho_phi, rho_m
    ens_H = np.zeros((Ny, Nx), dtype=float)
    ens_rphi = np.zeros_like(ens_H)
    ens_rm = np.zeros_like(ens_H)
    mask_global = np.ones((Ny, Nx), dtype=bool)
    mm = int(args.mask_margin)
    mask_global[:mm,:] = mask_global[-mm:,:] = mask_global[:,:mm] = mask_global[:,-mm:] = False

    per_run_rows = []
    ensemble_rows = []

    # PDF for residual vs N
    pdf_path = os.path.join(args.outdir, "ensemble_progress.pdf")
    pdf = PdfPages(pdf_path)

    for i in range(1, args.N+1):
        run_dir = os.path.join(args.outdir, f"run_{i:03d}")
        os.makedirs(run_dir, exist_ok=True)

        # Sample perturbations
        alpha_i = args.alpha0 + rng.uniform(-args.dalpha, args.dalpha)
        mu_i    = args.mu0    + rng.uniform(-args.dmu,    args.dmu)
        noise   = args.noise_Teff_std * Tstd * rng.normal(size=T0_base.shape)

        # Build perturbed pair
        T0 = sanitize(T0_base + noise)
        T1 = evolve_step_gradflow(T0, mu_i, args.dt)

        # Save run H5s
        for tag, T in (("t0", T0), ("t1", T1)):
            out_h5 = os.path.join(run_dir, f"series_{tag}_fields.h5")
            with h5py.File(out_h5, "w") as w:
                w.create_dataset("Teff", data=T)
                w.create_dataset("rho_m", data=rho_m_base)
                if ax is not None: w.create_dataset("ax", data=ax)
                if ay is not None: w.create_dataset("ay", data=ay)

        # Hamiltonian for this run
        H_LHS, rphi, rm, mask, R2 = hamiltonian_from_pair(
            T0, T1, rho_m_base, alpha_i, args.smooth_px, args.lnom_clip,
            args.omega_floor, args.dt, args.lapse, args.mask_margin
        )
        beta, kappa, twoLam, R2fit, rms_res, rel_res, rms_lhs = robust_fit(H_LHS, rphi, rm, mask)

        per_run_rows.append(dict(
            run=i, alpha=alpha_i, mu=mu_i,
            beta=beta, kappa=kappa, twoLambda=twoLam,
            R2_score=R2fit, rms_res=rms_res, rel_res=rel_res, rms_LHS=rms_lhs
        ))
        pd.DataFrame(per_run_rows).to_csv(os.path.join(args.outdir, "per_run_results.csv"), index=False)

        # Update ensemble running means
        ens_H    = (ens_H*(i-1) + H_LHS) / i
        ens_rphi = (ens_rphi*(i-1) + rphi) / i
        ens_rm   = (ens_rm*(i-1) + rm)   / i

        # Fit on ensemble means
        e_beta, e_kappa, e_twoLam, e_R2, e_rms_res, e_rel_res, e_rms_lhs = robust_fit(
            ens_H, ens_rphi, ens_rm, mask_global
        )
        ensemble_rows.append(dict(
            n=i, beta=e_beta, kappa=e_kappa, twoLambda=e_twoLam,
            R2_score=e_R2, rms_res=e_rms_res, rel_res=e_rel_res, rms_LHS=e_rms_lhs
        ))
        ens_df = pd.DataFrame(ensemble_rows)
        ens_df.to_csv(os.path.join(args.outdir, "ensemble_progress.csv"), index=False)

        # Plot progress
        fig, axp = plt.subplots(figsize=(6.2,3.8))
        axp.plot(ens_df["n"], ens_df["rel_res"], marker="o")
        axp.set_xlabel("Number of tori in ensemble")
        axp.set_ylabel("Ensemble relative residual (Hamiltonian)")
        axp.set_title("Einstein emergence vs ensemble size")
        axp.grid(True, alpha=0.3)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        print(f"[run {i:02d}] per-run rel_res={rel_res:.3f} | ensemble rel_res={e_rel_res:.3f}")

    pdf.close()
    print("Saved:",
          os.path.join(args.outdir, "per_run_results.csv"),
          os.path.join(args.outdir, "ensemble_progress.csv"),
          pdf_path)

if __name__ == "__main__":
    main()