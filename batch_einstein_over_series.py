#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Einstein + momentum over consecutive real slices (stable version).
- Unit-box spectral Laplacian (no blow-ups).
- Safe lnΩ -> Ω (mean-center + clip + floor).
- Sanitizes arrays and masks non-finite entries.
- Robust Hamiltonian fit (standardize + tiny ridge).

Outputs:
  pair_results.csv, summary.csv, series_summary.pdf
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

def direct_fit_robust(R2, rho_phi, rho_m, mask, ridge=1e-8):
    # build masked vectors
    r = R2[mask].ravel()
    a = rho_phi[mask].ravel()
    b = rho_m[mask].ravel()
    # finite mask
    finite = np.isfinite(r) & np.isfinite(a) & np.isfinite(b)
    r, a, b = r[finite], a[finite], b[finite]
    if r.size < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # standardize features for numerics
    A = (a - a.mean()) / (a.std() + 1e-12)
    B = (b - b.mean()) / (b.std() + 1e-12)
    X = np.vstack([A, B, np.ones_like(A)]).T
    y = r - r.mean()

    # tiny ridge on [A,B] only
    XtX = X.T @ X
    lam = ridge * (y.var() + 1e-30)
    R = np.diag([lam, lam, 0.0])
    coef = np.linalg.solve(XtX + R, X.T @ y)  # stable
    Ahat_s, Bhat_s, Chat = coef.tolist()

    # unstandardize: r ≈ A*a + B*b + C
    Ahat = Ahat_s / (a.std() + 1e-12)
    Bhat = Bhat_s / (b.std() + 1e-12)
    Chat = Chat + r.mean() - Ahat*a.mean() - Bhat*b.mean()

    fit = Ahat*rho_phi + Bhat*rho_m + Chat
    resid = R2 - fit

    # R^2 & residual norms on masked finite set
    r_mask = R2[mask].ravel()
    f_mask = fit[mask].ravel()
    finite2 = np.isfinite(r_mask) & np.isfinite(f_mask)
    r_mask, f_mask = r_mask[finite2], f_mask[finite2]
    ss_res = float(np.sum((r_mask - f_mask)**2))
    ss_tot = float(np.sum((r_mask - r_mask.mean())**2)) + 1e-30
    R2_score = 1.0 - ss_res/ss_tot
    ein_rms = float(np.sqrt(np.mean((r_mask - f_mask)**2)))
    ein_rel = float(ein_rms / (np.sqrt(np.mean(r_mask**2)) + 1e-12))
    return Ahat, Bhat, Chat, ein_rms, ein_rel, R2_score

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Batch Einstein+momentum over consecutive slices (stable)")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--k_max", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=4.9)
    ap.add_argument("--smooth_px", type=float, default=2.0)
    ap.add_argument("--lnom_clip", type=float, default=3.0)
    ap.add_argument("--omega_floor", type=float, default=1e-3)
    ap.add_argument("--mask_margin", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--lapse", type=float, default=1.0)
    ap.add_argument("--outdir", default="series_reports_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for k in range(0, args.k_max):
        h5_a = f"{args.prefix}_t{k}_fields.h5"
        h5_b = f"{args.prefix}_t{k+1}_fields.h5"
        if not (os.path.exists(h5_a) and os.path.exists(h5_b)):
            print("Skipping missing pair:", h5_a, h5_b)
            continue

        with h5py.File(h5_a, "r") as fa, h5py.File(h5_b, "r") as fb:
            T0 = fa["Teff"][:]; T1 = fb["Teff"][:]
            rho_m = fa["rho_m"][:]

        Ny, Nx = T0.shape
        mm = int(args.mask_margin)
        mask = np.ones((Ny, Nx), bool)
        mask[:mm,:] = mask[-mm:,:] = mask[:,:mm] = mask[:,-mm:] = False

        # t0 fields via safe pipeline
        lnOm0 = safe_lnOmega(T0, args.alpha, args.smooth_px, args.lnom_clip)
        Om0   = lnOmega_to_Omega(lnOm0, args.omega_floor)
        lap_ln0 = laplacian_spectral_unit(lnOm0)
        Om02 = Om0*Om0
        R2 = sanitize(-2.0 * lap_ln0 / Om02)

        dTdx0, dTdy0 = central_grad_unit(T0)
        rho_phi = sanitize(0.5 * (dTdx0*dTdx0 + dTdy0*dTdy0) / Om02)

        # Hamiltonian robust fit
        A,B,C,ein_rms,ein_rel,R2_score = direct_fit_robust(R2, rho_phi, rho_m, mask)

        # momentum (source-subtracted) on ensemble-mean = here single pair
        lnOm1 = safe_lnOmega(T1, args.alpha, args.smooth_px, args.lnom_clip)
        Om1   = lnOmega_to_Omega(lnOm1, args.omega_floor)

        Kxx = -(Om0 * ((Om1 - Om0) / args.dt)) / args.lapse
        Kyy = Kxx.copy()
        ginv = 1.0 / (Om02 + 1e-30)
        K = ginv * (Kxx + Kyy)
        Kxx_c = (ginv*ginv) * Kxx
        Kyy_c = (ginv*ginv) * Kyy
        Sxx = Kxx_c - ginv * K
        Syy = Kyy_c - ginv * K
        Sxy = np.zeros_like(Sxx); Syx = np.zeros_like(Sxx)

        # simple divergence (unit spacing)
        dTxx_dx = 0.5*(np.roll(Sxx, -1, 1) - np.roll(Sxx, 1, 1))
        dTxy_dy = 0.5*(np.roll(Sxy, -1, 0) - np.roll(Sxy, 1, 0))
        dTyx_dx = 0.5*(np.roll(Syx, -1, 1) - np.roll(Syx, 1, 1))
        dTyy_dy = 0.5*(np.roll(Syy, -1, 0) - np.roll(Syy, 1, 0))
        Cx = dTxx_dx + dTxy_dy
        Cy = dTyx_dx + dTyy_dy

        Cnorm = (Om02) * (Cx*Cx + Cy*Cy)
        dKdx, dKdy = central_grad_unit(K)
        gradK_norm = (Om02) * (dKdx*dKdx + dKdy*dKdy)

        # finite mask for momentum norms
        mfin = mask & np.isfinite(Cnorm) & np.isfinite(gradK_norm)
        if not mfin.any():
            mom_rms_abs = np.nan; mom_rel = np.nan
        else:
            mom_rms_abs = float(np.sqrt(np.mean(Cnorm[mfin])))
            mom_rms_ref = float(np.sqrt(np.mean(gradK_norm[mfin])) + 1e-12)
            mom_rel = float(mom_rms_abs / mom_rms_ref)

        # source subtraction j^i
        Tdot = (T1 - T0) / args.dt
        jx = ginv * Tdot * dTdx0
        jy = ginv * Tdot * dTdy0
        num = float(np.nansum(Cx[mfin]*jx[mfin] + Cy[mfin]*jy[mfin]))
        den = float(np.nansum(jx[mfin]*jx[mfin] + jy[mfin]*jy[mfin]) + 1e-30)
        c_phi = num / den
        Cx_s = Cx - c_phi * jx
        Cy_s = Cy - c_phi * jy
        Cnorm_s = (Om02) * (Cx_s*Cx_s + Cy_s*Cy_s)
        mom_rms_abs_src = float(np.sqrt(np.nanmean(Cnorm_s[mfin])))
        mom_rel_src = float(mom_rms_abs_src / (np.sqrt(np.nanmean(gradK_norm[mfin])) + 1e-12))

        rows.append(dict(pair=f"t{k}->t{k+1}", A=A,B=B,C=C, ein_rel=ein_rel,
                         R2_score=R2_score, mom_rel=mom_rel,
                         c_phi=c_phi, mom_rel_src=mom_rel_src))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "pair_results.csv"), index=False)
    agg = df.agg({"ein_rel":["median","min"], "mom_rel_src":["median","min"]})
    agg.to_csv(os.path.join(args.outdir, "summary.csv"))

    # quick PDF
    with PdfPages(os.path.join(args.outdir, "series_summary.pdf")) as pp:
        fig,ax = plt.subplots(figsize=(7,4.6))
        ax.plot(df["ein_rel"].values, marker='o', label="ein_rel")
        ax.plot(df["mom_rel_src"].values, marker='s', label="mom_rel_src")
        ax.set_xticks(range(len(df))); ax.set_xticklabels(df["pair"], rotation=45, ha='right')
        ax.set_ylabel("residual"); ax.set_title("Einstein & momentum over real time pairs")
        ax.legend(); plt.tight_layout(); pp.savefig(fig); plt.close(fig)
    print("Saved results to:", args.outdir)

if __name__ == "__main__":
    main()