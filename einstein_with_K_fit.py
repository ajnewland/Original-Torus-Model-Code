#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
einstein_with_K_fit.py

Add the missing extrinsic curvature invariant to the emergent Einstein fit, and
(optionally) switch to an R (not R^2) target. Works with your existing H5 slices.

Fits:   target ≈ β ρ_φ + κ ρ_m + γ (K^2 - K_ij K^ij) + δ V(T) + c0

Inputs (HDF5): expects datasets for at least T (Teff), optionally R2, rho_phi, rho_m.
If R2/rho_* missing, they'll be computed from T.

Author: Z3 Torus project
"""

import os, json, argparse
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# -------------------- Utilities --------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_fields(h5_path):
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        d = {}
        # typical field names we've used
        for name in ['T','Teff','R2','rho_phi','rho_m','ax','ay','x','y','Omega','lnOm']:
            if name in f: d[name] = np.array(f[name])
        # Sometimes stored under different groups:
        # Try to detect common alt names
        if 'Teff' in d and 'T' not in d: d['T'] = d['Teff']
        return d

def finite_diff_gradients(Z, dx=1.0, dy=1.0):
    dZdy, dZdx = np.gradient(Z, dy, dx)  # note numpy returns (rows->y, cols->x)
    return dZdx, dZdy

def laplacian_9pt(Z, dx=1.0, dy=1.0):
    # 9-point discrete Laplacian (periodic) on a torus
    Zxx = (np.roll(Z, -1, axis=1) - 2*Z + np.roll(Z, 1, axis=1)) / (dx*dx)
    Zyy = (np.roll(Z, -1, axis=0) - 2*Z + np.roll(Z, 1, axis=0)) / (dy*dy)
    # small cross-term average to mimic 9-pt (optional refinement)
    # Here we just do 5-pt for robustness; feel free to enhance if you like.
    return Zxx + Zyy

def safe_exp(x, clip=None):
    if clip is not None:
        x = np.clip(x, -abs(clip), abs(clip))
    return np.exp(x)

def gauss_smooth(Z, sigma_px):
    if sigma_px is None or sigma_px <= 0:
        return Z
    return gaussian_filter(Z, sigma_px, mode='wrap')

def rms(a, mask=None):
    if mask is not None:
        a = a[mask]
    return np.sqrt(np.mean(a*a))

def r2_score(y_true, y_pred, mask=None):
    if mask is not None:
        y_true = y_true[mask]; y_pred = y_pred[mask]
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - (ss_res/ss_tot if ss_tot>0 else np.nan)

# -------------------- Core builders --------------------

def build_geom_from_T(T, alpha=4.9, smooth_px=0.0, lnom_clip=None, omega_floor=None):
    lnOm = alpha * T
    if lnom_clip is not None:
        lnOm = np.clip(lnOm, -abs(lnom_clip), abs(lnom_clip))
    Om = np.exp(lnOm)
    if omega_floor is not None:
        Om = np.maximum(Om, omega_floor)
    lnOm_s = gauss_smooth(lnOm, smooth_px) if smooth_px else lnOm
    Om_s   = gauss_smooth(Om,   smooth_px) if smooth_px else Om
    return lnOm_s, Om_s

def compute_R_from_lnOm(lnOm, Om, dx=1.0, dy=1.0):
    # Conformal 2D identity: R = -2 Ω^{-2} ∇² ln Ω
    lap_ln = laplacian_9pt(lnOm, dx=dx, dy=dy)
    R = -2.0 * lap_ln / (Om*Om + 1e-30)
    return R

def compute_rho_phi_from_T(T, Om, dx=1.0, dy=1.0):
    dTdx, dTdy = finite_diff_gradients(T, dx=dx, dy=dy)
    rho_phi = 0.5 * (dTdx*dTdx + dTdy*dTdy) / (Om*Om + 1e-30)
    return rho_phi

def compute_metric_from_Om(Om):
    # 2D conformal Euclidean metric
    gxx = Om*Om
    gyy = Om*Om
    gxy = np.zeros_like(Om)
    # inverse
    inv = 1.0/(Om*Om + 1e-30)
    igxx = inv
    igyy = inv
    igxy = np.zeros_like(Om)
    return gxx, gyy, gxy, igxx, igyy, igxy

def compute_extrinsic_from_metric(gxx0, gyy0, gxy0, gxx1, gyy1, gxy1, dt, lapse=1.0):
    # K_ij ≈ - (1/2N) (g_ij(t+dt) - g_ij(t)) / dt
    fac = -1.0/(2.0*max(lapse,1e-12)*max(dt,1e-12))
    Kxx = fac * (gxx1 - gxx0)
    Kyy = fac * (gyy1 - gyy0)
    Kxy = fac * (gxy1 - gxy0)  # should be ~0 if metric stays diagonal
    return Kxx, Kyy, Kxy

def K_invariants(Kxx, Kyy, Kxy, igxx, igyy, igxy):
    # Raise one index: K^i_j = g^{ik} K_kj; but we want K = trace(g^{ij} K_ij)
    K_trace = igxx*Kxx + igyy*Kyy + 2.0*igxy*Kxy  # igxy terms vanish if igxy=0
    # K_ij K^{ij} = g^{ik} g^{jl} K_ij K_kl
    # For diagonal metric and Kxy present, approximate:
    KijKij = (igxx*igxx)*(Kxx*Kxx) + (igyy*igyy)*(Kyy*Kyy) + 2.0*(igxx*igyy)*(Kxy*Kxy)
    K2 = K_trace*K_trace
    return K_trace, K2, KijKij

def make_mask(shape, margin):
    mask = np.ones(shape, dtype=bool)
    if margin and margin>0:
        mask[:margin,:] = False
        mask[-margin:,:] = False
        mask[:,:margin] = False
        mask[:,-margin:] = False
    return mask

# -------------------- Fitting --------------------

def fit_linear(target, columns, names, mask):
    # Build design matrix with intercept
    X = np.stack(columns, axis=-1)
    if mask is not None:
        tar = target[mask].reshape(-1,1)
        X2 = X[mask].reshape(-1, X.shape[-1])
    else:
        tar = target.reshape(-1,1)
        X2 = X.reshape(-1, X.shape[-1])
    # add intercept
    Xd = np.concatenate([X2, np.ones((X2.shape[0],1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(Xd, tar, rcond=None)
    coeffs = {name: coef[i,0] for i,name in enumerate(names)}
    coeffs['c0'] = coef[-1,0]
    pred = (Xd @ coef).reshape(-1)
    # stats
    y = tar.reshape(-1)
    resid = y - pred
    rms_res = np.sqrt(np.mean(resid*resid))
    rms_y   = np.sqrt(np.mean(y*y)) + 1e-30
    rel_res = rms_res / rms_y
    # R^2
    R2 = 1.0 - np.sum((y-pred)**2)/np.sum((y-np.mean(y))**2 + 1e-30)
    return coeffs, rms_res, rel_res, R2

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Einstein fit with extrinsic curvature invariant and optional R target.")
    ap.add_argument("--h5_t0", required=True)
    ap.add_argument("--h5_t1", required=True)
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--lapse", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, required=True, help="Conformal factor exponent: lnΩ = α T")
    ap.add_argument("--smooth_px", type=float, default=90.0, help="Gaussian coarse-grain (px)")
    ap.add_argument("--mask_margin", type=int, default=12)
    ap.add_argument("--lnom_clip", type=float, default=None)
    ap.add_argument("--omega_floor", type=float, default=None)
    ap.add_argument("--target", choices=["R","R2"], default="R", help="Which curvature target to fit.")
    ap.add_argument("--add_V", choices=["none","quad","exp"], default="none", help="Optional potential term of T as extra regressor.")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Load slices
    d0 = load_fields(args.h5_t0)
    d1 = load_fields(args.h5_t1)
    if 'T' not in d0 or 'T' not in d1:
        raise RuntimeError("Missing T/Teff in one of the input H5 files.")
    T0 = d0['T'].astype(np.float64)
    T1 = d1['T'].astype(np.float64)

    # Build Ω (no renorm), coarse-grain at MACRO scale
    lnOm0, Om0 = build_geom_from_T(T0, alpha=args.alpha, smooth_px=args.smooth_px,
                                   lnom_clip=args.lnom_clip, omega_floor=args.omega_floor)
    lnOm1, Om1 = build_geom_from_T(T1, alpha=args.alpha, smooth_px=args.smooth_px,
                                   lnom_clip=args.lnom_clip, omega_floor=args.omega_floor)

    # Metric per slice
    gxx0, gyy0, gxy0, igxx0, igyy0, igxy0 = compute_metric_from_Om(Om0)
    gxx1, gyy1, gxy1, igxx1, igyy1, igxy1 = compute_metric_from_Om(Om1)

    # Extrinsic curvature (finite difference)
    Kxx, Kyy, Kxy = compute_extrinsic_from_metric(gxx0, gyy0, gxy0, gxx1, gyy1, gxy1,
                                                  dt=args.dt, lapse=args.lapse)
    # Invariants at t0 (use inverse at t0)
    K_trace, K2, KijKij = K_invariants(Kxx, Kyy, Kxy, igxx0, igyy0, igxy0)
    Kcomb = K2 - KijKij

    # Build curvature targets
    # Prefer compute from lnΩ & Ω to keep definitions consistent across runs
    R0 = compute_R_from_lnOm(lnOm0, Om0)
    if args.target == "R2":
        target = R0*R0
    else:
        target = R0

    # Build rho_phi, rho_m (compute if missing, at t0)
    if 'rho_phi' in d0:
        rho_phi = gauss_smooth(d0['rho_phi'].astype(np.float64), args.smooth_px)
    else:
        rho_phi = gauss_smooth(compute_rho_phi_from_T(T0, Om0), args.smooth_px)

    if 'rho_m' in d0:
        rho_m = gauss_smooth(d0['rho_m'].astype(np.float64), args.smooth_px)
    else:
        # If missing, set to zeros and still fit (lets you test vacuum)
        rho_m = np.zeros_like(T0)

    # Optional simple potential channel
    V = None
    if args.add_V == "quad":
        V = gauss_smooth(T0*T0, args.smooth_px)
    elif args.add_V == "exp":
        V = gauss_smooth(np.expm1(args.alpha*T0), args.smooth_px)

    # Smooth Kcomb on the same macro scale
    Kcomb_s = gauss_smooth(Kcomb, args.smooth_px)

    # Mask (edges)
    mask = make_mask(target.shape, args.mask_margin)

    # Assemble regressors
    cols = [rho_phi, rho_m, Kcomb_s]
    names = ["beta_phi", "kappa_m", "gamma_K"]
    if V is not None:
        cols.append(V); names.append("delta_V")

    coeffs, rms_res, rel_res, R2sc = fit_linear(target, cols, names, mask)

    # Predictions for plots
    # Make design with mask but reconstruct full pred for visualization:
    Xfull = np.stack(cols, axis=-1).reshape(-1, len(cols))
    Xfull = np.concatenate([Xfull, np.ones((Xfull.shape[0],1))], axis=1)
    coef_vec = np.array([coeffs[n] for n in names] + [coeffs['c0']]).reshape(-1,1)
    pred_full = (Xfull @ coef_vec).reshape(target.shape)

    # Save summary CSV
    summary = dict(
        target=args.target,
        alpha=args.alpha,
        smooth_px=args.smooth_px,
        mask_margin=args.mask_margin,
        dt=args.dt,
        lapse=args.lapse,
        beta_phi=coeffs["beta_phi"],
        kappa_m=coeffs["kappa_m"],
        gamma_K=coeffs["gamma_K"],
        c0=coeffs["c0"],
        add_V=args.add_V,
        delta_V=coeffs.get("delta_V", 0.0),
        rms_res=float(rms_res),
        rel_res=float(rel_res),
        R2_score=float(R2sc),
    )
    with open(os.path.join(args.outdir, "Kfit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Also save a flat CSV of coefficients
    with open(os.path.join(args.outdir, "Kfit_coeffs.csv"), "w", encoding="utf-8") as f:
        header = ["target","alpha","smooth_px","mask_margin","dt","lapse"] + names + ["c0","rms_res","rel_res","R2_score"]
        f.write(",".join(header)+"\n")
        row = [args.target, args.alpha, args.smooth_px, args.mask_margin, args.dt, args.lapse] + \
              [coeffs[n] for n in names] + [coeffs['c0'], rms_res, rel_res, R2sc]
        f.write(",".join(str(x) for x in row)+"\n")

    # Quick PDF report
    try:
        fig,axs = plt.subplots(2,3, figsize=(12,7))
        im0=axs[0,0].imshow(target, cmap='viridis'); axs[0,0].set_title(f"Target: {args.target}")
        fig.colorbar(im0, ax=axs[0,0], fraction=0.046)
        im1=axs[0,1].imshow(pred_full, cmap='viridis'); axs[0,1].set_title("Linear fit")
        fig.colorbar(im1, ax=axs[0,1], fraction=0.046)
        im2=axs[0,2].imshow((target-pred_full), cmap='coolwarm'); axs[0,2].set_title("Residual")
        fig.colorbar(im2, ax=axs[0,2], fraction=0.046)

        axs[1,0].imshow(rho_phi, cmap='inferno'); axs[1,0].set_title(r"$\rho_\phi$")
        axs[1,1].imshow(rho_m, cmap='inferno');   axs[1,1].set_title(r"$\rho_m$")
        axs[1,2].imshow(Kcomb_s, cmap='magma');   axs[1,2].set_title(r"$K^2 - K_{ij}K^{ij}$")

        for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Einstein fit with K-term  |  rel_res={rel_res:0.3f}, R2={R2sc:0.3f}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "Kfit_summary.pdf"))
        plt.close(fig)
    except Exception as e:
        # Non-fatal if plotting fails
        with open(os.path.join(args.outdir, "Kfit_plot_error.txt"), "w") as f:
            f.write(str(e))

    print("=== Einstein fit with K-term ===")
    print(f"target     : {args.target}")
    print(f"alpha      : {args.alpha}")
    print(f"smooth_px  : {args.smooth_px}   mask_margin: {args.mask_margin}")
    print(f"dt, lapse  : {args.dt}, {args.lapse}")
    print(f"coeffs     : {json.dumps(coeffs, indent=2)}")
    print(f"rms_res    : {rms_res:0.6g}")
    print(f"rel_res    : {rel_res:0.6g}")
    print(f"R2_score   : {R2sc:0.6g}")
    print(f"Saved: {os.path.join(args.outdir,'Kfit_summary.json')} and Kfit_coeffs.csv (+ PDF)")
    
if __name__ == "__main__":
    main()
