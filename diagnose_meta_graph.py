#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose temporal 'tubes' between torus slices without modifying data.

Inputs: a series of H5 files like   <prefix>_t0_fields.h5 ... <prefix>_tK_fields.h5

We compute:
  - cross-slice correlations for lnΩ, |∇T|, ρ_m
  - per-slice mean/std for lnΩ and |∇T| (drift check)
  - common mask overlap across time
  - hypothetical time-edge weight histogram for given (sim_gamma, epsilon)
  - temporal-vs-spatial degree ratio proxy (local neighborhoods)

Outputs: CSVs + PNGs in --outdir
"""

import os, glob, json, argparse
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def load_fields(h5_path):
    with h5py.File(h5_path, 'r') as f:
        # Handle naming variants
        if 'lnOmega' in f:
            lnOm = f['lnOmega'][()]
        elif 'lnOm' in f:
            lnOm = f['lnOm'][()]
        elif 'Omega' in f:
            Om = f['Omega'][()]
            Om = np.clip(Om, 1e-30, None)
            lnOm = np.log(Om)
        else:
            raise KeyError(f"No lnOmega/lnOm/Omega in {h5_path}")

        if 'Teff' in f:
            T = f['Teff'][()]
        elif 'T' in f:
            T = f['T'][()]
        else:
            # If Teff missing, make a harmless zero-field; we’ll note it in summary
            T = np.zeros_like(lnOm)

        rho_m = f['rho_m'][()] if 'rho_m' in f else np.zeros_like(lnOm)

        x = f['x'][()] if 'x' in f else np.linspace(0,1,lnOm.shape[1])
        y = f['y'][()] if 'y' in f else np.linspace(0,1,lnOm.shape[0])

    return lnOm, T, rho_m, x, y

def grad_mag(T, dx=1.0, dy=1.0):
    dTx = np.gradient(T, dy, axis=0)
    dTy = np.gradient(T, dx, axis=1)
    return np.sqrt(dTx*dTx + dTy*dTy)

def make_mask(lnOm, mask_margin):
    h, w = lnOm.shape
    mask = np.ones_like(lnOm, dtype=bool)
    if mask_margin > 0:
        mask[:mask_margin,:] = False
        mask[-mask_margin:,:] = False
        mask[:, :mask_margin] = False
        mask[:, -mask_margin:] = False
    # also drop NaNs/infs
    mask &= np.isfinite(lnOm)
    return mask

def circ_shift(arr, dy, dx):
    return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)

def time_weight(lnOm_t, lnOm_tp1, mask, sim_gamma=1.0, epsilon=0.05, smooth_px=0.0):
    A = lnOm_t.copy()
    B = lnOm_tp1.copy()
    if smooth_px and smooth_px > 0:
        A = gaussian_filter(A, smooth_px)
        B = gaussian_filter(B, smooth_px)
    d = (A - B)**2
    # normalize by robust scale to avoid huge exponents
    s = np.nanmedian(np.abs(A[mask] - np.nanmedian(A[mask]))) + 1e-12
    z = d / (s*s)
    w = epsilon + np.exp(-sim_gamma * z)
    w[~mask] = np.nan
    return w

def spatial_degree_proxy(mask, k_radius=1, periodic=True):
    # count local neighbors in a (2r+1)x(2r+1) box, excluding self
    h, w = mask.shape
    deg = np.zeros_like(mask, dtype=float)
    for dy in range(-k_radius, k_radius+1):
        for dx in range(-k_radius, k_radius+1):
            if dy == 0 and dx == 0:
                continue
            if periodic:
                shifted = circ_shift(mask, dy, dx)
            else:
                shifted = np.zeros_like(mask, dtype=bool)
                sy0 = max(0,dy); sy1 = h+min(0,dy)
                sx0 = max(0,dx); sx1 = w+min(0,dx)
                shifted[sy0:sy1, sx0:sx1] = mask[sy0-dy:sy1-dy, sx0-dx:sx1-dx]
            deg += shifted.astype(float)
    deg[~mask] = np.nan
    return deg

def series_paths(prefix, k_max):
    paths = []
    for k in range(k_max+1):
        cand = f"{prefix}_t{k}_fields.h5"
        if os.path.exists(cand):
            paths.append(cand)
    return paths

def main():
    ap = argparse.ArgumentParser(description="Diagnose temporal connectivity between torus slices.")
    ap.add_argument('--series_prefix', required=True,
                    help='Prefix of H5s like <prefix>_t0_fields.h5 ...')
    ap.add_argument('--k_max', type=int, required=True,
                    help='Max index to look for (we load up to existing)')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--mask_margin', type=int, default=8)
    ap.add_argument('--sim_gamma', type=float, default=0.8, help='temporal similarity sharpness')
    ap.add_argument('--epsilon', type=float, default=0.08, help='baseline time weight floor')
    ap.add_argument('--smooth_px', type=float, default=12.0, help='light smoothing for similarity only')
    ap.add_argument('--periodic', type=int, default=1, help='1: wrap spatial neighbors (torus), 0: no wrap')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # discover slices
    paths = series_paths(args.series_prefix, args.k_max)
    if len(paths) < 2:
        raise RuntimeError("Need at least t0,t1 ...")

    # load all
    slices = []
    for p in paths:
        lnOm, T, rho_m, x, y = load_fields(p)
        mask = make_mask(lnOm, args.mask_margin)
        GM = grad_mag(T)  # |∇T|
        slices.append(dict(path=p, lnOm=lnOm, GM=GM, rho_m=rho_m, mask=mask))

    # ----- per-slice drift diagnostics -----
    rows = []
    for i, S in enumerate(slices):
        m_ln = np.nanmean(S['lnOm'][S['mask']])
        s_ln = np.nanstd(S['lnOm'][S['mask']])
        m_gm = np.nanmean(S['GM'][S['mask']])
        s_gm = np.nanstd(S['GM'][S['mask']])
        m_rho = np.nanmean(S['rho_m'][S['mask']])
        s_rho = np.nanstd(S['rho_m'][S['mask']])
        rows.append([i, m_ln, s_ln, m_gm, s_gm, m_rho, s_rho])
    drift = np.array(rows)
    np.savetxt(os.path.join(args.outdir, 'per_slice_drift.csv'),
               drift, delimiter=',',
               header='t,mean_lnOmega,std_lnOmega,mean_|gradT|,std_|gradT|,mean_rho_m,std_rho_m',
               comments='')
    # simple plots
    plt.figure(figsize=(7,4))
    plt.plot(drift[:,0], drift[:,1], '-o', label='mean lnΩ')
    plt.plot(drift[:,0], drift[:,3], '-o', label='mean |∇T|')
    plt.plot(drift[:,0], drift[:,5], '-o', label='mean ρ_m')
    plt.xlabel('t'); plt.ylabel('mean over mask'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir,'per_slice_means.png'), dpi=140); plt.close()

    # ----- mask overlap diagnostics -----
    common = slices[0]['mask'].copy()
    for S in slices[1:]:
        common &= S['mask']
    frac_common = np.sum(common) / np.sum(slices[0]['mask'])
    with open(os.path.join(args.outdir,'mask_overlap.txt'),'w') as f:
        f.write(f"Common-mask fraction (vs t0 mask): {frac_common:.4f}\n")
    # save visualization of common mask
    plt.figure(figsize=(4,4))
    plt.imshow(common, cmap='gray'); plt.title('Common mask'); plt.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir,'mask_common.png'), dpi=140); plt.close()

    # ----- cross-slice correlations + would-be time weights -----
    corr_rows = []
    w_all = []
    for i in range(len(slices)-1):
        A = slices[i]; B = slices[i+1]
        mask = A['mask'] & B['mask'] & common
        if np.sum(mask) < 100:
            continue

        a_ln = A['lnOm'][mask].ravel(); b_ln = B['lnOm'][mask].ravel()
        a_gm = A['GM'][mask].ravel();   b_gm = B['GM'][mask].ravel()
        a_rm = A['rho_m'][mask].ravel();b_rm = B['rho_m'][mask].ravel()

        def corr(x,y):
            if x.size < 10: return np.nan
            xm = x - np.nanmean(x); ym = y - np.nanmean(y)
            num = np.nansum(xm*ym)
            den = np.sqrt(np.nansum(xm*xm)*np.nansum(ym*ym)) + 1e-30
            return num/den

        c_ln = corr(a_ln, b_ln)
        c_gm = corr(a_gm, b_gm)
        c_rm = corr(a_rm, b_rm)

        # hypothetical time weights from lnΩ (smoothed), histogram later
        W = time_weight(A['lnOm'], B['lnOm'], mask=(A['mask']&B['mask']),
                        sim_gamma=args.sim_gamma, epsilon=args.epsilon, smooth_px=args.smooth_px)
        w_all.append(W[np.isfinite(W)])

        corr_rows.append([i, i+1, c_ln, c_gm, c_rm])
    corr_rows = np.array(corr_rows) if corr_rows else np.zeros((0,6))
    np.savetxt(os.path.join(args.outdir,'cross_slice_corr.csv'),
               corr_rows, delimiter=',',
               header='t,t+1,pearson_lnOmega,pearson_|gradT|,pearson_rho_m',
               comments='')

    # plot time weight hist
    if len(w_all) > 0:
        Wcat = np.concatenate(w_all)
        plt.figure(figsize=(6,4))
        plt.hist(Wcat, bins=60, density=True, alpha=0.8)
        plt.xlabel('hypothetical time-edge weight'); plt.ylabel('pdf')
        plt.title(f'γ={args.sim_gamma}, ε={args.epsilon}, smooth_px={args.smooth_px}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir,'time_weight_hist.png'), dpi=140); plt.close()

        # simple stats
        with open(os.path.join(args.outdir,'time_weight_stats.txt'),'w') as f:
            f.write(f"count={Wcat.size}\n")
            f.write(f"mean={np.nanmean(Wcat):.4f}  std={np.nanstd(Wcat):.4f}\n")
            q = np.nanpercentile(Wcat, [1,5,25,50,75,95,99])
            f.write("quantiles (1,5,25,50,75,95,99): " + ", ".join(f"{v:.4f}" for v in q) + "\n")

    # ----- degree ratio proxy -----
    # spatial degree (periodic torus neighborhood)
    deg_space = spatial_degree_proxy(common, k_radius=1, periodic=bool(args.periodic))
    # temporal degree proxy ~ fraction of pixels with weight above mid-level
    # use the previous Wcat distribution threshold  (if available)
    deg_time_ratio = np.nan
    if len(w_all) > 0:
        thr = 0.5  # consider weights >0.5 as active temporal neighbors
        # For each t, estimate fraction >thr
        active_fracs = [np.mean(Wv > thr) for Wv in w_all]
        # Spatial avg degree inside mask:
        mean_deg_space = np.nanmean(deg_space)
        if mean_deg_space > 0:
            deg_time_ratio = np.mean(active_fracs) / mean_deg_space

    with open(os.path.join(args.outdir,'degree_proxy.txt'),'w') as f:
        f.write(f"mean spatial degree (r=1): {np.nanmean(deg_space):.3f}\n")
        f.write(f"temporal/spatial degree ratio (proxy): {deg_time_ratio:.3f}\n")

    # ----- final summary -----
    summary = {
        "n_slices": len(slices),
        "common_mask_fraction_vs_t0": float(frac_common),
        "mean_lnOmega_drift": float(np.nanstd(drift[:,1])),
        "mean_gradT_drift": float(np.nanstd(drift[:,3])),
        "mean_rho_m_drift": float(np.nanstd(drift[:,5])),
        "mean_corr_lnOmega": float(np.nanmean(corr_rows[:,2])) if corr_rows.size else np.nan,
        "mean_corr_gradT":  float(np.nanmean(corr_rows[:,3])) if corr_rows.size else np.nan,
        "mean_corr_rho_m":  float(np.nanmean(corr_rows[:,4])) if corr_rows.size else np.nan,
        "time_weight_gamma": args.sim_gamma,
        "time_weight_epsilon": args.epsilon,
        "time_weight_smooth_px": args.smooth_px,
        "degree_ratio_proxy": float(deg_time_ratio) if not np.isnan(deg_time_ratio) else None
    }
    with open(os.path.join(args.outdir,'diagnostic_summary.json'),'w') as f:
        json.dump(summary, f, indent=2)

    print("=== Diagnose complete ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to: {args.outdir}")

if __name__ == "__main__":
    main()