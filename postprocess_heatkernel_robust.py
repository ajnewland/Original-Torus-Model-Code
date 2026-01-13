#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust post-processing of heat-trace data to spectral dimension d_s(t).

- Smooths Theta(t) safely in log-space
- Derivative via sliding local linear fits (log–log)
- 68% bootstrap uncertainty bands
- Trims flat UV and noisy IR tails
- Detects plateaus (running CV)
- Writes CSV + publication-quality PNG

Required input: theta_mean.csv with columns [t, theta]
Optional: ensemble_meta.csv (to infer n_nodes)

Usage (examples at bottom of file).
Author: ChatGPT (for Anthony)
"""
import argparse, math, os
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter, medfilt
except Exception:
    savgol_filter = None
    medfilt = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- config -----------------------------
@dataclass
class Config:
    theta_path: str
    out_csv: str
    out_plot: str
    meta_path: Optional[str]
    n_nodes: Optional[int]
    uv_flat_frac: float      # keep theta < uv_flat_frac * max(theta)
    ir_floor_mult: float     # keep theta > ir_floor_mult / N
    win_points: int          # odd; sliding window for local slope
    medf_win: int            # odd; 0/1 disables
    savgol_win: int          # odd; 0/1 disables
    savgol_poly: int
    bootstrap_B: int
    plateau_cv: float        # <= this is “flat enough” for plateau
    plateau_min_dec: float   # min decades span
    verbose: bool

# ----------------------------- helpers ---------------------------
def read_theta(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    t_col, th_col = None, None
    for i, c in enumerate(cols):
        if c in ("t", "time", "tau"): t_col = df.columns[i]
        if c in ("theta", "heat_trace", "heat", "theta_mean"): th_col = df.columns[i]
    if t_col is None or th_col is None:
        t_col, th_col = df.columns[0], df.columns[1]
    out = df[[t_col, th_col]].rename(columns={t_col:"t", th_col:"theta"}).dropna()
    out = out[(out.t>0) & (out.theta>0)].sort_values("t").reset_index(drop=True)
    return out

def infer_n_from_meta(meta_path: Optional[str]) -> Optional[int]:
    if not meta_path or not os.path.exists(meta_path): return None
    try:
        df = pd.read_csv(meta_path)
        for cand in ("n_nodes","nodes","num_nodes"):
            if cand in df.columns:
                return max(1, int(round(df[cand].astype(float).mean())))
    except Exception:
        return None
    return None

def apply_smoothing(logt, logtheta, cfg: Config):
    y = logtheta.copy()
    if medfilt and cfg.medf_win>=3 and cfg.medf_win%2==1:
        y = medfilt(y, kernel_size=cfg.medf_win)
    if savgol_filter and cfg.savgol_win>=5 and cfg.savgol_win%2==1:
        poly = min(cfg.savgol_poly, cfg.savgol_win-2)
        y = savgol_filter(y, cfg.savgol_win, poly, mode="interp")
    return y

def local_lin_slope(x, y):
    if x.size < 3: return np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta[0]

def sliding_derivative_with_bootstrap(logt, logtheta, win, B):
    n, half = logt.size, win//2
    slopes = np.full(n, np.nan); lo = np.full(n, np.nan); hi = np.full(n, np.nan)
    rng = np.random.default_rng(1234)
    for i in range(n):
        i0, i1 = max(0,i-half), min(n,i+half+1)
        x, y = logt[i0:i1], logtheta[i0:i1]
        if x.size < 5: continue
        slopes[i] = local_lin_slope(x,y)
        if B>0 and x.size>=8:
            idx = np.arange(x.size)
            boots = [local_lin_slope(x[rng.choice(idx, x.size, True)], y[rng.choice(idx, x.size, True)]) for _ in range(B)]
            s = np.sort(np.asarray(boots))
            lo[i], hi[i] = np.nanpercentile(s, 16), np.nanpercentile(s, 84)
    return slopes, lo, hi

def detect_plateaus(logt, ds, cv_thresh, min_decades):
    good = np.isfinite(ds)
    if not np.any(good): return []
    # window ≈ one third decade
    dec_span = (logt[-1]-logt[0]) / math.log(10)
    pts_per_dec = len(logt)/dec_span if dec_span>0 else 100
    w = int(max(7, round(0.33*pts_per_dec))); w += (w%2==0)
    half = w//2
    mu = np.full_like(ds, np.nan); sd = np.full_like(ds, np.nan)
    for i in range(len(ds)):
        i0, i1 = max(0,i-half), min(len(ds), i+half+1)
        vec = ds[i0:i1]; vec = vec[np.isfinite(vec)]
        if vec.size>=5:
            mu[i], sd[i] = np.mean(vec), np.std(vec)
    cv = sd/np.maximum(mu,1e-9)
    is_pl = (cv<cv_thresh) & np.isfinite(cv) & np.isfinite(mu)

    res=[]; start=None
    for i,flag in enumerate(is_pl):
        if flag and start is None: start=i
        if (not flag or i==len(is_pl)-1) and start is not None:
            end = i if not flag else i
            span_dec = (logt[end]-logt[start])/math.log(10)
            if span_dec>=min_decades:
                tmin, tmax = float(np.exp(logt[start])), float(np.exp(logt[end]))
                level = float(np.nanmean(ds[start:end+1]))
                res.append((tmin,tmax,level))
            start=None
    return res

# ----------------------------- main ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Spectral d_s(t) from heat-trace.")
    ap.add_argument("--theta", required=True)
    ap.add_argument("--outds", required=True)
    ap.add_argument("--plot", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--n_nodes", type=int, default=None)
    ap.add_argument("--uv_flat_frac", type=float, default=0.98)
    ap.add_argument("--ir_floor_mult", type=float, default=5.0)
    ap.add_argument("--win_points", type=int, default=41)
    ap.add_argument("--medf_win", type=int, default=9)
    ap.add_argument("--savgol_win", type=int, default=31)
    ap.add_argument("--savgol_poly", type=int, default=3)
    ap.add_argument("--bootstrap_B", type=int, default=200)
    ap.add_argument("--plateau_cv", type=float, default=0.05)
    ap.add_argument("--plateau_min_dec", type=float, default=0.5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = Config(
        theta_path=args.theta,
        out_csv=args.outds,
        out_plot=args.plot,
        meta_path=args.meta,
        n_nodes=args.n_nodes,
        uv_flat_frac=args.uv_flat_frac,
        ir_floor_mult=args.ir_floor_mult,
        win_points=args.win_points if args.win_points%2==1 else args.win_points+1,
        medf_win=args.medf_win if args.medf_win%2==1 else max(1,args.medf_win-1),
        savgol_win=args.savgol_win if args.savgol_win%2==1 else args.savgol_win+1,
        savgol_poly=args.savgol_poly,
        bootstrap_B=args.bootstrap_B,
        plateau_cv=args.plateau_cv,
        plateau_min_dec=args.plateau_min_dec,
        verbose=args.verbose
    )

    df = read_theta(cfg.theta_path)
    t = df["t"].to_numpy(float)
    theta = df["theta"].to_numpy(float)

    # infer N
    N = cfg.n_nodes or infer_n_from_meta(cfg.meta_path)
    if N is None:
        floor = float(np.nanmin(theta))
        N = max(1, int(round(1.0/floor))) if floor>0 else 1000
        if cfg.verbose: print(f"[warn] inferred n_nodes ≈ {N} from min(theta)")

    # trim UV/IR
    th_max = float(np.nanmax(theta))
    uv_mask = theta < cfg.uv_flat_frac * th_max
    ir_mask = theta > (cfg.ir_floor_mult / float(N))
    keep = uv_mask & ir_mask
    idx = np.where(keep)[0]
    if idx.size >= 5:
        i0, i1 = idx[0], idx[-1]
        t_trim, theta_trim = t[i0:i1+1], theta[i0:i1+1]
    else:
        k0, k1 = int(0.05*len(t)), int(0.95*len(t))
        t_trim, theta_trim = t[k0:k1], theta[k0:k1]
        if cfg.verbose: print("[warn] trimming too strict; used central band")

    logt = np.log(t_trim)
    logtheta = np.log(theta_trim)
    logtheta_s = apply_smoothing(logt, logtheta, cfg)

    slopes, slo, shi = sliding_derivative_with_bootstrap(logt, logtheta_s, cfg.win_points, cfg.bootstrap_B)
    ds = -2.0*slopes
    ds_lo = -2.0*shi
    ds_hi = -2.0*slo

    finite = np.isfinite(ds)
    plateaus = detect_plateaus(logt[finite], ds[finite], cfg.plateau_cv, cfg.plateau_min_dec)

    # write CSV
    out = pd.DataFrame({"t": np.exp(logt), "ds": ds, "ds_lo": ds_lo, "ds_hi": ds_hi})
    out.to_csv(cfg.out_csv, index=False)

    # plot
    fig = plt.figure(figsize=(8,9))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(t, theta/theta[0], lw=1, alpha=0.6, label="Raw heat trace (norm.)")
    ax1.plot(np.exp(logt), np.exp(logtheta_s)/theta[0], lw=2, color="black", label="Smoothed")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_ylabel(r"$\Theta(t)$ (norm.)"); ax1.set_title("Heat Trace vs Diffusion Time")
    ax1.grid(True, which="both", ls=":", alpha=0.4); ax1.legend()

    ax2 = plt.subplot(2,1,2, sharex=ax1)
    ax2.plot(np.exp(logt), ds, lw=2, color="#d33", label=r"$d_s(t)$")
    if np.isfinite(ds_lo).any() and np.isfinite(ds_hi).any():
        ax2.fill_between(np.exp(logt), ds_lo, ds_hi, color="#d33", alpha=0.18, label="68% CI")
    for (tmin,tmax,level) in plateaus:
        ax2.axhline(level, ls="--", lw=1.5, color="#0a7")
        ax2.axvspan(tmin, tmax, color="#0a7", alpha=0.12)
        ax2.text((tmin*tmax)**0.5, level+0.06, f"plateau ≈ {level:.2f}", ha="center", va="bottom", fontsize=9, color="#074")
    ax2.set_xscale("log"); ax2.set_xlabel(r"Diffusion scale $t$")
    ax2.set_ylabel(r"Spectral dimension $d_s(t)$")
    ax2.set_title("Spectral Dimension Flow (robust)")
    ax2.grid(True, which="both", ls=":", alpha=0.4); ax2.legend()
    fig.tight_layout(); fig.savefig(cfg.out_plot, dpi=220); plt.close(fig)

    if cfg.verbose:
        print(f"[ok] wrote: {cfg.out_csv}")
        print(f"[ok] wrote: {cfg.out_plot}")
        for i,(t0,t1,lev) in enumerate(plateaus,1):
            print(f"  plateau {i}: d_s ~ {lev:.3f},  t∈[{t0:.3g},{t1:.3g}]")

if __name__ == "__main__":
    main()