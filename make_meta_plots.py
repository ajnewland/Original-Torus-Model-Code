#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a combined meta-geometry summary from multiple runs and plot:
  1) Correlation length ξ vs N (color = smooth_px)
  2) d_s^meta (plateau mean) vs N (color = alpha)

Assumes each run directory contains:
  - meta_summary.json
  - meta_heat_trace.csv
  - corr_vs_distance.csv

Usage (Windows example):
  python make_meta_plots.py ^
    --runs_root "C:\...\FINAL_RUN\meta" ^
    --outdir    "C:\...\FINAL_RUN\meta\combined_plots"

You can point --runs_root at a folder that contains many subfolders
(e.g. N64_R2path, N100_nine90_rot5, N100_graphLocal, …).
The script will walk the tree and auto-discover run folders.
"""

import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_csv_any(p: Path) -> pd.DataFrame:
    """Try to read CSV, fallback to semicolon if needed."""
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.read_csv(p, sep=";")

def fit_corr_length(df: pd.DataFrame):
    """
    Fit exponential decay: corr(d) ~ exp(a + b d)  => ln|corr| = a + b d
    Return (xi, R2), where xi = -1/b (pixels) if b<0, else NaN.
    """
    if df is None or df.empty:
        return np.nan, np.nan

    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("d_mid") or cols.get("distance") or cols.get("d") or list(df.columns)[0]
    ccol = cols.get("corr")  or cols.get("c")        or list(df.columns)[1]

    d = pd.to_numeric(df[dcol], errors="coerce").to_numpy()
    c = pd.to_numeric(df[ccol], errors="coerce").to_numpy()
    ok = np.isfinite(d) & np.isfinite(c) & (d > 0)
    d = d[ok]; c = c[ok]
    if d.size < 8:
        return np.nan, np.nan

    c_abs = np.abs(c)
    mask = c_abs > 1e-8
    d = d[mask]; c_abs = c_abs[mask]
    if d.size < 6:
        return np.nan, np.nan

    y = np.log(c_abs)
    A = np.vstack([np.ones_like(d), d]).T
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = coef
        xi = -1.0 / b if b < 0 else np.nan
        yhat = a + b * d
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return float(xi), float(r2)
    except Exception:
        return np.nan, np.nan

def spectral_dim_from_heat(df: pd.DataFrame):
    """
    If ds column is present (e.g. ds_meta, ds, d_s), use it.
    Else compute from heat trace K(t): d_s = -2 d ln K / d ln t (central diff).
    Returns (t_array, ds_array) or (None, None).
    """
    if df is None or df.empty:
        return None, None
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("t") or list(df.columns)[0]

    # Prefer explicit ds column if available
    for cand in ("ds_meta", "ds", "d_s", "dmeta"):
        if cand in cols:
            sub = df[[tcol, cols[cand]]].replace([np.inf, -np.inf], np.nan).dropna()
            if sub.empty: return None, None
            sub = sub.sort_values(by=tcol)
            return sub[tcol].to_numpy(), sub[cols[cand]].to_numpy()

    # Otherwise compute from heat trace
    kcol = cols.get("k") or cols.get("heat")
    if kcol is None:
        return None, None

    sub = df[[tcol, kcol]].replace([np.inf, -np.inf], np.nan).dropna().sort_values(by=tcol)
    if len(sub) < 5:
        return None, None
    t = sub[tcol].to_numpy()
    K = sub[kcol].to_numpy()
    lt = np.log(t)
    lK = np.log(K)
    dlnK = (lK[2:] - lK[:-2])
    dlt  = (lt[2:] - lt[:-2])
    ds   = -2.0 * dlnK / dlt
    t_mid = np.exp(0.5 * (lt[2:] + lt[:-2]))
    return t_mid, ds

def plateau_stats(t, ds):
    """Return (plateau_mean, plateau_std, ds_max, ds_median)."""
    if t is None or ds is None or len(ds) < 5:
        return np.nan, np.nan, np.nan, np.nan
    n = len(ds)
    # Middle 50% as a crude plateau proxy
    mid = ds[n//4: 3*n//4] if n >= 8 else ds
    return float(np.nanmean(mid)), float(np.nanstd(mid)), float(np.nanmax(ds)), float(np.nanmedian(ds))

def collect_runs(runs_root: Path):
    """
    Walk runs_root and collect any directory containing:
      meta_summary.json, meta_heat_trace.csv, corr_vs_distance.csv
    """
    run_dirs = []
    for root, dirs, files in os.walk(runs_root):
        fset = set(files)
        if {"meta_summary.json", "meta_heat_trace.csv", "corr_vs_distance.csv"} <= fset:
            run_dirs.append(Path(root))
    return run_dirs

def main():
    ap = argparse.ArgumentParser(description="Combine meta-geometry runs and make plots.")
    ap.add_argument("--runs_root", required=True, help="Folder containing per-run outputs")
    ap.add_argument("--outdir", required=True, help="Where to write the combined CSV and plots")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = collect_runs(runs_root)
    if not run_dirs:
        print("No runs found.")
        return

    rows = []
    all_ds = []

    for rd in run_dirs:
        # Load meta
        meta = {}
        try:
            meta = json.loads((rd / "meta_summary.json").read_text())
        except Exception:
            pass

        # Load CSVs
        corr = read_csv_any(rd / "corr_vs_distance.csv")
        heat = read_csv_any(rd / "meta_heat_trace.csv")

        # Fit correlation length
        xi, r2xi = fit_corr_length(corr)

        # Spectral dimension (direct or derived)
        t, ds = spectral_dim_from_heat(heat)
        pmean, pstd, pmax, pmed = plateau_stats(t, ds)
        if t is not None and ds is not None:
            all_ds.append((rd.name, t, ds))

        # Flexible metadata extraction
        N = meta.get("N", meta.get("n", np.nan))
        operator = meta.get("operator", meta.get("laplacian", "unknown"))
        sigma_graph = meta.get("sigma_graph", meta.get("sigma", np.nan))
        smooth_px = meta.get("smooth_px", np.nan)
        rot_deg = meta.get("rot_deg", meta.get("rotation", 0))
        alpha = meta.get("alpha", np.nan)

        rows.append({
            "run": rd.name,
            "path": rd.as_posix(),
            "N": pd.to_numeric(pd.Series([N]), errors="coerce").iloc[0],
            "operator": operator,
            "sigma_graph": pd.to_numeric(pd.Series([sigma_graph]), errors="coerce").iloc[0],
            "smooth_px": pd.to_numeric(pd.Series([smooth_px]), errors="coerce").iloc[0],
            "rot_deg": pd.to_numeric(pd.Series([rot_deg]), errors="coerce").iloc[0],
            "alpha": pd.to_numeric(pd.Series([alpha]), errors="coerce").iloc[0],
            "xi_pixels": xi,
            "xi_logfit_R2": r2xi,
            "ds_plateau_mean": pmean,
            "ds_plateau_spread": pstd,
            "ds_max": pmax,
            "ds_median": pmed
        })

    summary = pd.DataFrame(rows)
    summary_csv = outdir / "meta_combined_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved combined summary: {summary_csv}")

    # ---- Plots ----
    # ξ vs N (color = smooth_px)
    plt.figure()
    Nnum = pd.to_numeric(summary["N"], errors="coerce")
    Sm  = pd.to_numeric(summary["smooth_px"], errors="coerce")
    sc = plt.scatter(Nnum, summary["xi_pixels"], c=Sm, cmap="viridis")
    plt.xlabel("N (number of tori)")
    plt.ylabel("Correlation length ξ (pixels)")
    plt.title("ξ vs N (color = smooth_px)")
    cbar = plt.colorbar(sc); cbar.set_label("smooth_px")
    plt.tight_layout()
    plt.savefig(outdir / "xi_vs_N_smooth.png", dpi=220)
    plt.close()

    # d_s plateau mean vs N (color = alpha)
    plt.figure()
    Alp = pd.to_numeric(summary["alpha"], errors="coerce")
    sc2 = plt.scatter(Nnum, summary["ds_plateau_mean"], c=Alp, cmap="plasma")
    plt.axhline(4.0, linestyle="--")
    plt.xlabel("N (number of tori)")
    plt.ylabel(r"$d_s^{meta}$ (plateau mean)")
    plt.title(r"$d_s^{meta}$ plateau vs N (color = α)")
    cbar2 = plt.colorbar(sc2); cbar2.set_label("alpha")
    plt.tight_layout()
    plt.savefig(outdir / "ds_plateau_vs_N_alpha.png", dpi=220)
    plt.close()

    print("Saved plots:")
    print(" -", outdir / "xi_vs_N_smooth.png")
    print(" -", outdir / "ds_plateau_vs_N_alpha.png")

if __name__ == "__main__":
    main()