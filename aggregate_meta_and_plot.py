#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate meta-geometry outputs across many grid runs and produce summary plots.

- Recursively scans any number of root folders (e.g., grid_runs/ grid_runs_nine/ grid_runs_nine1/)
- Extracts robust metrics from meta_summary.json
- Optionally enriches with rot_deg, sigma_graph, windowsize if present
- Writes grid_meta_summary_all.csv
- Produces plots:
    * xi vs N (scatter with trend + model fits)
    * xi vs smooth_px (scatter + operator coloring)
    * ds_plateau_mean vs N (trend + 1/N and 1/sqrt(N) extrapolations)
    * Operator comparison boxplots
    * Heatmaps (N x smooth_px) if the grid is dense enough

Usage (Windows example):
  python aggregate_meta_and_plot.py ^
      --roots "C:\\path\\to\\grid_runs" "C:\\path\\to\\grid_runs_nine" "C:\\path\\to\\grid_runs_nine1" ^
      --outdir "C:\\path\\to\\summary_out"

"""

import argparse, os, sys, json, glob, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- helpers ----------------------------------------------------------------

def _to_float(x):
    if x is None: return np.nan
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s in ("", "nan", "None", "NaN"): return np.nan
        # allow scientific strings like "1e-6"
        try: return float(s)
        except: return np.nan
    return np.nan

def _get(d, *keys, default=np.nan):
    for k in keys:
        if k in d: return d[k]
    return default

def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def read_meta_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None

def collect_rows(root):
    rows = []
    for meta_path in glob.glob(os.path.join(root, "**", "meta_summary.json"), recursive=True):
        d = read_meta_json(meta_path)
        if d is None:
            continue

        # Normalise/collect keys (robust to different writers)
        row = {
            "folder_root": str(root),
            "meta_path": meta_path.replace("\\", "/"),
            "run_dir": str(Path(meta_path).parent).replace("\\", "/"),

            # core identifiers
            "N": _to_float(_get(d, "N")),
            "alpha": _to_float(_get(d, "alpha", "alpha_used", "alpha0")),
            "operator": _get(d, "operator", default="unknown"),
            "smooth_px": _to_float(_get(d, "smooth_px")),
            "sigma_graph": _to_float(_get(d, "sigma_graph", "sigma_graph_px")),
            "rot_deg": _to_float(_get(d, "rot_deg", "rotation_deg")),
            "windowsize": _to_float(_get(d, "windowsize", "window_px")),

            # correlation length
            "xi_pixels": _to_float(_get(d, "xi_pixels", "xi", "xi_px")),
            "xi_logfit_R2": _to_float(_get(d, "xi_logfit_R2", "xi_R2", "xi_r2")),

            # spectral dimension
            "ds_plateau_mean": _to_float(_get(d, "ds_plateau_mean", "ds_mean", "ds_plateau")),
            "ds_plateau_spread": _to_float(_get(d, "ds_plateau_spread", "ds_spread")),
            "ds_max": _to_float(_get(d, "ds_max")),
            "ds_median": _to_float(_get(d, "ds_median"))
        }

        rows.append(row)
    return rows

def summarize_missing(df, cols):
    lines = []
    lines.append("=== Column completeness ===")
    for c in cols:
        m = df[c].isna().mean()*100 if c in df.columns else 100.0
        if c in df.columns and np.issubdtype(df[c].dtype, np.number):
            series = df[c].dropna()
            if len(series):
                desc = f"mean={series.mean():.4g}  std={series.std():.4g}  min={series.min():.4g}  max={series.max():.4g}"
            else:
                desc = "no valid numbers"
        else:
            desc = "n/a"
        lines.append(f"{c:18s}  missing={m:5.1f}%   {desc}")
    return "\n".join(lines)

def fit_trend_xi_vs_N(df):
    # Try xi ~ a * N^p; take logs on valid rows
    d = df.dropna(subset=["N", "xi_pixels"]).copy()
    d = d[(d["N"]>0) & (d["xi_pixels"]>0)]
    if len(d) < 5:
        return None
    x = np.log(d["N"].values)
    y = np.log(d["xi_pixels"].values)
    A = np.vstack([np.ones_like(x), x]).T
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    loga, p = coeff
    a = np.exp(loga)
    # R^2
    yhat = A @ coeff
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-30
    r2 = 1.0 - ss_res/ss_tot
    return {"a": a, "p": p, "R2": r2, "n": len(d)}

def fit_ds_extrap(df):
    """
    Fit two simple asymptotics for ds_plateau_mean:
      Model A: ds = A + B*(1/N)
      Model B: ds = A + C*(1/sqrt(N))
    Return both with R^2 and A (the inferred Einstein-limit ds).
    """
    results = {}
    d = df.dropna(subset=["N", "ds_plateau_mean"]).copy()
    d = d[(d["N"]>0)]
    if len(d) < 5:
        return None

    y = d["ds_plateau_mean"].values

    # Model A: 1/N
    X_A = np.vstack([np.ones_like(y), 1.0/d["N"].values]).T
    coefA, *_ = np.linalg.lstsq(X_A, y, rcond=None)
    A_A, B_A = coefA
    yhatA = X_A @ coefA
    r2A = 1.0 - np.sum((y-yhatA)**2)/(np.sum((y-y.mean())**2)+1e-30)

    # Model B: 1/sqrt(N)
    X_B = np.vstack([np.ones_like(y), 1.0/np.sqrt(d["N"].values)]).T
    coefB, *_ = np.linalg.lstsq(X_B, y, rcond=None)
    A_B, C_B = coefB
    yhatB = X_B @ coefB
    r2B = 1.0 - np.sum((y-yhatB)**2)/(np.sum((y-y.mean())**2)+1e-30)

    results["model_1_over_N"] = {"A": A_A, "B": B_A, "R2": r2A}
    results["model_1_over_sqrtN"] = {"A": A_B, "C": C_B, "R2": r2B}
    results["n"] = int(len(d))
    return results

# ---- plotting ---------------------------------------------------------------

def plot_xi_vs_N(df, outdir):
    d = df.dropna(subset=["N","xi_pixels"]).copy()
    plt.figure(figsize=(7,5))
    for op, sub in d.groupby("operator"):
        plt.scatter(sub["N"], sub["xi_pixels"], s=20, alpha=0.7, label=str(op))
    fit = fit_trend_xi_vs_N(d)
    if fit:
        xs = np.linspace(d["N"].min(), d["N"].max(), 200)
        ys = fit["a"]*(xs**fit["p"])
        plt.plot(xs, ys, lw=2, label=f"fit: ξ ≈ {fit['a']:.2g}·N^{fit['p']:.3f} (R²={fit['R2']:.3f})")
    plt.xlabel("N (number of tori)")
    plt.ylabel("ξ (pixels)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "xi_vs_N.png"), dpi=160)
    plt.close()

def plot_xi_vs_smooth(df, outdir):
    d = df.dropna(subset=["smooth_px","xi_pixels"]).copy()
    plt.figure(figsize=(7,5))
    for op, sub in d.groupby("operator"):
        plt.scatter(sub["smooth_px"], sub["xi_pixels"], s=20, alpha=0.7, label=str(op))
    plt.xlabel("smooth_px")
    plt.ylabel("ξ (pixels)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "xi_vs_smooth_px.png"), dpi=160)
    plt.close()

def plot_ds_vs_N(df, outdir):
    d = df.dropna(subset=["N","ds_plateau_mean"]).copy()
    plt.figure(figsize=(7,5))
    for op, sub in d.groupby("operator"):
        plt.scatter(sub["N"], sub["ds_plateau_mean"], s=20, alpha=0.7, label=str(op))
    # fits
    fits = fit_ds_extrap(d)
    if fits:
        xs = np.linspace(d["N"].min(), d["N"].max(), 200)
        # 1/N
        A = fits["model_1_over_N"]["A"]; B = fits["model_1_over_N"]["B"]
        plt.plot(xs, A + B*(1.0/xs), lw=2, label=f"ds ≈ A + B/N  (A={A:.3f}, R²={fits['model_1_over_N']['R2']:.3f})")
        # 1/sqrt(N)
        A2 = fits["model_1_over_sqrtN"]["A"]; C = fits["model_1_over_sqrtN"]["C"]
        plt.plot(xs, A2 + C*(1.0/np.sqrt(xs)), lw=2, label=f"ds ≈ A + C/√N (A={A2:.3f}, R²={fits['model_1_over_sqrtN']['R2']:.3f})")
    plt.axhline(4.0, color="k", lw=1, ls="--", label="Einstein limit (d_s=4)")
    plt.xlabel("N (number of tori)")
    plt.ylabel("d_s plateau mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ds_plateau_mean_vs_N.png"), dpi=160)
    plt.close()

def plot_operator_boxplots(df, outdir):
    d = df.dropna(subset=["operator","xi_pixels","ds_plateau_mean"]).copy()
    if d.empty: return
    plt.figure(figsize=(8,4))
    order = sorted(d["operator"].unique().tolist())
    data = [d.loc[d["operator"]==op, "xi_pixels"].values for op in order]
    plt.boxplot(data, labels=order, showfliers=False)
    plt.ylabel("ξ (pixels)")
    plt.title("ξ by operator")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "box_xi_by_operator.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8,4))
    data2 = [d.loc[d["operator"]==op, "ds_plateau_mean"].values for op in order]
    plt.boxplot(data2, labels=order, showfliers=False)
    plt.ylabel("d_s plateau mean")
    plt.title("d_s plateau by operator")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "box_ds_by_operator.png"), dpi=160)
    plt.close()

def plot_heatmaps(df, outdir):
    # try heatmaps if grid is dense: pivot N x smooth_px
    d = df.dropna(subset=["N","smooth_px","xi_pixels","ds_plateau_mean"]).copy()
    if d.empty: return
    # require at least a small grid
    if d["N"].nunique() < 4 or d["smooth_px"].nunique() < 4:
        return
    for col, fname in [("xi_pixels","heat_xi.png"), ("ds_plateau_mean","heat_ds.png")]:
        piv = d.pivot_table(index="N", columns="smooth_px", values=col, aggfunc="mean")
        plt.figure(figsize=(7,5))
        im = plt.imshow(piv.values, aspect="auto", origin="lower",
                        extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()])
        plt.colorbar(im, label=col)
        plt.xlabel("smooth_px")
        plt.ylabel("N")
        plt.title(f"{col} (mean over operator)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=160)
        plt.close()

# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate meta-geometry runs and plot.")
    ap.add_argument("--roots", nargs="+", required=True, help="Root folders to scan (e.g., grid_runs grid_runs_nine grid_runs_nine1)")
    ap.add_argument("--outdir", required=True, help="Where to write the CSV and plots")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    # 1) Aggregate
    all_rows = []
    for r in args.roots:
        print(f"[SCAN] {r}")
        rows = collect_rows(r)
        print(f"  -> found {len(rows)} meta_summary.json")
        all_rows.extend(rows)

    if not all_rows:
        print("[ERROR] No meta_summary.json found under given roots.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)

    # 2) Save CSV
    out_csv = os.path.join(args.outdir, "grid_meta_summary_all.csv")
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {len(df)} rows to {out_csv}")

    # 3) Text integrity summary (printed)
    key_cols = [
        "N","alpha","operator","smooth_px","sigma_graph","rot_deg","windowsize",
        "xi_pixels","xi_logfit_R2","ds_plateau_mean","ds_plateau_spread","ds_max","ds_median"
    ]
    print("\n" + summarize_missing(df, key_cols) + "\n")

    # 4) Plots
    plot_xi_vs_N(df, args.outdir)
    plot_xi_vs_smooth(df, args.outdir)
    plot_ds_vs_N(df, args.outdir)
    plot_operator_boxplots(df, args.outdir)
    plot_heatmaps(df, args.outdir)

    # 5) Print quick convergence diagnostics
    fit_xi = fit_trend_xi_vs_N(df)
    fit_ds = fit_ds_extrap(df)
    print("=== Convergence diagnostics ===")
    if fit_xi:
        print(f"xi ~ a * N^p   with   a={fit_xi['a']:.3g},  p={fit_xi['p']:.4f},  R^2={fit_xi['R2']:.3f},  n={fit_xi['n']}")
    else:
        print("xi fit: insufficient data")

    if fit_ds:
        A1, R21 = fit_ds["model_1_over_N"]["A"], fit_ds["model_1_over_N"]["R2"]
        A2, R22 = fit_ds["model_1_over_sqrtN"]["A"], fit_ds["model_1_over_sqrtN"]["R2"]
        print(f"ds ≈ A + B/N         => A={A1:.4f}  (Einstein-limit extrapolate),  R^2={R21:.3f},  n={fit_ds['n']}")
        print(f"ds ≈ A + C/√N        => A={A2:.4f}  (Einstein-limit extrapolate),  R^2={R22:.3f},  n={fit_ds['n']}")
        print("Note: A close to 4.0 and increasing R^2 with larger N indicates approach to the Einstein smooth 4D limit.")
    else:
        print("ds fit: insufficient data")

    print(f"[DONE] Plots saved in: {args.outdir}")

if __name__ == "__main__":
    main()