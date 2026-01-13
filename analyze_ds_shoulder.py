#!/usr/bin/env python3
# analyze_ds_shoulder.py
# Runs three diagnostics to test for a spectral-dimension "shoulder":
# (1) smoothing-window robustness, (2) curvature/inflection, (3) two-segment BIC on log P.

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_ds(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df[['t','ds']].dropna().sort_values('t')
    df = df[df['t'] > 0]  # enforce positive t for logs
    return df

def smoothing_robustness_plot(t, ds, outpath, windows=(15, 21, 31, 41)):
    fig, ax = plt.subplots(figsize=(7.0,4.5), constrained_layout=True)
    for win in windows:
        win = int(win)
        if win % 2 == 0:
            win += 1
        # keep within valid range for savgol
        win = min(win, len(ds) - 1 if (len(ds) % 2 == 0) else len(ds))
        win = max(win, 5)
        if win >= len(ds):
            continue
        ds_smooth = savgol_filter(ds, window_length=win, polyorder=3)
        ax.plot(t, ds_smooth, label=f'window={win}')
    ax.plot(t, ds, color='k', alpha=0.15, linewidth=1, label='raw')
    ax.set_xscale('log')
    ax.set_xlabel("Diffusion scale $t$")
    ax.set_ylabel(r"$d_s(t)$")
    ax.set_title("Spectral dimension vs smoothing window")
    ax.legend()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def curvature_test(t, ds, sigma=3.0):
    logt = np.log10(t.values)
    ds_sm = gaussian_filter1d(ds.values, sigma)
    curv = np.gradient(np.gradient(ds_sm, logt), logt)  # d^2 ds / d(log t)^2
    idx_min = int(np.argmin(curv))
    return {
        "logt": logt,
        "ds_sm": ds_sm,
        "curv": curv,
        "idx_min": idx_min,
        "t_star": t.values[idx_min],
        "ds_star": ds.values[idx_min]
    }

def curvature_plot(curv_res, outpath):
    fig, ax = plt.subplots(figsize=(7.0,4.5), constrained_layout=True)
    logt = curv_res["logt"]
    curv = curv_res["curv"]
    idx = curv_res["idx_min"]
    ax.plot(logt, curv, 'r')
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.axvline(logt[idx], color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel(r'$\log_{10} t$')
    ax.set_ylabel(r'Curvature: $\frac{d^2 d_s}{d(\log t)^2}$')
    ax.set_title("Curvature/inflection test")
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def approx_logP_from_ds(t, ds):
    """
    ds = -2 d(ln P)/d(ln t)  =>  ln P ≈ -0.5 * ∫ ds d(ln t).
    Cumulative trapezoid in log-space up to a constant.
    """
    logt = np.log10(t.values)
    dlogt = np.gradient(logt)
    logP = -0.5 * np.cumsum(ds.values * dlogt)
    logP -= np.min(logP)  # normalize
    return logt, logP

def bic_two_segment_break(logt, logP, min_margin=20):
    # single-line model
    X = logt.reshape(-1,1)
    reg_all = LinearRegression().fit(X, logP)
    yhat_all = reg_all.predict(X)
    rss_all = mean_squared_error(logP, yhat_all)*len(logP)
    k_all = 2
    bic_all = len(logP)*np.log(rss_all/len(logP)) + k_all*np.log(len(logP))

    best = {"i": None, "bic": np.inf, "reg1": None, "reg2": None, "yhat": None}

    for i in range(min_margin, len(logt)-min_margin):
        x1, x2 = logt[:i].reshape(-1,1), logt[i:].reshape(-1,1)
        y1, y2 = logP[:i], logP[i:]
        reg1, reg2 = LinearRegression().fit(x1,y1), LinearRegression().fit(x2,y2)
        yhat = np.concatenate([reg1.predict(x1), reg2.predict(x2)])
        rss = mean_squared_error(logP, yhat)*len(logP)
        k = 4
        bic = len(logP)*np.log(rss/len(logP)) + k*np.log(len(logP))
        if bic < best["bic"]:
            best.update({"i": i, "bic": bic, "reg1": reg1, "reg2": reg2, "yhat": yhat})

    delta_bic = bic_all - best["bic"]
    return {
        "i": best["i"],
        "bic_all": bic_all,
        "bic_2seg": best["bic"],
        "delta_bic": float(delta_bic),
        "yhat_all": yhat_all,
        "yhat_2seg": best["yhat"],
        "reg_all": reg_all,
        "reg1": best["reg1"],
        "reg2": best["reg2"]
    }

def bic_plot(logt, logP, bic_res, t_vals, ds_vals, outpath):
    i = bic_res["i"]
    fig, ax1 = plt.subplots(figsize=(7.6,4.8), constrained_layout=True)
    ax1.plot(10**logt, logP, 'k', alpha=0.6, label='approx log P(t)')
    if i is not None and bic_res["yhat_2seg"] is not None:
        ax1.plot(10**logt, bic_res["yhat_2seg"], 'C3', lw=2, label='two-segment fit')
        ax1.axvline(10**logt[i], color='gray', ls=':', label='best break')
    else:
        ax1.plot(10**logt, bic_res["yhat_all"], 'b--', label='single linear fit')
    ax1.set_xscale('log')
    ax1.set_xlabel('Diffusion scale $t$')
    ax1.set_ylabel(r'approx. $\ln P(t)$')  # raw string to avoid \l warning
    ax1.set_title('Two-segment BIC fit on log P(t)')
    ax1.legend(loc='upper left')

    # Optional inset for d_s around the break
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_ins = inset_axes(ax1, width="40%", height="40%", loc='lower right', borderpad=1.2)
        ax_ins.plot(10**logt, ds_vals, 'C0', alpha=0.8)
        if i is not None:
            ax_ins.axvline(10**logt[i], color='gray', ls=':')
        ax_ins.set_xscale('log')
        ax_ins.set_title(r'$d_s(t)$ (zoom)', fontsize=9)
        ax_ins.set_xlabel('t', fontsize=8)
        ax_ins.set_ylabel(r'$d_s$', fontsize=8)
    except Exception:
        pass

    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def bic_strength(delta_bic: float) -> str:
    if delta_bic < 2:
        return "negligible"
    if delta_bic < 6:
        return "weak"
    if delta_bic < 10:
        return "moderate"
    return "strong"

def main():
    ap = argparse.ArgumentParser(description="Analyze spectral-dimension shoulder features.")
    ap.add_argument("--csv", required=True, help="Path to ds_mean.csv (columns: t,ds)")
    ap.add_argument("--outdir", default=".", help="Where to write figures and report")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_ds(args.csv)
    t = df['t']
    ds = df['ds']

    # 1) Smoothing-robustness
    fig1 = os.path.join(args.outdir, "smooth_windows.png")
    smoothing_robustness_plot(t, ds, fig1)

    # 2) Curvature/inflection
    curv_res = curvature_test(t, ds, sigma=3.0)
    fig2 = os.path.join(args.outdir, "curvature.png")
    curvature_plot(curv_res, fig2)

    # 3) Two-segment BIC on approx log P(t)
    logt, logP = approx_logP_from_ds(t, ds)
    bic_res = bic_two_segment_break(logt, logP, min_margin=20)
    fig3 = os.path.join(args.outdir, "bic_break.png")
    bic_plot(logt, logP, bic_res, t.values, ds.values, fig3)

    # Summaries
    t_star = curv_res["t_star"]
    ds_star = curv_res["ds_star"]
    delta_bic = float(bic_res["delta_bic"])
    strength = bic_strength(delta_bic)
    t_break = 10**logt[bic_res["i"]] if bic_res["i"] is not None else np.nan
    if not np.isnan(t_break):
        j = int(np.argmin(np.abs(t.values - t_break)))
        ds_break = float(ds.values[j])
    else:
        ds_break = np.nan

    report_lines = [
        "# Shoulder diagnostics report",
        "",
        f"Smoothing robustness plot: {os.path.basename(fig1)}",
        f"Curvature (inflection) plot: {os.path.basename(fig2)}",
        f"Two-segment BIC plot: {os.path.basename(fig3)}",
        "",
        "Key results:",
        f"- Curvature inflection (most concave point): t* ~ {t_star:.3e}, d_s(t*) ~ {ds_star:.3f}",
        f"- Two-segment break (BIC): t_break ~ {t_break:.3e}, d_s(t_break) ~ {ds_break:.3f}",
        f"- ΔBIC (one-line vs two-segment) = {delta_bic:.2f} => evidence: {strength}",
        "",
        "Interpretation guide:",
        "• If the shoulder/kink remains at a consistent t across smoothing windows, it’s likely physical.",
        "• A pronounced negative curvature peak near that t supports a real shoulder.",
        "• ΔBIC ≥ 10 is strong evidence for a real break in log P(t); 6–10 moderate; 2–6 weak; <2 negligible.",
    ]

    report_path = os.path.join(args.outdir, "shoulder_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:  # ensure UTF-8 on Windows
        f.write("\n".join(report_lines))

    # Console summary
    print("\n".join(report_lines))

if __name__ == "__main__":
    main()