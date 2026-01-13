#!/usr/bin/env python3
# lqgsm_fit.py
# Self-contained isotonic mass fitting with optional z-flip.
# - No external deps beyond numpy, matplotlib (only if --plot)
# - Implements PAVA for increasing/decreasing fits
# - Supports auto monotonic direction detection
# - Writes predicted_masses.csv (+ optional plots)

import argparse, csv, math, sys, os, statistics, random
from typing import List, Tuple, Dict, Optional
import numpy as np

# ---------- Utilities ----------

def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv_rows(path: str, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    # simple Spearman using rankdata via numpy argsort twice
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx.astype(float); ry = ry.astype(float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))

# ---------- PAVA isotonic regression ----------

def pava(y: np.ndarray, w: Optional[np.ndarray] = None, increasing: bool = True) -> np.ndarray:
    """
    Pool-Adjacent-Violators Algorithm.
    Returns the fitted y* (same order as input y).
    If increasing=False, performs decreasing isotonic fit.
    """
    n = len(y)
    if n == 0:
        return y.copy()
    if w is None:
        w = np.ones(n, dtype=float)
    y = y.astype(float).copy()
    w = w.astype(float).copy()

    # For decreasing fit, flip signs to reuse same logic
    sgn = 1.0 if increasing else -1.0
    y_work = sgn * y

    # Initialize blocks
    level = y_work.copy()
    weight = w.copy()
    # Each point starts its own block
    i = 0
    while i < n - 1:
        if level[i] <= level[i+1] + 1e-15:
            i += 1
            continue
        # Violation: pool blocks
        new_level = (weight[i]*level[i] + weight[i+1]*level[i+1]) / (weight[i] + weight[i+1])
        new_weight = weight[i] + weight[i+1]
        level[i] = new_level; weight[i] = new_weight

        # Shift everything left by removing i+1th block
        level = np.delete(level, i+1); weight = np.delete(weight, i+1)

        # Backtrack while previous violations exist
        j = i
        while j > 0 and level[j-1] > level[j] + 1e-15:
            new_level = (weight[j-1]*level[j-1] + weight[j]*level[j]) / (weight[j-1] + weight[j])
            new_weight = weight[j-1] + weight[j]
            level[j-1] = new_level; weight[j-1] = new_weight
            level = np.delete(level, j); weight = np.delete(weight, j)
            j -= 1
        i = j

    # Now expand block-averaged levels back to pointwise
    # We need to reconstruct mapping; easiest is second pass
    y_fit = np.empty(n, dtype=float)
    # Re-run to assign block values
    # Build boundaries by scanning again
    idx = 0
    # Make arrays of (level, count)
    # To rebuild, we need counts. Let's reconstruct by greedily assigning.
    # We'll redo PAVA but record block sizes.
    # Simpler: run a second pass to produce block lengths.
    # We'll implement a stack-based PAVA that stores blocks.
    # For correctness, rewrite with explicit block stack:

def pava_with_blocks(y: np.ndarray, w: Optional[np.ndarray] = None, increasing: bool = True) -> np.ndarray:
    if w is None:
        w = np.ones_like(y, dtype=float)
    sgn = 1.0 if increasing else -1.0
    z = sgn * y
    blocks = []  # each block: (sum_w, avg)
    for yi, wi in zip(z, w):
        # create new block
        sumw = wi
        avg = yi
        blocks.append([sumw, avg])
        # pool while violation
        while len(blocks) >= 2 and blocks[-2][1] > blocks[-1][1] + 1e-15:
            w1, a1 = blocks[-2]
            w2, a2 = blocks[-1]
            blocks.pop(); blocks.pop()
            w12 = w1 + w2
            a12 = (w1*a1 + w2*a2)/w12
            blocks.append([w12, a12])
    # expand back
    y_fit = []
    for sumw, avg in blocks:
        count = int(round(sumw / (w[0] if np.allclose(w, w[0]) else 1.0)))  # rough; fallback below
        # Use a safer approach: we don't know point counts if w varies; keep a parallel list
        # Instead, rebuild by a second loop distributing by original order — not tracked here.
        # We'll re-run a second pass that assigns block avgs while pooling; simpler approach:

    # Simpler, deterministic approach: run PAVA in "online" mode assigning outputs.
    # We’ll implement a stable algorithm returning fitted values directly.

def isotonic_fit(x: np.ndarray, y: np.ndarray, increasing: bool = True, w: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs isotonic regression on (x, y). Returns (x_sorted, y_fit_sorted).
    """
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    ws = np.ones_like(ys) if w is None else w[order].astype(float)

    # Stack of blocks: each block keeps sum_w, sum_wy, start_idx, end_idx
    sum_w = []
    sum_wy = []
    start = []
    end = []

    def block_avg(i):
        return sum_wy[i] / sum_w[i]

    for i in range(len(xs)):
        sum_w.append(ws[i])
        sum_wy.append(ws[i]*ys[i])
        start.append(i)
        end.append(i)
        # pool violations
        while len(sum_w) >= 2:
            a1 = (block_avg(len(sum_w)-2))
            a2 = (block_avg(len(sum_w)-1))
            # for decreasing fit, reverse inequality by flipping sign logic:
            ok = (a1 <= a2 + 1e-15) if increasing else (a1 >= a2 - 1e-15)
            if ok:
                break
            # merge last two
            i2 = len(sum_w)-1; i1 = len(sum_w)-2
            sw = sum_w[i1] + sum_w[i2]
            swy = sum_wy[i1] + sum_wy[i2]
            s = start[i1]
            e = end[i2]
            # delete last two, append merged
            for arr in (sum_w, sum_wy, start, end):
                arr.pop(); arr.pop()
            sum_w.append(sw); sum_wy.append(swy); start.append(s); end.append(e)

    # expand
    y_fit_sorted = np.empty_like(ys, dtype=float)
    for sw, swy, s, e in zip(sum_w, sum_wy, start, end):
        val = swy / sw
        y_fit_sorted[s:e+1] = val

    return xs, y_fit_sorted

def predict_from_isotonic(x_train: np.ndarray, yfit_train: np.ndarray, x_query: np.ndarray, increasing: bool = True) -> np.ndarray:
    """
    Piecewise-constant (right-continuous) isotonic predictor with linear edge extrapolation.
    """
    # x_train must be sorted and yfit aligned
    idx = np.argsort(x_train)
    xt = x_train[idx]; yt = yfit_train[idx]

    # Build piecewise-constant steps at each xt, with linear extrapolation at ends
    yq = np.empty_like(x_query, dtype=float)
    for i, x in enumerate(x_query):
        if x <= xt[0]:
            # linear extrapolation using first two points
            if len(xt) >= 2:
                slope = (yt[1]-yt[0]) / (xt[1]-xt[0] + 1e-18)
                yq[i] = yt[0] + slope*(x - xt[0])
            else:
                yq[i] = yt[0]
        elif x >= xt[-1]:
            if len(xt) >= 2:
                slope = (yt[-1]-yt[-2]) / (xt[-1]-xt[-2] + 1e-18)
                yq[i] = yt[-1] + slope*(x - xt[-1])
            else:
                yq[i] = yt[-1]
        else:
            j = np.searchsorted(xt, x, side="right") - 1
            yq[i] = yt[j]
    return yq

# ---------- Core pipeline ----------

def fit_isotonic_from_anchors(anchors_path: str, flip_z: bool, monotonic: str, bootstrap: int = 0, seed: int = 0):
    rows = read_csv_rows(anchors_path)
    if not rows:
        raise RuntimeError("No rows in anchors CSV.")
    sp = []
    z = []
    pdg = []
    for r in rows:
        sp.append(r.get("sp","").strip())
        zval = float(r["z"])
        z.append(-zval if flip_z else zval)
        pdg.append(float(r["PDG_GeV"]))
    sp = np.array(sp)
    z = np.array(z, dtype=float)
    y = np.log(np.array(pdg, dtype=float))  # target in log mass

    # choose monotonic direction
    if monotonic == "auto":
        rho = spearman_rank_corr(z, y)
        inc = True if rho >= 0 else False
    else:
        inc = (monotonic == "increasing")

    xs, yfit_sorted = isotonic_fit(z, y, increasing=inc)
    # Also keep the mapping from sorted xs to yfit
    model = {"x_sorted": xs, "yfit_sorted": yfit_sorted, "increasing": inc}

    # Bootstrap (optional)
    boot_sigma = None
    if bootstrap and bootstrap > 1:
        rng = random.Random(seed)
        yfits = []
        n = len(z)
        for _ in range(bootstrap):
            idxs = [rng.randrange(0, n) for __ in range(n)]
            xb = z[idxs]; yb = y[idxs]
            xsb, yfs = isotonic_fit(xb, yb, increasing=inc)
            # Evaluate back on original anchor z for a consistent sigma estimate
            pred = predict_from_isotonic(xsb, yfs, z, increasing=inc)
            yfits.append(pred)
        yfits = np.stack(yfits, axis=0)  # [B, n]
        boot_sigma = np.std(yfits, axis=0)  # sigma on log-mass at anchor points
    return model, (sp, z, y, boot_sigma)

def predict_to_csv(model, inputs_path: Optional[str], out_csv: str, anchors_info, include_anchors: bool = True):
    x_sorted = model["x_sorted"]; yfit_sorted = model["yfit_sorted"]; inc = model["increasing"]
    rows_out = []

    # Optional: predict for arbitrary inputs
    if inputs_path and os.path.exists(inputs_path):
        pts = read_csv_rows(inputs_path)
        for r in pts:
            spn = r.get("sp","").strip() or "?"
            zq = float(r["z"])
            mq_log = predict_from_isotonic(x_sorted, yfit_sorted, np.array([zq]), increasing=inc)[0]
            m = float(np.exp(mq_log))
            rows_out.append({"sp": spn, "z": zq, "m_pred_GeV": f"{m:.9g}", "source": "inputs"})
    # Include anchors back (pred and PDG)
    if include_anchors and anchors_info is not None:
        sp, z, y_log, boot_sigma = anchors_info
        y_pred_log = predict_from_isotonic(x_sorted, yfit_sorted, z, increasing=inc)
        for i in range(len(sp)):
            m_pred = float(np.exp(y_pred_log[i]))
            m_pdg = float(np.exp(y_log[i]))
            ratio = m_pred / m_pdg if m_pdg > 0 else float("nan")
            sigma = float(boot_sigma[i]) if boot_sigma is not None else float("nan")
            rows_out.append({
                "sp": sp[i],
                "z": f"{z[i]:.9g}",
                "m_pred_GeV": f"{m_pred:.9g}",
                "PDG_GeV": f"{m_pdg:.9g}",
                "ratio": f"{ratio:.6g}",
                "sigma_logm_boot": f"{sigma:.6g}",
                "source": "anchor"
            })

    # sort anchors first, then inputs
    rows_out.sort(key=lambda r: (0 if r["source"]=="anchor" else 1, r["sp"]))
    fields = ["sp","z","m_pred_GeV","PDG_GeV","ratio","sigma_logm_boot","source"]
    write_csv_rows(out_csv, fields, rows_out)
    return out_csv

def maybe_make_plots(model, anchors_info, out_prefix: str):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available; skipping plots ({e})", file=sys.stderr)
        return []

    x_sorted = model["x_sorted"]; yfit_sorted = model["yfit_sorted"]
    sp, z, y_log, boot_sigma = anchors_info
    # 1) z vs logmass fit (step plot)
    plt.figure()
    plt.plot(x_sorted, yfit_sorted, drawstyle="steps-post")
    plt.scatter(z, y_log)
    plt.xlabel("z (possibly flipped)")
    plt.ylabel("log(mass GeV)")
    plt.title("Isotonic fit: z → log(m)")
    fig1 = f"{out_prefix}_z_vs_logmass_fit.png"
    plt.tight_layout(); plt.savefig(fig1, dpi=160); plt.close()

    # 2) mass barplot (anchors, pred vs PDG)
    y_pred_log = predict_from_isotonic(x_sorted, yfit_sorted, z, increasing=model["increasing"])
    m_pred = np.exp(y_pred_log); m_pdg = np.exp(y_log)
    labels = list(sp)
    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, m_pdg, width, label="PDG")
    plt.bar(x + width/2, m_pred, width, label="Pred")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Mass (GeV)")
    plt.title("Predicted vs PDG (anchors)")
    plt.legend()
    fig2 = f"{out_prefix}_mass_barplot.png"
    plt.tight_layout(); plt.savefig(fig2, dpi=160); plt.close()

    return [fig1, fig2]

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="LQG–SM isotonic mass fit with optional z-flip.")
    ap.add_argument("--anchors", required=True, help="Path to anchors CSV with columns sp,z,PDG_GeV")
    ap.add_argument("--inputs", default=None, help="Optional CSV with columns sp,z to predict")
    ap.add_argument("--flip-z", action="store_true", help="Flip z → -z before fitting")
    ap.add_argument("--monotonic", choices=["increasing","decreasing","auto"], default="auto",
                    help="Monotonic direction for isotonic fit (default: auto)")
    ap.add_argument("--bootstrap", type=int, default=0, help="Bootstrap resamples for sigma on anchors (0=off)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap")
    ap.add_argument("--plot", action="store_true", help="Save fit plots")
    ap.add_argument("--out", default="predicted_masses.csv", help="Output CSV path")
    args = ap.parse_args()

    model, anchors_info = fit_isotonic_from_anchors(
        anchors_path=args.anchors,
        flip_z=args.flip_z,
        monotonic=args.monotonic,
        bootstrap=args.bootstrap,
        seed=args.seed
    )

    out_csv = predict_to_csv(model, args.inputs, args.out, anchors_info, include_anchors=True)
    print(f"[ok] wrote: {out_csv}")

    if args.plot:
        pics = maybe_make_plots(model, anchors_info, out_prefix=os.path.splitext(args.out)[0])
        for p in pics:
            print(f"[ok] figure: {p}")

if __name__ == "__main__":
    main()