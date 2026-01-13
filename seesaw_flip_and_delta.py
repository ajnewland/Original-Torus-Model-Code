#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seesaw: fit a geometric mixing weight w from cycle-conditioners
and compare three predictions vs the electroweak target:
  (a) ideal w* (upper bound; needs A,B)
  (b) logistic on a chosen feature of (cA,cB)
  (c) logistic + global bias delta

INPUT CSV must have (name tolerant):
  ay, A or A_mean, B or B_mean,
  A_condG (aka cA), B_condG (aka cB)

USAGE
  python seesaw_flip_and_delta.py file1.csv [file2.csv ...]
    [--feature {logratio,ratio}] [--no-delta] [--target 0.231]
    [--outdir OUTDIR] [--no-plot]

Outputs per file:
  <stem>_flip_delta.csv     (augmented table)
  <stem>_flip_delta.png     (residual plots)

Plus a summary CSV and a short text report beside the FIRST file.
"""

import argparse, csv, math, os, sys
import numpy as np

TARGET_DEFAULT = 0.231

# ----------------------------- IO helpers -----------------------------------
PREF_KEYS = {
    "ay": ["ay"],
    "A": ["A", "A_mean", "A_val"],
    "B": ["B", "B_mean", "B_val"],
    "cA": ["A_condG", "condG(A)", "cA", "condA"],
    "cB": ["B_condG", "condG(B)", "cB", "condB"],
}

def read_csv_rows(path):
    rows=[]
    with open(path, newline="", encoding="utf-8") as f:
        rdr=csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows

def pick(row, keys, cast=float, default=None):
    for k in keys:
        if k in row and row[k] not in ("", None, "nan", "NaN"):
            try: return cast(row[k])
            except Exception: pass
    return default

def extract_arrays(rows):
    ay=[]; A=[]; B=[]; cA=[]; cB=[]
    for r in rows:
        _ay = pick(r, PREF_KEYS["ay"])
        _A  = pick(r, PREF_KEYS["A"])
        _B  = pick(r, PREF_KEYS["B"])
        _cA = pick(r, PREF_KEYS["cA"])
        _cB = pick(r, PREF_KEYS["cB"])
        if None in (_ay,_A,_B,_cA,_cB):  # skip malformed
            continue
        ay.append(_ay); A.append(_A); B.append(_B); cA.append(_cA); cB.append(_cB)
    return map(np.asarray, (ay,A,B,cA,cB))

# ----------------------------- math utils -----------------------------------
def sigmoid(z):
    z = np.asarray(z, float)
    # guard extreme values
    z = np.clip(z, -60.0, 60.0)
    return 1.0/(1.0+np.exp(-z))

def standardize(x):
    x = np.asarray(x, float)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if not (sd > 0): sd = 1.0
    return (x - mu)/sd, mu, sd

def soften01(w, eps=1e-3):
    return np.clip(np.asarray(w, float), eps, 1.0-eps)

# ----------------------------- fitting --------------------------------------
def build_feature(cA, cB, mode="logratio"):
    cA = np.asarray(cA, float); cB = np.asarray(cB, float)
    # keep strictly positive to avoid nan/inf
    cA = np.maximum(cA, 1e-12); cB = np.maximum(cB, 1e-12)
    if mode == "ratio":
        x = cA / cB
    elif mode == "logratio":
        x = np.log(cA / cB)
    else:
        raise ValueError("unknown feature mode")
    return x

def fit_logistic(x_raw, w_star):
    """Fit w ≈ sigmoid(k * x_std + b) with bounds; returns (k,b,mu,sd)."""
    from scipy.optimize import curve_fit
    x_std, mu, sd = standardize(x_raw)
    y = soften01(w_star, 1e-3)

    def model(x, k, b):
        return sigmoid(k*x + b)

    # sensible bounds to avoid runaway fits
    bounds = ([-12.0, -3.0], [12.0, 3.0])
    k0, b0 = 4.0, 0.0
    popt, _ = curve_fit(model, x_std, y, p0=(k0,b0), bounds=bounds, maxfev=20000)
    k,b = float(popt[0]), float(popt[1])
    return k,b,mu,sd

# ----------------------------- core pipeline --------------------------------
def process_one(path, target, feature_mode, model_params=None, apply_delta=True, outdir=None, make_plot=True):
    rows = read_csv_rows(path)
    ay,A,B,cA,cB = extract_arrays(rows)
    if len(ay)==0:
        raise RuntimeError(f"No valid rows in {path}")

    # Ideal weight (upper bound diagnostic)
    w_star_raw = (target - B) / (A - B)
    w_star = np.clip(w_star_raw, 0.0, 1.0)
    S_star = w_star*A + (1.0 - w_star)*B
    err_star = S_star - target

    # Feature & logistic fit/predict
    x = build_feature(cA, cB, feature_mode)

    if model_params is None:
        k,b,xmu,xsd = fit_logistic(x, w_star)
        fitted_here = True
    else:
        k,b,xmu,xsd = model_params
        fitted_here = False

    x_std = (x - xmu)/ (xsd if xsd>0 else 1.0)
    w_log = sigmoid(k*x_std + b)
    S_log = w_log*A + (1.0 - w_log)*B
    err_log = S_log - target

    # single global bias per file (if enabled)
    delta = float(np.mean(err_log)) if apply_delta else 0.0
    S_log_bias = S_log - delta
    err_log_bias = S_log_bias - target

    # write per-file CSV
    stem = os.path.splitext(os.path.basename(path))[0]
    outdir = outdir or os.path.dirname(path) or "."
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, f"{stem}_flip_delta.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["ay","A","B","cA","cB","feature", "w_star","S_star","err_star",
                     "w_log","S_log","err_log","delta","S_log_bias","err_log_bias"])
        for i in range(len(ay)):
            wr.writerow([float(ay[i]), float(A[i]), float(B[i]),
                         float(cA[i]), float(cB[i]), float(x[i]),
                         float(w_star[i]), float(S_star[i]), float(err_star[i]),
                         float(w_log[i]), float(S_log[i]), float(err_log[i]),
                         delta, float(S_log_bias[i]), float(err_log_bias[i])])

    # quick aggregates for summary
    agg = {
        "file": path,
        "k": k, "b": b, "xmu": xmu, "xsd": xsd, "delta": delta,
        "mean_abs_err_star": float(np.mean(np.abs(err_star))),
        "mean_abs_err_log": float(np.mean(np.abs(err_log))),
        "mean_abs_err_log_bias": float(np.mean(np.abs(err_log_bias))),
        "min_abs_err_log_bias": float(np.min(np.abs(err_log_bias))),
        "max_abs_err_log_bias": float(np.max(np.abs(err_log_bias))),
        "fitted_here": fitted_here
    }

    # plot
    out_png = os.path.join(outdir, f"{stem}_flip_delta.png")
    if make_plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,5))
            plt.plot(ay, err_star, "o-", label="ideal")
            plt.plot(ay, err_log, "o-", label="logistic")
            plt.plot(ay, err_log_bias, "o-", label="logistic+δ")
            plt.axhline(0.0, ls="--", c="k", lw=1)
            plt.xlabel("ay")
            plt.ylabel("Residuals (S - target)")
            plt.title(f"Residuals vs ay  [{feature_mode}]")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_png, dpi=160)
            plt.close()
        except Exception as e:
            print(f"(plot skipped: {e})")

    return agg, (k,b,xmu,xsd), out_csv, out_png

# --------------------------------- CLI --------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Seesaw logistic fit with feature choice and global delta.")
    ap.add_argument("files", nargs="+", help="CSV files to analyze")
    ap.add_argument("--feature", choices=["logratio","ratio"], default="logratio",
                    help="Feature of (cA,cB) to feed logistic (default: logratio)")
    ap.add_argument("--no-delta", action="store_true", help="Disable global bias alignment")
    ap.add_argument("--target", type=float, default=TARGET_DEFAULT, help="Target sin^2 value (default 0.231)")
    ap.add_argument("--outdir", default=None, help="Optional output directory (default: alongside inputs)")
    ap.add_argument("--no-plot", action="store_true", help="Skip PNG plots")
    args = ap.parse_args()

    # Fit on the first file; reuse parameters for any additional files
    agg_all = []
    first_agg, params, out_csv, out_png = process_one(
        args.files[0], args.target, args.feature, model_params=None,
        apply_delta=not args.no_delta, outdir=args.outdir, make_plot=not args.no_plot
    )
    agg_all.append(first_agg)
    print(f"Fitted logistic on first file: k={first_agg['k']:.4f}, b={first_agg['b']:.4f}  (feature={args.feature})")

    for extra in args.files[1:]:
        agg, _, c, p = process_one(
            extra, args.target, args.feature, model_params=params,
            apply_delta=not args.no_delta, outdir=args.outdir, make_plot=not args.no_plot
        )
        agg_all.append(agg)

    # Summary CSV + text
    base_dir = args.outdir or (os.path.dirname(args.files[0]) or ".")
    sum_csv = os.path.join(base_dir, os.path.splitext(os.path.basename(args.files[0]))[0] + "_flip_delta_summary.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file","feature","k","b","xmu","xsd","delta",
                     "mean|err|_ideal","mean|err|_log","mean|err|_log+delta",
                     "min|err|_log+delta","max|err|_log+delta","fit_on_this_file"])
        for a in agg_all:
            wr.writerow([a["file"], args.feature, a["k"], a["b"], a["xmu"], a["xsd"], a["delta"],
                         a["mean_abs_err_star"], a["mean_abs_err_log"], a["mean_abs_err_log_bias"],
                         a["min_abs_err_log_bias"], a["max_abs_err_log_bias"], a["fitted_here"]])
    # Short text report (ASCII-safe)
    rep_txt = os.path.join(base_dir, os.path.splitext(os.path.basename(args.files[0]))[0] + "_flip_delta_report.txt")
    with open(rep_txt, "w", encoding="utf-8") as f:
        f.write("Seesaw logistic fit report\n")
        f.write(f"Feature          : {args.feature}\n")
        f.write(f"Target (sin^2)   : {args.target:.6f}\n")
        f.write(f"Delta applied    : {not args.no_delta}\n")
        f.write(f"Fit parameters   : k={first_agg['k']:.6f}, b={first_agg['b']:.6f}, xmu={first_agg['xmu']:.6f}, xsd={first_agg['xsd']:.6f}\n")
        f.write(f"Mean|err| ideal  : {first_agg['mean_abs_err_star']:.6f}\n")
        f.write(f"Mean|err| logist : {first_agg['mean_abs_err_log']:.6f}\n")
        f.write(f"Mean|err| log+δ  : {first_agg['mean_abs_err_log_bias']:.6f}\n")

    print(f"Wrote: {sum_csv}")
    print(f"Wrote: {rep_txt}")

if __name__ == "__main__":
    main()