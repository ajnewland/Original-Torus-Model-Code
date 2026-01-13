#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare three geometric seesaw rules on ay-sweep data:
  (i)  power_softmax    : w = sigmoid(alpha * (ln cA - ln cB))
  (ii) logistic         : w = sigmoid(k * r + b), r = (cA - cB)/(cA + cB)
  (iii) logistic + bias : S_pred -> S_pred - delta  (global δ)

Input CSV columns (from ew_seesaw_ay_sweep.py):
  ay, A_mean, A_std, A_condG, B_mean, B_std, B_condG, S_mean, S_std,
  A_minus_B, abs_S_minus_target

Outputs:
  - seesaw_weight_fit_compare.csv (all predictions side-by-side)
  - seesaw_Spred_compare.png (S vs ay curves)
  - seesaw_error_compare.png (residuals vs ay)
  - seesaw_w_compare.png (weights vs ay)
"""

import csv, math, os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

def read_table(path):
    rows = []
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                rows.append({
                    "ay": float(r["ay"]),
                    "A": float(r.get("A_mean", r.get("A"))),
                    "B": float(r.get("B_mean", r.get("B"))),
                    "cA": float(r.get("A_condG", r.get("condG(A)", np.nan))),
                    "cB": float(r.get("B_condG", r.get("condG(B)", np.nan))),
                    "S":  float(r.get("S_mean", r.get("S", np.nan))),
                })
            except Exception:
                # skip malformed lines
                pass
    rows.sort(key=lambda t: t["ay"])
    return rows

def sigmoid(x):
    # numerically safe-ish
    x = np.clip(x, -60, 60)
    return 1.0/(1.0 + np.exp(-x))

def fit_power_softmax(rows, target):
    # w = sigmoid(alpha * (ln cA - ln cB))
    # choose alpha that minimizes sum |S_pred - target|
    xs = []
    AB = []
    for t in rows:
        if t["cA"]>0 and t["cB"]>0:
            xs.append(math.log(t["cA"]) - math.log(t["cB"]))
            AB.append((t["A"], t["B"]))
    xs = np.array(xs)
    def loss(alpha):
        w = sigmoid(alpha * xs)
        A = np.array([a for a,b in AB]); B = np.array([b for a,b in AB])
        Sp = w*A + (1-w)*B
        return np.mean(np.abs(Sp - target))
    # simple 1D search
    grid = np.linspace(0.1, 4.0, 200)
    vals = [loss(a) for a in grid]
    alpha = float(grid[int(np.argmin(vals))])
    return alpha

def contrast_ratio(cA, cB):
    # symmetric bounded contrast in [-1,1]
    if cA<=0 and cB<=0: return 0.0
    return (cA - cB) / (cA + cB) if (cA + cB)!=0 else 0.0

def fit_logistic(rows, target):
    # w = sigmoid(k * r + b), r = (cA - cB)/(cA + cB)
    xs, AB = [], []
    for t in rows:
        r = contrast_ratio(t["cA"], t["cB"])
        xs.append(r); AB.append((t["A"], t["B"]))
    xs = np.array(xs)

    def loss_kb(k, b):
        w = sigmoid(k*xs + b)
        A = np.array([a for a,b in AB]); B = np.array([b for a,b in AB])
        Sp = w*A + (1-w)*B
        return np.mean(np.abs(Sp - target))

    # small grid search around reasonable scales
    ks = np.linspace(0.2, 6.0, 120)
    bs = np.linspace(-0.5, 0.5, 101)
    best = (9e9, 0.0, 0.0)
    for k in ks:
        # quick coarse b search using a few points (speeds up)
        errs = [(loss_kb(k,b), b) for b in bs[::5]]
        b0 = min(errs, key=lambda z:z[0])[1]
        # refine around b0
        bgrid = np.linspace(b0-0.1, b0+0.1, 81)
        for b in bgrid:
            L = loss_kb(k,b)
            if L < best[0]:
                best = (L, k, b)
    _, kbest, bbest = best
    return float(kbest), float(bbest)

def evaluate(rows, target, alpha, k, b, delta=0.0):
    out = []
    for t in rows:
        ay, A, B, cA, cB, S = t["ay"], t["A"], t["B"], t["cA"], t["cB"], t["S"]

        # star (ideal) convex weight to hit the target exactly
        w_star = (target - B)/(A - B) if (A!=B) else float("nan")
        w_star = float(np.clip(w_star, 0.0, 1.0))
        S_star = w_star*A + (1-w_star)*B

        # power-softmax
        if cA>0 and cB>0:
            x = math.log(cA) - math.log(cB)
        else:
            x = 0.0
        w_pow = float(sigmoid(alpha * x))
        S_pow = w_pow*A + (1-w_pow)*B

        # logistic
        r = contrast_ratio(cA, cB)
        w_log = float(sigmoid(k*r + b))
        S_log = w_log*A + (1-w_log)*B

        # logistic + delta
        S_log_bias = S_log - delta

        out.append({
            "ay": ay, "A": A, "B": B, "cA": cA, "cB": cB, "S": S,
            "w_star": w_star, "S_star": S_star, "err_star": S_star - target,
            "w_pow": w_pow, "S_pow": S_pow, "err_pow": S_pow - target,
            "w_log": w_log, "S_log": S_log, "err_log": S_log - target,
            "w_log_bias": w_log, "S_log_bias": S_log_bias, "err_log_bias": S_log_bias - target
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to seesaw_ay_sweep.csv")
    ap.add_argument("--target", type=float, default=0.231, help="Target sin^2(theta_W)")
    ap.add_argument("--outdir", default=".", help="Directory for outputs")
    ap.add_argument("--no-plot", action="store_true", help="Disable PNG plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = read_table(args.csv)
    if not rows:
        print("No usable rows found in CSV.")
        sys.exit(1)

    target = args.target

    # learn parameters
    alpha = fit_power_softmax(rows, target)
    k, b   = fit_logistic(rows, target)

    # choose a global bias δ to minimize mean absolute error of logistic+δ
    tmp = evaluate(rows, target, alpha, k, b, delta=0.0)
    errs = np.array([t["S_log"] - target for t in tmp])
    # best L1 bias is the median of errors
    delta = float(np.median(errs))

    print("\n=== Learned parameters ===")
    print(f"(i)  power_softmax:  alpha = {alpha:.4f}  (w = sigmoid(alpha * (ln cA - ln cB)))")
    print(f"(ii) logistic:       k = {k:.4f}, b = {b:.4f}  (w = sigmoid(k*r + b))")
    print(f"(iii) +global bias:  delta = {delta:+.6f}  (S_pred -> S_pred - delta)")

    # final evaluation
    res = evaluate(rows, target, alpha, k, b, delta=delta)

    def mae(key):
        v = np.array([abs(t[key]) for t in res if np.isfinite(t[key])])
        return float(np.mean(v)), float(np.min(v)), float(np.max(v))

    mae_star = mae("err_star")
    mae_pow  = mae("err_pow")
    mae_log  = mae("err_log")
    mae_lbd  = mae("err_log_bias")

    print("\n=== Absolute error vs target (sin^2) ===")
    print(f"(ideal) using w*: mean|err|={mae_star[0]:.6f}, min|err|={mae_star[1]:.6f}, max|err|={mae_star[2]:.6f}")
    print(f"(i)     power_softmax: mean|err|={mae_pow[0]:.6f}, min|err|={mae_pow[1]:.6f}, max|err|={mae_pow[2]:.6f}")
    print(f"(ii)    logistic(k,b): mean|err|={mae_log[0]:.6f}, min|err|={mae_log[1]:.6f}, max|err|={mae_log[2]:.6f}")
    print(f"(iii)   logistic+delta: mean|err|={mae_lbd[0]:.6f}, min|err|={mae_lbd[1]:.6f}, max|err|={mae_lbd[2]:.6f}")

    # write compare CSV
    out_csv = os.path.join(args.outdir, "seesaw_weight_fit_compare.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ay","A","B","cA","cB",
            "w_star","S_star","err_star",
            "w_pow","S_pow","err_pow",
            "w_log","S_log","err_log",
            "w_log_bias","S_log_bias","err_log_bias"
        ])
        for t in res:
            w.writerow([
                t["ay"], t["A"], t["B"], t["cA"], t["cB"],
                t["w_star"], t["S_star"], t["err_star"],
                t["w_pow"],  t["S_pow"],  t["err_pow"],
                t["w_log"],  t["S_log"],  t["err_log"],
                t["w_log_bias"], t["S_log_bias"], t["err_log_bias"]
            ])
    print(f"\nWrote: {out_csv}")

    if not args.no_plot:
        ay = np.array([t["ay"] for t in res])
        S  = np.array([t["S"]  for t in res])
        S_pow = np.array([t["S_pow"] for t in res])
        S_log = np.array([t["S_log"] for t in res])
        S_lbd = np.array([t["S_log_bias"] for t in res])

        # S vs ay
        plt.figure()
        plt.plot(ay, S, "o-", label="S (raw mean)")
        plt.plot(ay, S_pow, "o-", label="power_softmax")
        plt.plot(ay, S_log, "o-", label="logistic")
        plt.plot(ay, S_lbd, "o-", label="logistic+δ")
        plt.axhline(target, linestyle="--", linewidth=1)
        plt.xlabel("ay")
        plt.ylabel("sin^2")
        plt.title("Seesaw predictions vs ay")
        plt.legend()
        plt.tight_layout()
        png1 = os.path.join(args.outdir, "seesaw_Spred_compare.png")
        plt.savefig(png1, dpi=160); plt.close()
        print(f"Saved: {png1}")

        # residuals vs ay
        def resid(y): return y - target
        plt.figure()
        plt.plot(ay, resid(S), "o-", label="raw mean")
        plt.plot(ay, resid(S_pow), "o-", label="power_softmax")
        plt.plot(ay, resid(S_log), "o-", label="logistic")
        plt.plot(ay, resid(S_lbd), "o-", label="logistic+δ")
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("ay")
        plt.ylabel("S - target")
        plt.title("Residuals vs ay")
        plt.legend()
        plt.tight_layout()
        png2 = os.path.join(args.outdir, "seesaw_error_compare.png")
        plt.savefig(png2, dpi=160); plt.close()
        print(f"Saved: {png2}")

        # w vs ay
        w_star = np.array([t["w_star"] for t in res])
        w_pow  = np.array([t["w_pow"]  for t in res])
        w_log  = np.array([t["w_log"]  for t in res])
        plt.figure()
        plt.plot(ay, w_star, "o-", label="w* (ideal)")
        plt.plot(ay, w_pow,  "o-", label="power_softmax")
        plt.plot(ay, w_log,  "o-", label="logistic")
        plt.xlabel("ay")
        plt.ylabel("w")
        plt.title("Seesaw weights vs ay")
        plt.legend()
        plt.tight_layout()
        png3 = os.path.join(args.outdir, "seesaw_w_compare.png")
        plt.savefig(png3, dpi=160); plt.close()
        print(f"Saved: {png3}")

if __name__ == "__main__":
    main()