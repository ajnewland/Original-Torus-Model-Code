#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Learn a geometry-only cycle weight w(condA,condB) so that
S_w = w*A + (1-w)*B ≈ TARGET (default 0.231), using your seesaw_ay_sweep.csv.

We evaluate three candidate laws:
  R1 (ratio)         : w = condA/(condA+condB)
  R2 (power-softmax) : w = condA^p / (condA^p + condB^p)         (fit p)
  R3 (logistic)      : w = sigmoid(c0 + c1 * ln(condA/condB))     (fit c0,c1)

Outputs:
  - symm_fit_summary.txt   (chosen rule + parameters + errors)
  - symm_fit_results.csv   (per-row predictions & errors)
  - PNGs: w_true_vs_pred.png, S_pred_vs_ay.png, w_true_vs_logratio.png
"""

import csv, math, os, sys

TARGET = 0.231
PLOT = True  # set False to skip PNGs

def sigm(x): return 1.0/(1.0+math.exp(-x))
def clamp01(x): return 0.0 if x<0 else 1.0 if x>1 else x

def read_rows(csv_path):
    rows=[]
    with open(csv_path, newline="") as f:
        rdr=csv.DictReader(f)
        for r in rdr:
            try:
                ay = float(r.get("ay"))
                A  = float(r.get("A_mean"))
                B  = float(r.get("B_mean"))
                cA = float(r.get("A_condG"))
                cB = float(r.get("B_condG"))
            except:
                continue
            rows.append(dict(ay=ay,A=A,B=B,cA=cA,cB=cB))
    return rows

def ideal_w(A,B):
    if A==B: return float('nan')
    w = (TARGET - B)/(A - B)
    return clamp01(w)

def mse(xs):
    return sum(v*v for v in xs)/len(xs) if xs else float('nan')

def fit_power_softmax(rows):
    # Fit p by minimizing MSE of S_pred to TARGET (1D search over p)
    # Use a simple coarse-to-fine sweep (robust & dependency-free)
    def w_p(cA,cB,p):
        if cA<=0 or cB<=0: return 0.5
        a = cA**p; b = cB**p
        return a/(a+b) if (a+b)>0 else 0.5
    best=(None, float('inf'))
    grid = [-4,-3,-2,-1,-0.5,-0.25,0,0.25,0.5,1,2,3,4,5,6]
    # coarse
    for p in grid:
        errs=[]
        for r in rows:
            w = w_p(r["cA"],r["cB"],p)
            S = w*r["A"] + (1-w)*r["B"]
            errs.append(S - TARGET)
        E = mse(errs)
        if E<best[1]: best=(p,E)
    # refine around best
    p0 = best[0]
    for step in [0.5,0.2,0.1,0.05,0.02]:
        cand = [p0-2*step,p0-step,p0,p0+step,p0+2*step]
        for p in cand:
            errs=[]
            for r in rows:
                w = w_p(r["cA"],r["cB"],p)
                S = w*r["A"] + (1-w)*r["B"]
                errs.append(S - TARGET)
            E = mse(errs)
            if E<best[1]: best=(p,E); p0=p
    return best[0]  # p*

def fit_logistic(rows):
    # Fit c0,c1 for w = sigmoid(c0 + c1 * ln(cA/cB)) by least squares on w_true (unclipped region preferred)
    # Build dataset
    X=[]; Y=[]
    for r in rows:
        if r["cA"]>0 and r["cB"]>0:
            x = math.log(r["cA"]/r["cB"])
            wt = ideal_w(r["A"],r["B"])
            if not math.isnan(wt):
                # keep all; logistic will clamp
                X.append([1.0, x])
                Y.append(wt)
    if len(X)<2:
        return 0.0, 0.0
    # Solve (X'X) beta = X'Y
    # 2x2 normal equations
    s11 = sum(x[0]*x[0] for x in X)
    s12 = sum(x[0]*x[1] for x in X)
    s22 = sum(x[1]*x[1] for x in X)
    t1  = sum(x[0]*y for x,y in zip(X,Y))
    t2  = sum(x[1]*y for x,y in zip(X,Y))
    det = s11*s22 - s12*s12
    if abs(det)<1e-12:
        return 0.0, 0.0
    c0 = ( t1*s22 - s12*t2)/det
    c1 = (-t1*s12 + s11*t2)/det
    return c0, c1

def evaluate_rules(rows, p_star, c0, c1):
    def w_ratio(r):
        cA,cB=r["cA"],r["cB"]
        if cA<=0 and cB<=0: return 0.5
        if cA<=0: return 0.0
        if cB<=0: return 1.0
        return cA/(cA+cB)
    def w_power(r):
        cA,cB=r["cA"],r["cB"]
        if cA<=0 and cB<=0: return 0.5
        if cA<=0: return 0.0
        if cB<=0: return 1.0
        a=cA**p_star; b=cB**p_star
        return a/(a+b) if (a+b)>0 else 0.5
    def w_logistic(r):
        cA,cB=r["cA"],r["cB"]
        if cA<=0 or cB<=0: return 0.5
        x=math.log(cA/cB)
        return sigm(c0 + c1*x)

    rules = [
        ("ratio",        w_ratio,    {}),
        ("power_softmax",w_power,    {"p":p_star}),
        ("logistic",     w_logistic, {"c0":c0,"c1":c1}),
    ]

    scores=[]
    for name, wfun, pars in rules:
        errs=[]; rows_out=[]
        for r in rows:
            w = clamp01(wfun(r))
            S = w*r["A"] + (1-w)*r["B"]
            e = S - TARGET
            errs.append(e)
            rcp = dict(r)
            rcp.update(rule=name, w_pred=w, S_pred=S, S_err=e)
            rows_out.append(rcp)
        scores.append((name, mse(errs), pars, rows_out))
    # sort by MSE
    scores.sort(key=lambda t:t[1])
    return scores  # list of (name, MSE, params, rows_out)

def write_csv(path, rows):
    if not rows: return
    keys = ["ay","A","B","cA","cB","rule","w_pred","S_pred","S_err"]
    with open(path,"w",newline="") as f:
        wr=csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k:r.get(k,"") for k in keys})

def main():
    if len(sys.argv)<2:
        print("Usage: python ew_symmetrized_cycles_fit.py <path\\to\\seesaw_ay_sweep.csv> [--no-plot]")
        sys.exit(1)
    csv_path = sys.argv[1]
    global PLOT
    if len(sys.argv)>2 and sys.argv[2]=="--no-plot":
        PLOT = False
    if not os.path.isfile(csv_path):
        print("File not found:", csv_path); sys.exit(2)

    rows = read_rows(csv_path)
    if not rows:
        print("No rows parsed."); sys.exit(3)

    # Fit parameters
    p_star = fit_power_softmax(rows)
    c0, c1 = fit_logistic(rows)

    # Evaluate rules
    scored = evaluate_rules(rows, p_star, c0, c1)
    best_name, best_mse, best_pars, best_rows = scored[0]

    out_dir = os.path.dirname(csv_path) or "."
    with open(os.path.join(out_dir,"symm_fit_summary.txt"),"w") as f:
        f.write("=== Symmetrized-cycles fit ===\n")
        f.write(f"TARGET sin2 = {TARGET}\n\n")
        f.write(f"R2 power-softmax: p* = {p_star:.4f}\n")
        f.write(f"R3 logistic     : c0 = {c0:.4f}, c1 = {c1:.4f}\n\n")
        for name,mse_val,pars,_rows in scored:
            f.write(f"{name:14s}  MSE={mse_val:.6e}  params={pars}\n")
        f.write("\nBest rule: %s  MSE=%.6e  params=%s\n" % (best_name, best_mse, best_pars))
    print(f"Wrote: {os.path.join(out_dir,'symm_fit_summary.txt')}")

    out_csv = os.path.join(out_dir,"symm_fit_results.csv")
    write_csv(out_csv, best_rows)
    print(f"Wrote: {out_csv}")

    # Optional plots
    if PLOT:
        try:
            import matplotlib.pyplot as plt
            # true w*
            w_true=[]; xlog=[]; ay=[]; w_pred=[]
            for r in best_rows:
                wt = ideal_w(r["A"],r["B"])
                if not math.isnan(wt):
                    w_true.append(wt)
                    w_pred.append(r["w_pred"])
                    ay.append(r["ay"])
                    if r["cA"]>0 and r["cB"]>0:
                        xlog.append(math.log(r["cA"]/r["cB"]))
                    else:
                        xlog.append(float('nan'))

            # w_true vs w_pred
            plt.figure(); plt.scatter(w_true, w_pred, s=24)
            plt.xlabel("w* (exact)"); plt.ylabel("w_pred (best rule)")
            plt.title(f"w_pred vs w*  [{best_name}]")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir,"w_true_vs_pred.png"), dpi=160); plt.close()

            # S_pred vs ay
            S_pred=[r["S_pred"] for r in best_rows]
            plt.figure(); plt.scatter(ay, S_pred, s=24)
            plt.axhline(TARGET, linestyle="--")
            plt.xlabel("ay"); plt.ylabel("S_pred")
            plt.title(f"S_pred across ay  [{best_name}]")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir,"S_pred_vs_ay.png"), dpi=160); plt.close()

            # w_true vs log(condA/condB)
            x = [v for v in xlog if not math.isnan(v)]
            w = [wt for wt,xv in zip(w_true,xlog) if not math.isnan(xv)]
            if len(x)>2:
                plt.figure(); plt.scatter(x, w, s=24)
                plt.xlabel("ln(condA/condB)"); plt.ylabel("w*")
                plt.title("w* vs log-cond ratio")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir,"w_true_vs_logratio.png"), dpi=160); plt.close()
        except Exception as e:
            print(f"(Plotting skipped: {e})")

if __name__ == "__main__":
    main()