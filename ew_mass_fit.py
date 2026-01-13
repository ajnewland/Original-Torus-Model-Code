#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EW seesaw → Yukawas → masses (with uncertainties)

Inputs (CSV: species_channels.csv):
  species, ay, A, A_std, cA, B, B_std, cB[, seeds]

Optional calibration CSV (e.g., seesaw_ay_sweep.csv) with columns:
  ay, A_mean (A), B_mean (B), A_condG (cA), B_condG (cB), S_mean (S)
to learn logistic w = sigmoid(k*ln(cA/cB) + b) and optional global δ
minimizing |S_pred - S| with S_pred = w*A + (1-w)*B - δ.

Outputs:
  - species_masses.csv
  - species_masses.txt
  - (optional) species_yukawas.png, species_masses.png

Author: you + assistant
"""

import csv, math, os, sys, random, statistics
from typing import List, Dict, Tuple, Optional

# ---------- helpers ----------
def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def read_csv(path: str) -> List[Dict[str,str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def try_float(row: Dict[str,str], key: str, default=None):
    try:
        return float(row[key])
    except Exception:
        return default

def percentile(vals, p):
    if not vals:
        return float('nan')
    xs = sorted(vals)
    k = (len(xs)-1)*p/100.0
    f = math.floor(k); c = math.ceil(k)
    if f == c: return xs[int(k)]
    return xs[f] + (xs[c]-xs[f])*(k-f)

# ---------- fit logistic (k,b) and bias δ from calib CSV ----------
def fit_logistic_and_delta(
    calib_rows: List[Dict[str,str]],
    learn_delta: bool = True
) -> Tuple[float, float, float]:
    """
    Returns (k, b, delta). If learn_delta=False, delta=0.
    Fit k,b by minimizing squared error between S and S_pred without δ first,
    then compute global δ = mean(S_pred - S) if learn_delta.
    """
    # Build pairs (r, target S and A,B)
    rs, As, Bs, Ss = [], [], [], []
    for r in calib_rows:
        A = try_float(r, "A_mean") or try_float(r, "A") or try_float(r, "A_mean_val")
        B = try_float(r, "B_mean") or try_float(r, "B")
        cA = try_float(r, "A_condG") or try_float(r, "cA") or try_float(r, "condG(A)")
        cB = try_float(r, "B_condG") or try_float(r, "cB") or try_float(r, "condG(B)")
        S  = try_float(r, "S_mean") or try_float(r, "S")
        if None in (A,B,cA,cB,S) or cA<=0 or cB<=0:
            continue
        rs.append(math.log(cA/cB))
        As.append(A); Bs.append(B); Ss.append(S)

    if len(rs) < 3:
        # Not enough data, return defaults close to earlier good fit
        return 6.0, -0.6, 0.00864 if learn_delta else 0.0

    # Simple grid search for (k,b) (robust and fast for small calib sets)
    k_grid = [i/2 for i in range(0, 25)]  # 0..12 step .5
    b_grid = [i/20 - 1.5 for i in range(0, 61)]  # -1.5..+1.55 step .05

    best = (1e99, 6.0, -0.6)
    for k in k_grid:
        for b in b_grid:
            err2 = 0.0
            for r, A, B, S in zip(rs, As, Bs, Ss):
                w = sigmoid(k*r + b)
                Sp = w*A + (1-w)*B  # no delta during k,b fit
                e = Sp - S
                err2 += e*e
            if err2 < best[0]:
                best = (err2, k, b)
    _, k_opt, b_opt = best

    # Compute delta as mean residual (Sp - S)
    delta = 0.0
    if learn_delta:
        res = []
        for r, A, B, S in zip(rs, As, Bs, Ss):
            w = sigmoid(k_opt*r + b_opt)
            Sp = w*A + (1-w)*B
            res.append(Sp - S)
        delta = statistics.fmean(res) if res else 0.0

    return k_opt, b_opt, delta

# ---------- main compute pipeline ----------
def compute_species(
    rows: List[Dict[str,str]],
    k: float, b: float, delta: float,
    norm: str = "top",
    v_gev: float = 246.22,
    mc: int = 0,
    param_jitter: bool = False,
    jitter_kb: float = 0.10,
    jitter_delta: float = 0.20,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Returns (table, meta) where table has per-species results and meta has scale factors, etc.
    norm: 'top' -> y_top=1; 'unit' -> linear normalize S_eff to [0,1]; 'none' -> y=S_eff
    MC: if >0, returns 16-84% percentiles for S,y,m as *_lo,*_hi
    """
    recs = []
    for r in rows:
        sp = r["species"].strip()
        A  = try_float(r, "A");  B = try_float(r, "B")
        sA = try_float(r, "A_std") or 0.0
        sB = try_float(r, "B_std") or 0.0
        cA = try_float(r, "cA"); cB = try_float(r, "cB")
        ay = try_float(r, "ay")
        if None in (A,B,cA,cB) or cA<=0 or cB<=0:
            continue
        rlog = math.log(cA/cB)
        w    = sigmoid(k*rlog + b)
        Seff = w*A + (1-w)*B - delta
        # analytic uncertainty
        varS = (w*w)*(sA*sA) + ((1-w)*(1-w))*(sB*sB)
        sSeff = math.sqrt(varS)

        recs.append({
            "species": sp, "ay": ay,
            "A": A, "B": B, "cA": cA, "cB": cB,
            "w": w, "Seff": Seff, "Seff_sigma": sSeff
        })

    if not recs:
        raise RuntimeError("No valid species rows parsed.")

    # Aggregate by species (average over multiple ay rows per species)
    species = {}
    for t in recs:
        sp = t["species"]
        species.setdefault(sp, []).append(t)
    agg = []
    for sp, items in species.items():
        # mean of Seff, w, etc. (weighted by 1/σ^2 if available)
        if all(i["Seff_sigma"]>0 for i in items):
            wts = [1.0/(i["Seff_sigma"]**2) for i in items]
            W = sum(wts);
            Seff = sum(w*i["Seff"] for w,i in zip(wts,items))/W
            # conservative: keep sample stdev OR 1/sqrt(W)
            s_eff = math.sqrt(1.0/W)
            wbar = sum(i["w"] for i in items)/len(items)
            Abar = sum(i["A"] for i in items)/len(items)
            Bbar = sum(i["B"] for i in items)/len(items)
        else:
            Seff = sum(i["Seff"] for i in items)/len(items)
            s_eff = statistics.pstdev([i["Seff"] for i in items]) if len(items)>1 else items[0]["Seff_sigma"]
            wbar = sum(i["w"] for i in items)/len(items)
            Abar = sum(i["A"] for i in items)/len(items)
            Bbar = sum(i["B"] for i in items)/len(items)

        agg.append({"species": sp, "Seff": Seff, "Seff_sigma": s_eff,
                    "w": wbar, "Abar": Abar, "Bbar": Bbar, "n": len(items)})

    # Normalization to Yukawas
    if norm.lower() == "top":
        # find top row (species named 't')
        top = next((x for x in agg if x["species"].lower() in ("t","top","top_quark")), None)
        if top is None:
            raise RuntimeError("Top species ('t') not found for --norm top.")
        scale = 1.0/max(top["Seff"], 1e-12)
        offset = 0.0
    elif norm.lower() == "unit":
        smin = min(x["Seff"] for x in agg)
        smax = max(x["Seff"] for x in agg)
        rng  = max(smax - smin, 1e-12)
        scale = 1.0/rng
        offset = -smin
    else: # none
        scale = 1.0
        offset= 0.0

    # produce y and m
    for x in agg:
        y = (x["Seff"] + offset) * scale
        # propagate sigma to y linearly
        y_sigma = x["Seff_sigma"] * abs(scale)
        m = y * v_gev / math.sqrt(2.0)
        m_sigma = y_sigma * v_gev / math.sqrt(2.0)
        x.update({"y": y, "y_sigma": y_sigma, "m_GeV": m, "m_sigma_GeV": m_sigma})

    # Optional Monte-Carlo
    if mc and mc > 0:
        rng = random.Random(12345)
        for x in agg:
            # collect all matching raw rows to resample A,B per species
            items = species[x["species"]]
            S_draws = []
            for _ in range(mc):
                # optionally jitter k,b,delta
                kk, bb, dd = k, b, delta
                if param_jitter:
                    kk *= rng.lognormvariate(0.0, jitter_kb)  # multiplicative jitter
                    bb += rng.gauss(0.0, abs(b)*jitter_kb + 1e-6)  # small additive
                    dd += rng.gauss(0.0, abs(delta)*jitter_delta + 1e-6)
                # average Seff across this species' items per draw
                Sd = 0.0
                for it in items:
                    A = rng.gauss(it["A"], it["A"]*0.0 + try_float(it, "A_std") or 0.0)
                    B = rng.gauss(it["B"], it["B"]*0.0 + try_float(it, "B_std") or 0.0)
                    cA = it["cA"]; cB = it["cB"]
                    rlog = math.log(max(cA,1e-18)/max(cB,1e-18))
                    w = sigmoid(kk*rlog + bb)
                    Sd += (w*A + (1-w)*B - dd)
                Sd /= len(items)
                S_draws.append(Sd)
            lo = percentile(S_draws, 16)
            hi = percentile(S_draws, 84)
            x["Seff_lo"] = lo
            x["Seff_hi"] = hi
            # map to y,m
            y_lo  = (lo + offset)*scale
            y_hi  = (hi + offset)*scale
            m_lo  = y_lo * v_gev / math.sqrt(2.0)
            m_hi  = y_hi * v_gev / math.sqrt(2.0)
            x["y_lo"], x["y_hi"] = y_lo, y_hi
            x["m_lo_GeV"], x["m_hi_GeV"] = m_lo, m_hi

    meta = {"scale": scale, "offset": offset, "v_gev": v_gev,
            "k": k, "b": b, "delta": delta, "norm": norm}
    return agg, meta

# ---------- CLI ----------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="EW seesaw → Yukawas → masses")
    ap.add_argument("species_csv", help="species_channels.csv")
    ap.add_argument("--calib", help="optional seesaw_ay_sweep.csv to fit k,b,(δ)")
    ap.add_argument("--learn-delta", action="store_true", help="also learn global δ from calib CSV")
    ap.add_argument("--k", type=float, default=None, help="override logistic k")
    ap.add_argument("--b", type=float, default=None, help="override logistic b")
    ap.add_argument("--delta", type=float, default=None, help="override global δ")
    ap.add_argument("--norm", choices=["top","unit","none"], default="top", help="Yukawa normalization")
    ap.add_argument("--v", type=float, default=246.22, help="Higgs vev in GeV (default 246.22)")
    ap.add_argument("--mc", type=int, default=0, help="Monte-Carlo draws for 68% CI")
    ap.add_argument("--param-jitter", action="store_true", help="jitter (k,b,δ) during MC")
    ap.add_argument("--plot", action="store_true", help="save PNG plots")
    args = ap.parse_args()

    # Read species rows
    rows = read_csv(args.species_csv)

    # Determine logistic params
    k,b,delta = 6.0, -0.6, 0.00864  # solid defaults from your ax=2.59 series
    if args.calib:
        calib = read_csv(args.calib)
        k_fit, b_fit, d_fit = fit_logistic_and_delta(calib, learn_delta=args.learn_delta)
        k, b = k_fit, b_fit
        if args.learn_delta:
            delta = d_fit
    if args.k is not None: k = args.k
    if args.b is not None: b = args.b
    if args.delta is not None: delta = args.delta

    # Compute
    table, meta = compute_species(
        rows, k=k, b=b, delta=delta, norm=args.norm,
        v_gev=args.v, mc=args.mc, param_jitter=args.param_jitter
    )

    # Sort by mass desc
    table.sort(key=lambda x: x["m_GeV"], reverse=True)

    # Write CSV
    out_csv = os.path.join(os.path.dirname(args.species_csv) or ".", "species_masses.csv")
    flds = ["species","n","Seff","Seff_sigma","w","Abar","Bbar",
            "y","y_sigma","m_GeV","m_sigma_GeV"]
    # include MC if present
    if args.mc and args.mc>0:
        flds += ["Seff_lo","Seff_hi","y_lo","y_hi","m_lo_GeV","m_hi_GeV"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=flds)
        wr.writeheader()
        for x in table:
            wr.writerow({k: x.get(k, "") for k in flds})

    # Write TXT pretty
    out_txt = os.path.join(os.path.dirname(args.species_csv) or ".", "species_masses.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Logistic params: k={meta['k']:.4f}, b={meta['b']:.4f}, δ={meta['delta']:+.6f}\n")
        f.write(f"Normalization: {meta['norm']} ; scale={meta['scale']:.6g}, offset={meta['offset']:.6g}\n")
        f.write(f"Higgs vev v={meta['v_gev']} GeV; m = y*v/√2\n\n")
        head = ["species","n","Seff","±","y","±","m[GeV]","±"]
        f.write("{:>8} {:>3} {:>9} {:>7} {:>9} {:>7} {:>10} {:>10}\n".format(*head))
        for x in table:
            f.write("{:>8} {:>3d} {:>9.6f} {:>7.6f} {:>9.6f} {:>7.6f} {:>10.3f} {:>10.3f}\n".format(
                x["species"], x["n"], x["Seff"], x["Seff_sigma"],
                x["y"], x["y_sigma"], x["m_GeV"], x["m_sigma_GeV"]
            ))
        if args.mc and args.mc>0:
            f.write("\n(68% MC intervals are in CSV columns *_lo/*_hi)\n")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_txt}")

    # Optional plots
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            names = [x["species"] for x in table]
            ym    = [x["y"] for x in table]
            ms    = [x["m_GeV"] for x in table]

            plt.figure(figsize=(8,4.5))
            plt.bar(range(len(names)), ym)
            plt.xticks(range(len(names)), names, rotation=45, ha="right")
            plt.ylabel("Yukawa y")
            plt.title("Species Yukawas")
            plt.tight_layout()
            out_png = os.path.join(os.path.dirname(args.species_csv) or ".", "species_yukawas.png")
            plt.savefig(out_png, dpi=160); plt.close()

            plt.figure(figsize=(8,4.5))
            plt.bar(range(len(names)), ms)
            plt.xticks(range(len(names)), names, rotation=45, ha="right")
            plt.ylabel("Mass [GeV]")
            plt.title("Species Masses (m = y v/√2)")
            plt.tight_layout()
            out_png = os.path.join(os.path.dirname(args.species_csv) or ".", "species_masses.png")
            plt.savefig(out_png, dpi=160); plt.close()
            print("Saved plots: species_yukawas.png, species_masses.png")
        except Exception as e:
            print(f"(Plotting skipped: {e})")

if __name__ == "__main__":
    main()