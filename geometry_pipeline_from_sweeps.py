#!/usr/bin/env python3
"""
geometry_pipeline_from_sweeps.py

Usage (cmd.exe, one line):
  python geometry_pipeline_from_sweeps.py --outdir C:\path\to\out ^
    C:\...\ew_sweep_ax2p58_ay0p80\cycle_rows.csv ^
    C:\...\ew_sweep_ax2p58_ay0p81\cycle_rows.csv ^
    C:\...\ew_sweep_ax2p58_ay0p82\cycle_rows.csv

What it does:
1) Loads each ew_sweep CSV (with columns A,B,cA,cB,r,...).
   - If r is missing, computes r = ln(cA/cB), rho = cB/cA.
2) Learns geometry-only seesaw weights (k,b,delta) by minimizing
   the residual of S_pred = sigma(k*r+b)*A + (1-w)*B - delta
   versus the target S_tgt (default 0.231) across all files.
3) Writes:
   - summary.csv         (per-file A,B,cA,cB,r,S_pred,err,...)
   - kbd.json            (learned k,b,delta and S_tgt)
   - latent_z.csv        (per-file z = -(r - mean_r)/std_r)
"""

import argparse, csv, json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- utils ----------------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def read_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["__src__"] = path
    return df

def extract_globals(df: pd.DataFrame):
    """
    Each ew_sweep CSV repeats A,B,cA,cB per row; take the first.
    """
    A  = float(df["A"].iloc[0])   if "A"  in df.columns else float("nan")
    B  = float(df["B"].iloc[0])   if "B"  in df.columns else float("nan")
    cA = float(df["cA"].iloc[0])  if "cA" in df.columns else float("nan")
    cB = float(df["cB"].iloc[0])  if "cB" in df.columns else float("nan")
    r  = float(df["r"].iloc[0])   if "r"  in df.columns else (
         math.log((cA+1e-18)/(cB+1e-18)) if math.isfinite(cA) and math.isfinite(cB) else float("nan"))
    rho = float(df["rho"].iloc[0]) if "rho" in df.columns else (
          (cB+1e-18)/(cA+1e-18) if math.isfinite(cA) and math.isfinite(cB) else float("nan"))

    # Optional: pick up ax, ay if present
    ax = float(df["ax"].iloc[0]) if "ax" in df.columns else float("nan")
    ay = float(df["ay"].iloc[0]) if "ay" in df.columns else float("nan")
    return A, B, cA, cB, r, rho, ax, ay

def s_pred(A: float, B: float, r: float, k: float, b: float, delta: float) -> float:
    w = sigmoid(k*r + b)
    return w*A + (1.0 - w)*B - delta

def grid_search_kbd(records, S_tgt: float,
                    k_range=(0.5, 5.0), b_range=(-0.5, 0.5), d_range=(-0.005, 0.005),
                    steps=(60, 61, 51)):
    """
    Simple grid-search (coarse but robust) to learn (k,b,delta).
    records: list of dicts with A,B,r for each file.
    """
    k_min,k_max = k_range; b_min,b_max = b_range; d_min,d_max = d_range
    nk, nb, nd = steps

    best = None
    for k in np.linspace(k_min, k_max, nk):
        for b in np.linspace(b_min, b_max, nb):
            for d in np.linspace(d_min, d_max, nd):
                err2 = 0.0
                for rec in records:
                    A,B,r = rec["A"], rec["B"], rec["r"]
                    Sp = s_pred(A,B,r,k,b,d)
                    err2 += (Sp - S_tgt)**2
                if (best is None) or (err2 < best["err2"]):
                    best = {"k": float(k), "b": float(b), "delta": float(d), "err2": float(err2)}
    return best

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Output directory for summary.csv, kbd.json, latent_z.csv")
    ap.add_argument("--S_tgt", type=float, default=0.231, help="Target sin^2 (default 0.231)")
    ap.add_argument("--flip_z", action="store_true", default=True,
                    help="Use z = -(r - mean_r)/std_r (default True).")
    ap.add_argument("inputs", nargs="+", help="One or more ew_sweep CSVs (cycle_rows.csv or direct .csv outputs).")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # 1) Load all CSVs
    all_recs = []
    per_file_rows = []  # for summary.csv
    for p in args.inputs:
        df = read_one_csv(p)
        A,B,cA,cB,r,rho,ax,ay = extract_globals(df)
        src = p

        if not (math.isfinite(A) and math.isfinite(B) and math.isfinite(r)):
            print(f"[WARN] Skipping {p} (missing A/B/r).")
            continue

        all_recs.append({"src": src, "A": A, "B": B, "cA": cA, "cB": cB, "r": r, "rho": rho, "ax": ax, "ay": ay})

    if not all_recs:
        print("[ERROR] No valid inputs with A,B,r found.")
        sys.exit(2)

    # 2) Learn (k,b,delta) by minimizing residual vs S_tgt across files
    best = grid_search_kbd(all_recs, S_tgt=args.S_tgt)
    k, b, delta = best["k"], best["b"], best["delta"]

    # 3) Build per-file summary, compute predictions and errors
    r_values = []
    for rec in all_recs:
        A,B,r = rec["A"], rec["B"], rec["r"]
        Sp = s_pred(A,B,r,k,b,delta)
        err = Sp - args.S_tgt
        r_values.append(r)
        row = {
            "src": rec["src"], "ax": rec["ax"], "ay": rec["ay"],
            "A": A, "B": B, "cA": rec["cA"], "cB": rec["cB"],
            "r": r, "rho": rec["rho"],
            "S_pred": Sp, "err_vs_target": err
        }
        per_file_rows.append(row)

    # 4) Latent z from r (geometry-only, flipped & standardized)
    r_arr = np.array(r_values, float)
    r_mean = float(np.nanmean(r_arr))
    r_std  = float(np.nanstd(r_arr, ddof=0)) if np.isfinite(np.nanstd(r_arr)) and np.nanstd(r_arr) > 0 else 1.0
    zs = []
    for rec in all_recs:
        r = rec["r"]
        z = -(r - r_mean)/r_std if args.flip_z else (r - r_mean)/r_std
        zs.append({
            "src": rec["src"], "ax": rec["ax"], "ay": rec["ay"],
            "r": r, "z": float(z)
        })

    # 5) Write outputs
    summary_path = os.path.join(args.outdir, "summary.csv")
    pd.DataFrame(per_file_rows).to_csv(summary_path, index=False)

    kbd_path = os.path.join(args.outdir, "kbd.json")
    with open(kbd_path, "w", encoding="utf-8") as f:
        json.dump({
            "S_tgt": args.S_tgt,
            "k": k, "b": b, "delta": delta,
            "fit_err2": best["err2"],
            "r_mean": r_mean, "r_std": r_std,
            "z_rule": "z = -(r - r_mean)/r_std" if args.flip_z else "z = (r - r_mean)/r_std"
        }, f, indent=2)

    latent_path = os.path.join(args.outdir, "latent_z.csv")
    pd.DataFrame(zs).to_csv(latent_path, index=False)

    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {kbd_path}")
    print(f"[OK] wrote {latent_path}")
    print(f"[LEARNED] k={k:.4f}  b={b:.4f}  delta={delta:.6f}  (err2={best['err2']:.3e})")

if __name__ == "__main__":
    main()
