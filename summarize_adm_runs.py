#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize per-run ADM results under a parent ensemble folder.

Looks for: run_XXX/adm_t0t2_patches/summary.csv
Outputs:
  ensemble_adm_summary.csv
  ensemble_adm_summary.pdf
"""
import os, sys, glob, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def main():
    import argparse
    ap = argparse.ArgumentParser("Summarize per-run ADM results")
    ap.add_argument("--ensemble_dir", required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    if args.outdir is None:
        args.outdir = args.ensemble_dir
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for p in sorted(glob.glob(os.path.join(args.ensemble_dir, "run_*", "adm_t0t2_patches", "summary.csv"))):
        df = pd.read_csv(p)
        if len(df):
            row = df.iloc[0].to_dict()
            row["run"] = p.split(os.sep)[-3]
            rows.append(row)

    if not rows:
        print("No per-run summaries found."); sys.exit(0)

    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(args.outdir, "ensemble_adm_summary.csv"), index=False)

    med_rel = float(S["rel_res"].median())
    print(f"Runs: {len(S)} | median rel_res = {med_rel:.6f}")

    with PdfPages(os.path.join(args.outdir, "ensemble_adm_summary.pdf")) as pp:
        fig, ax = plt.subplots(figsize=(6.2,3.6))
        ax.plot(range(1, len(S)+1), S["rel_res"].values, "o-")
        ax.axhline(med_rel, ls="--", color="gray", label=f"median {med_rel:.3f}")
        ax.set_xlabel("run index")
        ax.set_ylabel("rel_res")
        ax.set_title("Per-run ADM (t0→t2) with β(x,y) patches")
        ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout(); pp.savefig(fig); plt.close(fig)

    print("Saved:", os.path.join(args.outdir, "ensemble_adm_summary.csv"),
                  os.path.join(args.outdir, "ensemble_adm_summary.pdf"))

if __name__ == "__main__":
    main()