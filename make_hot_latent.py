#!/usr/bin/env python3
"""
make_hot_latent.py
-------------------
Utility to generate "hotter" / earlier-epoch variants of your latent surface.

Inputs:
  --latent   input latent CSV (with at least columns ax, ay, z_pred)
  --out      output CSV path
Options:
  --scale    global scaling factor for z_pred (default 1.0 = no change)
  --tilt_ax  linear tilt factor along ax direction (default 0.0)
  --tilt_ay  linear tilt factor along ay direction (default 0.0)
  --jitter_std  add Gaussian noise with this std.dev (default 0.0)
  --seed     random seed (default 1234)
"""

import argparse
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True, help="input latent CSV")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--scale", type=float, default=1.0, help="global scale for z_pred")
    ap.add_argument("--tilt_ax", type=float, default=0.0, help="linear tilt along ax")
    ap.add_argument("--tilt_ay", type=float, default=0.0, help="linear tilt along ay")
    ap.add_argument("--jitter_std", type=float, default=0.0, help="Gaussian noise std")
    ap.add_argument("--seed", type=int, default=1234, help="random seed")
    args = ap.parse_args()

    # Load input latent surface
    df = pd.read_csv(args.latent)

    if not {"ax", "ay", "z_pred"}.issubset(df.columns):
        raise ValueError("Input CSV must contain columns: ax, ay, z_pred")

    np.random.seed(args.seed)

    # Apply deformation
    df["z_hot"] = (
        args.scale * df["z_pred"].values
        + args.tilt_ax * (df["ax"].values - df["ax"].mean())
        + args.tilt_ay * (df["ay"].values - df["ay"].mean())
    )

    if args.jitter_std > 0.0:
        df["z_hot"] += np.random.normal(0.0, args.jitter_std, size=len(df))

    # Keep same structure, rename z column
    df_out = df.copy()
    df_out = df_out.rename(columns={"z_pred": "z_orig"})
    df_out["z_pred"] = df["z_hot"]

    df_out.to_csv(args.out, index=False)
    print(f"[WROTE] {args.out}")
    print(f"scale={args.scale}, tilt_ax={args.tilt_ax}, tilt_ay={args.tilt_ay}, jitter_std={args.jitter_std}")

if __name__ == "__main__":
    main()