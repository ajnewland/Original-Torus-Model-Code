#!/usr/bin/env python3
"""
make_hot_latent_v2.py
Create an "early-epoch / hot" deformation of a latent surface on (ax, ay).

Inputs:
  --latent  <csv with at least ax, ay, z-like column>
  --out     <output csv>

Optional:
  --ax_col, --ay_col, --z_col  (defaults: try to auto-detect)
  --scale <float>      global scale for z (default 1.0)
  --tilt_ax <float>    linear tilt along ax (default 0.0)
  --tilt_ay <float>    linear tilt along ay (default 0.0)
  --jitter_std <float> add Gaussian noise (default 0.0)
  --seed <int>         RNG seed (default 1234)

We write all original columns +:
  z_orig : the source z-like column
  z_pred : the deformed ("hot") z
"""
import argparse
import pandas as pd
import numpy as np
import sys

CAND_Z_COLS = ["z_pred","z","Z","value","S_star","S_raw","lat_z"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ax_col", default=None)
    ap.add_argument("--ay_col", default=None)
    ap.add_argument("--z_col",  default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--tilt_ax", type=float, default=0.0)
    ap.add_argument("--tilt_ay", type=float, default=0.0)
    ap.add_argument("--jitter_std", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    df = pd.read_csv(args.latent)

    # Auto-detect columns
    ax_col = args.ax_col or ("ax" if "ax" in df.columns else None)
    ay_col = args.ay_col or ("ay" if "ay" in df.columns else None)

    z_col = args.z_col
    if z_col is None:
        for c in CAND_Z_COLS:
            if c in df.columns:
                z_col = c
                break

    missing = []
    if ax_col is None: missing.append("ax")
    if ay_col is None: missing.append("ay")
    if z_col  is None: missing.append("z-like (one of %s)" % CAND_Z_COLS)

    if missing:
        raise ValueError(
            "Could not find required columns: %s\nFound columns: %s" %
            (", ".join(missing), list(df.columns))
        )

    # Build hot deformation
    np.random.seed(args.seed)
    ax = df[ax_col].astype(float).values
    ay = df[ay_col].astype(float).values
    z  = df[z_col].astype(float).values

    z_hot = (
        args.scale * z
        + args.tilt_ax * (ax - np.nanmean(ax))
        + args.tilt_ay * (ay - np.nanmean(ay))
    )
    if args.jitter_std > 0.0:
        z_hot += np.random.normal(0.0, args.jitter_std, size=len(z_hot))

    out = df.copy()
    out["z_orig"] = z
    out["z_pred"] = z_hot

    out.to_csv(args.out, index=False)
    print(f"[WROTE] {args.out}")
    print(f"Detected columns: ax_col={ax_col}, ay_col={ay_col}, z_col={z_col}")
    print(f"Params: scale={args.scale}, tilt_ax={args.tilt_ax}, tilt_ay={args.tilt_ay}, jitter_std={args.jitter_std}, seed={args.seed}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)