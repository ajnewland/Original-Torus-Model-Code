#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
u1_band_extract.py

Create a U(1)-style constant-z band from sweep CSVs.

Usage (example with your files):
  python "C:\\...\\u1_band_extract.py" ^
    "C:\\...\\u1_prep.csv" ^
    --z-target 0.231 --z-tol 0.0015 ^
    --out "C:\\...\\u1_band.csv"

Column names are flexible; defaults match typical EW/U1 sweeps:
  --ax-name ax --ay-name ay --z-name z --r-name r

If your z column is S_star, add: --z-name S_star
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Extract a constant-z band from sweep CSV(s).")
    p.add_argument("inputs", nargs="+", help="Input CSV file(s)")
    p.add_argument("--out", required=True, help="Output CSV path for the band")
    p.add_argument("--ax-name", default="ax", help="Column name for ax (default: ax)")
    p.add_argument("--ay-name", default="ay", help="Column name for ay (default: ay)")
    p.add_argument("--z-name", default="z", help="Column name for z/value (default: z)")
    p.add_argument("--r-name", default=None, help="Optional column name for r (default: None)")
    p.add_argument("--z-target", type=float, required=True, help="Target z value")
    p.add_argument("--z-tol", type=float, required=True, help="Tolerance around z target (abs(z-zt) <= tol)")
    p.add_argument("--resid-ptile", type=float, default=None,
                   help="Optional: keep rows below this percentile of |z - z_target| within the band")
    p.add_argument("--erode", type=int, default=0,
                   help="Erode the band’s edge this many iterations (0 = no erosion).")
    p.add_argument("--min-internal", type=int, default=0,
                   help="Require at least this many cells after erosion (0 disables).")
    return p.parse_args()


def read_concat(inputs):
    frames = []
    for f in inputs:
        try:
            df = pd.read_csv(f)
            df["__source__"] = str(f)
            frames.append(df)
        except Exception as e:
            print(f"[warn] could not read {f}: {e}", file=sys.stderr)
    if not frames:
        print("[error] no readable inputs", file=sys.stderr)
        sys.exit(2)
    return pd.concat(frames, ignore_index=True)


def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[error] missing column(s): {missing}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(2)


def normalize01(x):
    x = x.astype(float)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax == xmin:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


def grid_keys(axn, ayn, bins=1024):
    """
    Stable, vectorized grid key without Series bit-shift.
    """
    i = np.floor(axn * bins).astype(np.int64)
    j = np.floor(ayn * bins).astype(np.int64)
    # clamp to valid range
    i = np.clip(i, 0, bins - 1)
    j = np.clip(j, 0, bins - 1)
    return i * bins + j, i, j


def erode_keys(keys, i, j, bins=1024, iterations=1):
    """
    Morphological-like erosion: keep cells whose 4-neighbors exist.
    """
    if iterations <= 0:
        return keys
    key_set = set(keys.tolist())
    kept = np.ones_like(keys, dtype=bool)

    for _ in range(iterations):
        # recompute each round on currently kept keys
        current = {k for k, keep in zip(keys, kept) if keep}
        if not current:
            return np.array([], dtype=np.int64)

        # neighbor predicates
        north = (i - 1) >= 0
        south = (i + 1) < bins
        west = (j - 1) >= 0
        east = (j + 1) < bins

        kN = (i - 1) * bins + j
        kS = (i + 1) * bins + j
        kW = i * bins + (j - 1)
        kE = i * bins + (j + 1)

        keep_now = (
            north & south & west & east &
            np.fromiter((k in current for k in kN), dtype=bool) &
            np.fromiter((k in current for k in kS), dtype=bool) &
            np.fromiter((k in current for k in kW), dtype=bool) &
            np.fromiter((k in current for k in kE), dtype=bool)
        )
        kept = kept & keep_now

    return keys[kept]


def main():
    args = parse_args()

    df = read_concat(args.inputs)

    # Column checks / mapping
    require_columns(df, [args.ax_name, args.ay_name, args.z_name])
    ax = df[args.ax_name].astype(float)
    ay = df[args.ay_name].astype(float)
    z = df[args.z_name].astype(float)

    r = None
    if args.r_name is not None:
        if args.r_name in df.columns:
            r = df[args.r_name].astype(float)
        else:
            print(f"[warn] r column '{args.r_name}' not found; proceeding without r.", file=sys.stderr)

    base = pd.DataFrame({"ax": ax, "ay": ay, "z": z})
    if r is not None:
        base["r"] = r

    # Filter finite
    base = base[np.isfinite(base["ax"]) & np.isfinite(base["ay"]) & np.isfinite(base["z"])]
    if "r" in base.columns:
        base = base[np.isfinite(base["r"])]

    if base.empty:
        print("[error] no finite rows after cleaning.", file=sys.stderr)
        sys.exit(2)

    # Band by target ± tol
    dz = np.abs(base["z"] - args.z_target)
    band = base[dz <= args.z_tol].copy()

    if band.empty:
        print("[error] no rows within z-band. Adjust --z-target/--z-tol.", file=sys.stderr)
        sys.exit(2)

    # Optional percentile tightening inside the band
    if args.resid_ptile is not None:
        if not (0 < args.resid_ptile <= 100):
            print("[warn] --resid-ptile should be in (0, 100]; ignoring.", file=sys.stderr)
        else:
            thr = np.percentile(np.abs(band["z"] - args.z_target), args.resid_ptile)
            band = band[np.abs(band["z"] - args.z_target) <= thr]
            if band.empty:
                print("[error] percentile filter removed all rows; relax --resid-ptile.", file=sys.stderr)
                sys.exit(2)

    # Interior (optional erosion)
    axn = normalize01(band["ax"].to_numpy())
    ayn = normalize01(band["ay"].to_numpy())
    keys, i, j = grid_keys(axn, ayn, bins=1024)

    if args.erode > 0:
        kept_keys = set(erode_keys(keys, i, j, bins=1024, iterations=args.erode).tolist())
        if kept_keys:
            band = band[[k in kept_keys for k in keys]]
        else:
            print("[warn] erosion removed all points; reverting to non-eroded band.", file=sys.stderr)

    # Optional size guard
    if args.min_internal and len(band) < args.min_internal:
        print(f"[warn] band size {len(band)} < --min-internal {args.min_internal}.", file=sys.stderr)

    # Residual column for convenience
    band["z_resid"] = band["z"] - args.z_target

    # Summary (console)
    summary = {
        "counts": {
            "input_rows": int(len(df)),
            "clean_rows": int(len(base)),
            "band_rows": int(len(band)),
        },
        "z_target": args.z_target,
        "z_tol": args.z_tol,
        "z_stats": {
            "mean_all": float(base["z"].mean()),
            "std_all": float(base["z"].std(ddof=0)),
            "mean_band": float(band["z"].mean()),
            "std_band": float(band["z"].std(ddof=0)),
        },
    }
    if "r" in base.columns:
        summary["r_stats"] = {
            "mean_all": float(base["r"].mean()),
            "mean_band": float(band["r"].mean()) if not band.empty else float("nan"),
        }

    print("=== U1 band extract ===")
    print(f"Inputs: {len(args.inputs)} file(s)")
    print(f"Output: {args.out}")
    print(f"Rows (input / clean / band): {summary['counts']['input_rows']} / "
          f"{summary['counts']['clean_rows']} / {summary['counts']['band_rows']}")
    print(f"z target ± tol: {args.z_target} ± {args.z_tol}")
    print("z stats (all/band): "
          f"mean={summary['z_stats']['mean_all']:.9f}/{summary['z_stats']['mean_band']:.9f}, "
          f"std={summary['z_stats']['std_all']:.9f}/{summary['z_stats']['std_band']:.9f}")
    if "r_stats" in summary:
        print(f"r mean (all/band): {summary['r_stats']['mean_all']:.9f}/{summary['r_stats']['mean_band']:.9f}")

    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["ax", "ay", "z", "z_resid"] + (["r"] if "r" in band.columns else [])
    # keep canonical order ax, ay, z, r?, z_resid (match earlier convention better)
    ordered = ["ax", "ay", "z"] + (["r"] if "r" in band.columns else []) + ["z_resid"]
    band[ordered].to_csv(out_path, index=False)

    # Sidecar JSON (nice to have)
    sidecar = out_path.with_suffix(".json")
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote band CSV: {out_path}")
    print(f"[ok] wrote summary JSON: {sidecar}")


if __name__ == "__main__":
    main()