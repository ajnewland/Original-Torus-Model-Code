# -*- coding: utf-8 -*-
"""
Coordinate-invariant derivation of sin^2(theta_W) from principal curvatures
on the boson/fermion interface.

Inputs per directory:
  - grid_ax_ay_z.csv   (columns: ax, ay, z  OR  ax, ay, z_pred)
  - matches.csv        (columns: ax, ay)     best-match coordinate

We:
  1) reconstruct a regular grid: Z(ay, ax)
  2) finite-difference to get z_x, z_y, z_xx, z_xy, z_yy using physical spacings
  3) build First form I and Second form II, then S = I^{-1} II
  4) principal curvatures = eigenvalues of S. Define impedances |k_i|.
  5) s* = |k2| / (|k1| + |k2|), averaged over a small neighborhood.

Usage (CMD):
  python "...\derive_sin2_from_impedance_v2.py" ^
    --dirs ^
    "...\boson_refined_W" ^
    "...\boson_refined_Z" ^
    "...\boson_refined_H" ^
    --halfwin 2 ^
    --out "...\boson_impedance_summary_v2.csv"
"""

import argparse, os, sys, math
import numpy as np
import pandas as pd

def _regularize_grid(df, colz):
    xs = np.unique(df["ax"].values.astype(float))
    ys = np.unique(df["ay"].values.astype(float))
    nx, ny = len(xs), len(ys)
    Z = np.full((ny, nx), np.nan, dtype=float)  # rows: y, cols: x
    for _, r in df.iterrows():
        ix = np.searchsorted(xs, float(r["ax"]))
        iy = np.searchsorted(ys, float(r["ay"]))
        if 0 <= ix < nx and 0 <= iy < ny:
            Z[iy, ix] = float(r[colz])
    if np.isnan(Z).any():
        raise RuntimeError("Grid has NaNs; ensure refined grid is complete.")
    return xs, ys, Z

def _finite_differences(xs, ys, Z):
    # gradients with *physical* spacing vectors (Matplotlib/NumPy convention: (rows, cols) ~ (y, x))
    dZ_dy, dZ_dx = np.gradient(Z, ys, xs)            # zy, zx
    d2Z_dyy, _    = np.gradient(dZ_dy, ys, xs)       # zyy
    _       , d2Z_dxx = np.gradient(dZ_dx, ys, xs)   # zxx
    # mixed second derivative: average of two routes for symmetry
    dZdy_dx, _ = np.gradient(dZ_dy, ys, xs)  # (first dy then dx)
    _, dZdx_dy = np.gradient(dZ_dx, ys, xs)  # (first dx then dy)
    d2Z_dxy = 0.5*(dZdy_dx + dZdx_dy)
    return dZ_dx, dZ_dy, d2Z_dxx, d2Z_dxy, d2Z_dyy

def _index_of(xs, x0):
    return int(np.argmin(np.abs(xs - x0)))

def _curvatures_at(i, j, zx, zy, zxx, zxy, zyy):
    # First fundamental form I
    E = 1.0 + zx[j,i]**2
    F = zx[j,i]*zy[j,i]
    G = 1.0 + zy[j,i]**2
    I = np.array([[E, F],[F, G]], dtype=float)

    # Unit normal denominator
    denom = math.sqrt(1.0 + zx[j,i]**2 + zy[j,i]**2)
    # Second fundamental form II
    e = zxx[j,i]/denom
    f = zxy[j,i]/denom
    g = zyy[j,i]/denom
    II = np.array([[e, f],[f, g]], dtype=float)

    # Shape operator: S = I^{-1} II
    try:
        S = np.linalg.solve(I, II)  # I^{-1} II
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    evals = np.linalg.eigvals(S)
    k1, k2 = float(np.real(evals[0])), float(np.real(evals[1]))
    return k1, k2

def analyze_dir(d, halfwin):
    grid = os.path.join(d, "grid_ax_ay_z.csv")
    match = os.path.join(d, "matches.csv")
    if not (os.path.isfile(grid) and os.path.isfile(match)):
        return None, f"[WARN] Missing files in {d}"

    g = pd.read_csv(grid)
    m = pd.read_csv(match)
    colz = "z" if "z" in g.columns else ("z_pred" if "z_pred" in g.columns else None)
    if colz is None:
        return None, f"[ERROR] {grid}: need 'z' or 'z_pred' column."
    if not {"ax","ay"}.issubset(g.columns) or not {"ax","ay"}.issubset(m.columns):
        return None, f"[ERROR] ax,ay columns missing."

    xs, ys, Z = _regularize_grid(g[["ax","ay",colz]].rename(columns={colz:"z"}), "z")
    zx, zy, zxx, zxy, zyy = _finite_differences(xs, ys, Z)

    ax0, ay0 = float(m.iloc[0]["ax"]), float(m.iloc[0]["ay"])
    ix0, iy0 = _index_of(xs, ax0), _index_of(ys, ay0)

    s_vals = []
    for j in range(max(0, iy0-halfwin), min(len(ys), iy0+halfwin+1)):
        for i in range(max(0, ix0-halfwin), min(len(xs), ix0+halfwin+1)):
            k1, k2 = _curvatures_at(i, j, zx, zy, zxx, zxy, zyy)
            if not (np.isfinite(k1) and np.isfinite(k2)):
                continue
            a1, a2 = abs(k1), abs(k2)
            denom = a1 + a2
            if denom <= 0:
                continue
            s = a2 / denom  # label-free ratio in [0,1]
            s_vals.append(s)

    s_vals = np.array(s_vals, dtype=float)
    s_vals = s_vals[np.isfinite(s_vals)]
    if s_vals.size == 0:
        return None, f"[WARN] No finite s* near ({ax0},{ay0}) in {d}"

    med = float(np.median(s_vals))
    mad = float(np.median(np.abs(s_vals - med)))
    return {
        "dir": d,
        "ax0": ax0, "ay0": ay0,
        "n_pts": int(s_vals.size),
        "s_med": med, "s_MAD": mad
    }, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="Boson refined directories (W,Z,H)")
    ap.add_argument("--halfwin", type=int, default=2, help="Neighborhood radius in grid indices (default 2)")
    ap.add_argument("--out", default=None, help="CSV summary output")
    args = ap.parse_args()

    rows, msgs = [], []
    for d in args.dirs:
        rec, msg = analyze_dir(d, args.halfwin)
        if rec: rows.append(rec)
        if msg: msgs.append(msg)

    if msgs:
        for s in msgs: print(s)
    if not rows:
        print("[ERROR] No usable directories.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    print("=== sin^2(theta_W) via principal-curvature ratio (coordinate-invariant) ===")
    for _, r in df.iterrows():
        print("dir=%s  (ax0=%.6f, ay0=%.6f)  N=%d  s_med=%.6f  s_MAD=%.6f" %
              (r["dir"], r["ax0"], r["ay0"], r["n_pts"], r["s_med"], r["s_MAD"]))

    s_all = df["s_med"].values
    S_med = float(np.median(s_all))
    S_mad = float(np.median(np.abs(s_all - S_med)))
    print("\nAggregate: s_med=%.6f  s_MAD=%.6f  (over %d bosons)" % (S_med, S_mad, len(s_all)))

    if args.out:
        df.to_csv(args.out, index=False)
        print("[DONE] Wrote summary ->", args.out)

if __name__ == "__main__":
    main()