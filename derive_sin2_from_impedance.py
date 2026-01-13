# -*- coding: utf-8 -*-
"""
Derive sin^2(theta_W) from local 'impedance' ratio on the boson/fermion interface.

Inputs (per boson dir):
  - grid_ax_ay_z.csv   (columns: ax, ay, z  OR  ax, ay, z_pred)
  - matches.csv        (columns: ax, ay) best-match coordinate

For each dir, we:
  1) reconstruct a regular grid and finite-difference it to get z_x, z_y, z_xx, z_xy, z_yy
  2) build the Hessian H = [[z_xx, z_xy],[z_xy, z_yy]] (shape operator up to a factor)
  3) get principal curvatures (eigs of H) and their eigenvectors
  4) define metric surrogates along those eigendirections:
       I_x ∝ sqrt(1 + (z_x along e1)^2) * |k1|
       I_y ∝ sqrt(1 + (z_y along e2)^2) * |k2|
     and the impedance ratio s* = I_y / (I_x + I_y)
  5) evaluate at the boson match (ax0,ay0) and average over a small neighborhood.

Usage example (CMD):
  python "...\derive_sin2_from_impedance.py" ^
    --dirs ^
    "...\boson_refined_W" ^
    "...\boson_refined_Z" ^
    "...\boson_refined_H" ^
    --halfwin 2 ^
    --out "...\boson_impedance_summary.csv"

Notes:
- We never insert 0.231 anywhere. This is a genuine derivation check.
- 'halfwin' sets the neighborhood radius (in grid indices) used for local averaging.
"""

import argparse, os, sys, math
import numpy as np
import pandas as pd

def _regularize_grid(df, colz):
    # unique sorted coords
    xs = np.unique(df["ax"].values.astype(float))
    ys = np.unique(df["ay"].values.astype(float))
    nx, ny = len(xs), len(ys)

    # build Z as ny x nx with ay as rows, ax as cols
    Z = np.full((ny, nx), np.nan, dtype=float)
    for _, r in df.iterrows():
        ix = np.searchsorted(xs, float(r["ax"]))
        iy = np.searchsorted(ys, float(r["ay"]))
        if 0 <= ix < nx and 0 <= iy < ny:
            Z[iy, ix] = float(r[colz])
    if np.isnan(Z).any():
        # try to fill small holes by nearest
        # (dense refined sweeps shouldn't need this, but keep robust)
        from scipy.ndimage import generic_filter
        def _nan_to_mean(win):
            w = win[~np.isnan(win)]
            return np.mean(w) if w.size else np.nan
        Z = generic_filter(Z, _nan_to_mean, size=3, mode='nearest')
    if np.isnan(Z).any():
        raise RuntimeError("Grid has NaNs; ensure the refined grid is complete.")
    return xs, ys, Z

def _finite_differences(xs, ys, Z):
    dy, dx = np.gradient(Z, ys, xs)            # zy, zx
    dyy, dyx = np.gradient(dy, ys, xs)         # zyy, zyx
    dxy, dxx = np.gradient(dx, ys, xs)         # zxy, zxx
    # Align shapes; np.gradient on (rows,cols) order: careful:
    # After checks: dy ~ dZ/dy, dx ~ dZ/dx; dxx, dxy, dyx, dyy ok.
    return dx, dy, dxx, dxy, dyy

def _index_of(xs, x0):
    return int(np.argmin(np.abs(xs - x0)))

def _unit(v):
    n = np.linalg.norm(v)
    return v/n if n>0 else v

def _impedance_at(ix, iy, dx, dy, dxx, dxy, dyy):
    # Hessian at (iy,ix)
    H = np.array([[dxx[iy,ix], dxy[iy,ix]],
                  [dxy[iy,ix], dyy[iy,ix]]], dtype=float)
    evals, evecs = np.linalg.eig(H)
    # principal curvatures and directions in (x,y) chart
    # sort by |k| descending so k1 has larger magnitude
    order = np.argsort(-np.abs(evals))
    k1, k2 = float(evals[order[0]]), float(evals[order[1]])
    e1 = _unit(evecs[:,order[0]])
    e2 = _unit(evecs[:,order[1]])

    # project gradient onto those directions to get directional slopes
    gx = dx[iy,ix]; gy = dy[iy,ix]
    grad = np.array([gx, gy], dtype=float)
    slope1 = float(np.dot(grad, e1))
    slope2 = float(np.dot(grad, e2))

    # metric surrogates along e1/e2: sqrt(1 + slope^2)
    I1 = math.sqrt(1.0 + slope1*slope1) * abs(k1)
    I2 = math.sqrt(1.0 + slope2*slope2) * abs(k2)

    # define "x" as the one aligned more with +ax (|e·ex| larger), purely for naming
    ex = np.array([1.0, 0.0])
    if abs(np.dot(e1, ex)) >= abs(np.dot(e2, ex)):
        Ix, Iy = I1, I2
        kx, ky = k1, k2
    else:
        Ix, Iy = I2, I1
        kx, ky = k2, k1

    s = Iy / (Ix + Iy) if (Ix + Iy) > 0 else float('nan')
    return s, kx, ky, Ix, Iy

def analyze_dir(d, halfwin):
    grid = os.path.join(d, "grid_ax_ay_z.csv")
    match = os.path.join(d, "matches.csv")
    if not (os.path.isfile(grid) and os.path.isfile(match)):
        return None, f"[WARN] Missing files in {d}"

    g = pd.read_csv(grid)
    m = pd.read_csv(match)
    if "z" in g.columns: colz = "z"
    elif "z_pred" in g.columns: colz = "z_pred"
    else: return None, f"[ERROR] {grid}: need 'z' or 'z_pred' column."

    if not {"ax","ay"}.issubset(g.columns) or not {"ax","ay"}.issubset(m.columns):
        return None, f"[ERROR] ax,ay columns missing."

    xs, ys, Z = _regularize_grid(g[["ax","ay",colz]].rename(columns={colz:"z"}), "z")
    dx, dy, dxx, dxy, dyy = _finite_differences(xs, ys, Z)

    ax0, ay0 = float(m.iloc[0]["ax"]), float(m.iloc[0]["ay"])
    ix0, iy0 = _index_of(xs, ax0), _index_of(ys, ay0)

    # Gather neighborhood values
    s_vals = []
    for j in range(max(0, iy0-halfwin), min(len(ys), iy0+halfwin+1)):
        for i in range(max(0, ix0-halfwin), min(len(xs), ix0+halfwin+1)):
            s, kx, ky, Ix, Iy = _impedance_at(i, j, dx, dy, dxx, dxy, dyy)
            s_vals.append(s)
    s_vals = np.array([v for v in s_vals if not np.isnan(v)])
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
    print("=== sin^2(theta_W) via impedance ratio (metric × curvature) ===")
    for _, r in df.iterrows():
        print("dir=%s  (ax0=%.6f, ay0=%.6f)  N=%d  s_med=%.6f  s_MAD=%.6f" %
              (r["dir"], r["ax0"], r["ay0"], r["n_pts"], r["s_med"], r["s_MAD"]))

    # Aggregate across bosons (robust)
    s_all = df["s_med"].values
    S_med = float(np.median(s_all))
    S_mad = float(np.median(np.abs(s_all - S_med)))
    print("\nAggregate: s_med=%.6f  s_MAD=%.6f  (over %d bosons)" % (S_med, S_mad, len(s_all)))

    if args.out:
        df.to_csv(args.out, index=False)
        print("[DONE] Wrote summary ->", args.out)

if __name__ == "__main__":
    main()