#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scan for new local minima just below the boson arc (dark-sector candidates).

Usage (CMD, one line):
  python "...\dark_band_scan.py" ^
    --latent "...\latent_z_merged2.csv" ^
    --known  "...\all_particles_locked.csv" ^
    --ax_min 2.46 --ax_max 2.58 --ax_steps 401 ^
    --ay_min 0.860 --ay_max 0.895 --ay_steps 401 ^
    --min_sep 0.006 ^
    --outdir "...\Predicted Masses\dark_band_scan"

Inputs
------
latent: CSV with columns at least [ax, ay, z, r] (your standard latent grid merges)
known : CSV of all locked particles with columns [species, ax, ay] (we only read ax,ay)

Outputs
-------
- candidates.csv  : list of candidate minima with metrics
- heatmap_z.png   : z(ax,ay) heatmap with known points and candidates
- heatmap_r.png   : r(ax,ay) heatmap with known points and candidates
"""

import os, argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fit_quad(ax, ay, val):
    """Fit quadratic val(ax,ay) = c0 + c1*ax + c2*ay + c3*ax^2 + c4*ay^2 + c5*ax*ay"""
    X = np.column_stack([
        np.ones_like(ax),
        ax, ay,
        ax*ax, ay*ay, ax*ay
    ])
    coef, *_ = np.linalg.lstsq(X, val, rcond=None)
    return coef

def eval_quad(coef, AX, AY):
    X = np.stack([
        np.ones_like(AX),
        AX, AY,
        AX*AX, AY*AY, AX*AY
    ], axis=-1)
    return np.tensordot(X, coef, axes=([-1],[0]))

def local_minima_mask(Z):
    """8-neighbourhood interior minima mask (True at strict minima)."""
    m, n = Z.shape
    M = np.zeros_like(Z, dtype=bool)
    for i in range(1, m-1):
        for j in range(1, n-1):
            z = Z[i,j]
            nb = Z[i-1:i+2, j-1:j+2].copy()
            nb[1,1] = np.inf
            if z < nb.min():
                M[i,j] = True
    return M

def approx_laplacian(Z, dx, dy):
    """Second-derivative sum (discrete Laplacian) for curvature strength."""
    d2x = (np.roll(Z,-1,axis=1) - 2*Z + np.roll(Z,1,axis=1)) / (dx*dx)
    d2y = (np.roll(Z,-1,axis=0) - 2*Z + np.roll(Z,1,axis=0)) / (dy*dy)
    return d2x + d2y

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent", required=True)
    p.add_argument("--known",  required=True, help="CSV with at least ax,ay columns (species optional)")
    p.add_argument("--ax_min", type=float, default=2.46)
    p.add_argument("--ax_max", type=float, default=2.58)
    p.add_argument("--ay_min", type=float, default=0.860)
    p.add_argument("--ay_max", type=float, default=0.895)
    p.add_argument("--ax_steps", type=int, default=401)
    p.add_argument("--ay_steps", type=int, default=401)
    p.add_argument("--min_sep", type=float, default=0.006, help="exclude anything closer than this to known points")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load latent samples; sanitize
    L = pd.read_csv(args.latent)
    for col in ["ax","ay","z","r"]:
        if col not in L.columns:
            raise ValueError(f"latent CSV missing column: {col}")
    L = L.dropna(subset=["ax","ay","z","r"]).copy()

    # Fit z(ax,ay) and r(ax,ay)
    coef_z = fit_quad(L["ax"].values, L["ay"].values, L["z"].values)
    coef_r = fit_quad(L["ax"].values, L["ay"].values, L["r"].values)

    # Grid
    ax_vals = np.linspace(args.ax_min, args.ax_max, args.ax_steps)
    ay_vals = np.linspace(args.ay_min, args.ay_max, args.ay_steps)
    AX, AY = np.meshgrid(ax_vals, ay_vals)
    Z = eval_quad(coef_z, AX, AY)
    R = eval_quad(coef_r, AX, AY)

    # Interior minima
    mins_mask = local_minima_mask(Z)
    # Curvature measure
    if args.ax_steps > 2 and args.ay_steps > 2:
        lap = approx_laplacian(Z, ax_vals[1]-ax_vals[0], ay_vals[1]-ay_vals[0])
    else:
        lap = np.zeros_like(Z)

    # Known points (to exclude a neighbourhood)
    K = pd.read_csv(args.known)
    # Pick columns robustly
    if "ax" not in K.columns or "ay" not in K.columns:
        raise ValueError("known CSV must have columns 'ax' and 'ay'")
    known_xy = K[["ax","ay"]].dropna().values

    def min_dist(ax, ay):
        if len(known_xy) == 0:
            return np.inf
        d = np.sqrt((known_xy[:,0]-ax)**2 + (known_xy[:,1]-ay)**2)
        return float(d.min())

    # Collect candidates
    cand = []
    for (ii, jj) in zip(*np.where(mins_mask)):
        ax = float(AX[ii,jj]); ay = float(AY[ii,jj])
        z  = float(Z[ii,jj]);   r  = float(R[ii,jj])
        sep = min_dist(ax, ay)
        if sep < args.min_sep:
            continue
        # stronger minima → more negative Laplacian
        curv = float(lap[ii,jj])
        cand.append(dict(ax=ax, ay=ay, z_pred=z, r_pred=r, sep_to_known=sep, laplacian=curv))

    C = pd.DataFrame(cand)
    if not C.empty:
        # Prefer deeper basins and decent curvature (more negative laplacian)
        C["rank_score"] = (-C["z_pred"].abs()) + 0.002*(-C["laplacian"])
        C = C.sort_values(by=["rank_score"], ascending=False)
    C.to_csv(os.path.join(args.outdir, "candidates.csv"), index=False)

    # Plots
    def plot_heat(X, Y, V, title, fname):
        plt.figure(figsize=(8,6))
        im = plt.pcolormesh(X, Y, V, shading="auto")
        plt.colorbar(im, label=title.split()[0])
        # Known points
        if len(known_xy):
            plt.scatter(known_xy[:,0], known_xy[:,1], s=18, facecolors="none", edgecolors="k", label="known")
        # Candidates
        if not C.empty:
            plt.scatter(C["ax"], C["ay"], s=24, marker="x", label="candidates")
        plt.xlabel("ax"); plt.ylabel("ay")
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, fname), dpi=150)
        plt.close()

    plot_heat(AX, AY, Z, "z(ax,ay) heatmap (dark-band scan)", "heatmap_z.png")
    plot_heat(AX, AY, R, "r(ax,ay) heatmap (dark-band scan)", "heatmap_r.png")

    if C.empty:
        print("[INFO] No interior minima passed the separation cut in this box.")
    else:
        top = C.head(10)
        print("\nTop candidates:")
        for i, row in enumerate(top.itertuples(index=False), start=1):
            print(f" #{i:>2}: ax={row.ax:.6f}  ay={row.ay:.6f}  z={row.z_pred:.6f}  "
                  f"r={row.r_pred:.6f}  sep={row.sep_to_known:.4f}  lap={row.laplacian:.2f}")
        print(f"\nWrote: {os.path.join(args.outdir,'candidates.csv')}")
        print(f"      {os.path.join(args.outdir,'heatmap_z.png')}")
        print(f"      {os.path.join(args.outdir,'heatmap_r.png')}")
    print("Done.")

if __name__ == "__main__":
    main()