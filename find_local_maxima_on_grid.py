# find_local_maxima_on_grid.py
# Usage:
#   python find_local_maxima_on_grid.py ^
#     --grid "...\grid_ax_ay_z.csv" ^
#     --known "...\all_particles_locked.csv" ^
#     --min_sep 0.006 ^
#     --outdir "...\dark_band_candidates"
#
# Outputs:
#   peaks.csv  - list of local maxima (ax, ay, z_pred, sep_min)
#   peaks_marked.png - heatmap with peaks marked (for a quick look)

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_grid(grid_csv):
    df = pd.read_csv(grid_csv)
    # Expect columns: ax, ay, z_pred (like your refined sweeps)
    needed = {"ax","ay","z_pred"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Grid needs columns {needed}, found {df.columns.tolist()}")
    # Build sorted unique axes and reshape z into 2D
    ax_vals = np.sort(df["ax"].unique())
    ay_vals = np.sort(df["ay"].unique())
    nx, ny = len(ax_vals), len(ay_vals)
    pivot = df.pivot(index="ay", columns="ax", values="z_pred").sort_index().sort_index(axis=1)
    Z = pivot.values  # shape (ny, nx), rows=ay, cols=ax
    # sanity
    if Z.shape != (ny, nx):
        raise RuntimeError("Grid reshape failed.")
    return ax_vals, ay_vals, Z

def load_known(known_csv):
    dk = pd.read_csv(known_csv)
    # must have ax, ay
    if not {"ax","ay"}.issubset(dk.columns):
        raise ValueError("Known CSV must have columns ax, ay")
    return dk[["ax","ay"]].values

def find_local_maxima(Z):
    ny, nx = Z.shape
    peaks = []
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            z = Z[j,i]
            nbrs = Z[j-1:j+2, i-1:i+2].copy()
            nbrs[1,1] = -np.inf
            if z > np.max(nbrs):
                peaks.append((i,j,z))
    return peaks

def min_sep_from_known(ax_vals, ay_vals, peaks, known_xy):
    out = []
    for (i,j,z) in peaks:
        ax = ax_vals[i]; ay = ay_vals[j]
        if known_xy is None or len(known_xy)==0:
            sep = np.inf
        else:
            d = np.sqrt((known_xy[:,0]-ax)**2 + (known_xy[:,1]-ay)**2)
            sep = float(np.min(d))
        out.append((ax, ay, z, sep))
    return out

def save_peaks(ax_vals, ay_vals, Z, peaks_xy, out_csv, out_png):
    dfp = pd.DataFrame(peaks_xy, columns=["ax","ay","z_pred","sep_min"])
    dfp = dfp.sort_values("z_pred", ascending=False).reset_index(drop=True)
    dfp.to_csv(out_csv, index=False)

    # Plot heatmap + peaks
    plt.figure(figsize=(7,6))
    extent = [ax_vals.min(), ax_vals.max(), ay_vals.min(), ay_vals.max()]
    plt.imshow(Z, origin="lower", aspect="auto", extent=extent)
    if len(peaks_xy):
        px = [p[0] for p in peaks_xy]
        py = [p[1] for p in peaks_xy]
        plt.scatter(px, py, s=20, marker="x")
    plt.xlabel("ax"); plt.ylabel("ay"); plt.title("z heatmap with local maxima")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--known", required=False, default=None)
    ap.add_argument("--min_sep", type=float, default=0.006)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ax_vals, ay_vals, Z = load_grid(args.grid)
    peaks = find_local_maxima(Z)

    known_xy = None
    if args.known and os.path.exists(args.known):
        known_xy = load_known(args.known)

    peaks_with_sep = min_sep_from_known(ax_vals, ay_vals, peaks, known_xy)
    # apply separation cut
    peaks_with_sep = [p for p in peaks_with_sep if p[3] >= args.min_sep]

    out_csv = os.path.join(args.outdir, "peaks.csv")
    out_png = os.path.join(args.outdir, "peaks_marked.png")
    save_peaks(ax_vals, ay_vals, Z, peaks_with_sep, out_csv, out_png)

    print(f"[DONE] peaks -> {out_csv}")
    print(f"[DONE] figure -> {out_png}")
    if not peaks_with_sep:
        print("[INFO] No peaks survived sep cut. Try relaxing --min_sep a bit.")

if __name__ == "__main__":
    main()