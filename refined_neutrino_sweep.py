# refined_neutrino_sweep.py
# 2-D mesh sweep in (ax, ay) to find z ~= target values (neutrino band).
# - Fits a quadratic z-surface to your latent points
# - Evaluates z on a grid in a chosen box
# - Finds best matches for a list of z-targets (keeps points separated)
# - Saves grid CSV, matches CSV, and a heatmap + contour plot
#
# Usage example (Windows, ASCII only in prints):
#   python refined_neutrino_sweep.py ^
#     --latent "C:\path\to\latent_z_merged2.csv" ^
#     --ax_min 2.55 --ax_max 2.58 --ax_steps 121 ^
#     --ay_min 0.72 --ay_max 0.75 --ay_steps 121 ^
#     --targets -1.4518 -1.45165 -1.4515 ^
#     --min_sep 0.002 ^
#     --outdir "C:\path\to\Predicted Masses\neutrino_refined_sweep"
#
# Notes:
# - Expects latent CSV with columns: ax, ay, z  (other columns OK, they are ignored)
# - If your latent file has 'r' instead of 'z', change ZCOL below to 'r'.

import argparse
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ZCOL = "z"   # change to "r" if your latent file uses r as the signed surface

def load_latent(path):
    df = pd.read_csv(path)
    # Be tolerant about column names/case/whitespace:
    df.columns = [c.strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}
    axc = cols.get("ax")
    ayc = cols.get("ay")
    zc  = cols.get(ZCOL) or cols.get(ZCOL.lower())
    if axc is None or ayc is None:
        raise ValueError("latent CSV must have columns 'ax' and 'ay'.")
    if zc is None:
        raise ValueError(f"latent CSV must include '{ZCOL}' column.")
    out = df[[axc, ayc, zc]].rename(columns={axc: "ax", ayc: "ay", zc: "z"})
    out = out.dropna(subset=["ax", "ay", "z"])
    return out

def fit_quad(df):
    # z = c0 + c1*ax + c2*ay + c3*ax^2 + c4*ax*ay + c5*ay^2
    A = np.column_stack([
        np.ones(len(df)),
        df["ax"].values,
        df["ay"].values,
        df["ax"].values**2,
        df["ax"].values*df["ay"].values,
        df["ay"].values**2,
    ])
    b = df["z"].values
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coef

def z_hat(ax, ay, coef):
    ax = np.asarray(ax)
    ay = np.asarray(ay)
    return (coef[0]
            + coef[1]*ax
            + coef[2]*ay
            + coef[3]*ax*ax
            + coef[4]*ax*ay
            + coef[5]*ay*ay)

def mk_grid(ax_min, ax_max, ax_steps, ay_min, ay_max, ay_steps):
    ax_lin = np.linspace(ax_min, ax_max, ax_steps)
    ay_lin = np.linspace(ay_min, ay_max, ay_steps)
    AX, AY = np.meshgrid(ax_lin, ay_lin, indexing="xy")
    return AX, AY, ax_lin, ay_lin

def find_separated_best(AX, AY, Z, targets, min_sep):
    """
    Greedy: for each target, pick the grid point minimizing |Z - target|,
    then mask out neighbors within Euclidean distance min_sep before the next target.
    """
    pts = []
    mask = np.ones_like(Z, dtype=bool)
    for t in targets:
        if not mask.any():
            pts.append((np.nan, np.nan, np.nan, np.nan))
            continue
        diff = np.abs(Z - t)
        diff_masked = np.where(mask, diff, np.inf)
        idx = np.unravel_index(np.argmin(diff_masked), diff_masked.shape)
        i, j = idx
        ax_best = AX[i, j]
        ay_best = AY[i, j]
        z_best  = Z[i, j]
        err     = float(abs(z_best - t))
        pts.append((ax_best, ay_best, z_best, err))
        # mask neighbors within min_sep
        d2 = (AX - ax_best)**2 + (AY - ay_best)**2
        mask &= (d2 >= (min_sep**2))
    return pts

def ensure_dir(d):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def main():
    p = argparse.ArgumentParser(description="Refined 2-D sweep to locate z ~ targets.")
    p.add_argument("--latent", required=True, help="latent_z_merged2.csv (must contain ax, ay, z)")
    p.add_argument("--ax_min", type=float, required=True)
    p.add_argument("--ax_max", type=float, required=True)
    p.add_argument("--ax_steps", type=int, default=121)
    p.add_argument("--ay_min", type=float, required=True)
    p.add_argument("--ay_max", type=float, required=True)
    p.add_argument("--ay_steps", type=int, default=121)
    p.add_argument("--targets", type=float, nargs="+", required=True,
                   help="z-targets, e.g. -1.4518 -1.45165 -1.4515")
    p.add_argument("--min_sep", type=float, default=0.002,
                   help="minimum separation in (ax, ay) between solutions")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    ensure_dir(args.outdir)
    grid_csv   = os.path.join(args.outdir, "grid_ax_ay_z.csv")
    matches_csv= os.path.join(args.outdir, "matches.csv")
    plot_png   = os.path.join(args.outdir, "heatmap_contours.png")

    # Load + fit
    df_lat = load_latent(args.latent)
    coef = fit_quad(df_lat)

    # Grid
    AX, AY, ax_lin, ay_lin = mk_grid(args.ax_min, args.ax_max, args.ax_steps,
                                     args.ay_min, args.ay_max, args.ay_steps)
    Z = z_hat(AX, AY, coef)

    # Save grid
    df_grid = pd.DataFrame({
        "ax": AX.ravel(),
        "ay": AY.ravel(),
        "z_pred": Z.ravel(),
    })
    df_grid.to_csv(grid_csv, index=False)

    # Find separated best points for each target
    targets = list(args.targets)
    best_pts = find_separated_best(AX, AY, Z, targets, args.min_sep)

    # Write matches CSV
    out_rows = []
    for k, (ax_b, ay_b, zb, err) in enumerate(best_pts, start=1):
        row = {
            "rank": k,
            "z_target": targets[k-1] if k-1 < len(targets) else np.nan,
            "ax": ax_b,
            "ay": ay_b,
            "z_pred": zb,
            "abs_err": err,
        }
        out_rows.append(row)
    pd.DataFrame(out_rows).to_csv(matches_csv, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(
        Z, origin="lower",
        extent=[ax_lin.min(), ax_lin.max(), ay_lin.min(), ay_lin.max()],
        aspect="auto"
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("z_pred")

    # Contours at target z-levels
    CS = ax.contour(
        AX, AY, Z,
        levels=targets, colors=["white", "yellow", "cyan"], linewidths=1.5
    )
    ax.clabel(CS, inline=True, fontsize=9, fmt=lambda v: f"z={v:.6f}")

    # Mark best points
    for k, (ax_b, ay_b, zb, err) in enumerate(best_pts, start=1):
        if not np.isnan(ax_b):
            ax.plot(ax_b, ay_b, marker="o", markersize=7, fillstyle="none",
                    markeredgewidth=2, color="black")
            ax.text(ax_b, ay_b, f"  #{k}\n  err={err:.2e}", fontsize=9,
                    va="bottom", ha="left", color="black")

    ax.set_title("Neutrino refined sweep: z-surface with target contours")
    ax.set_xlabel("ax")
    ax.set_ylabel("ay")
    plt.tight_layout()
    plt.savefig(plot_png, dpi=160)
    plt.close(fig)

    # Console summary (ASCII only)
    print("Refined sweep complete.")
    print(f"  Grid CSV     : {grid_csv}")
    print(f"  Matches CSV  : {matches_csv}")
    print(f"  Plot PNG     : {plot_png}")
    print("Best matches:")
    for k, (ax_b, ay_b, zb, err) in enumerate(best_pts, start=1):
        t = targets[k-1] if k-1 < len(targets) else float("nan")
        print(f"  #{k}: target={t:.6f}  ax={ax_b:.6f}  ay={ay_b:.6f}  z_pred={zb:.6f}  abs_err={err:.3e}")

if __name__ == "__main__":
    main()