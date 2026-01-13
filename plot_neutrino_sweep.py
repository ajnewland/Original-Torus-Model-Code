# plot_neutrino_sweep.py
# Plots z_pred vs ay from a merged sweep CSV; overlays optional target z lines
# and reports local minima on a smoothed curve.
#
# Expected columns (case-insensitive): 'ay', 'z_pred'
# Anything extra is ignored.

import argparse
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Plot z_pred vs ay from neutrino ay-sweep.")
    p.add_argument("--csv", required=True, help="Path to merged neutrino sweep CSV (with ay,z_pred).")
    p.add_argument("--out", default=None, help="Output image path (PNG). If omitted, just shows the plot.")
    p.add_argument("--ax", type=float, default=None, help="Fixed ax used in the sweep (for title/annotation).")
    p.add_argument("--title", default=None, help="Custom plot title.")
    p.add_argument("--zlines", nargs="*", type=float, default=None,
                   help="Optional horizontal z target lines to draw (e.g. for nu1,nu2,nu3).")
    p.add_argument("--smooth_window", type=int, default=7,
                   help="Odd window size for rolling mean smoothing (default: 7).")
    p.add_argument("--smooth_min_pts", type=int, default=5,
                   help="Minimum points required to compute smoothing (default: 5).")
    return p.parse_args()

def find_col(df, names):
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n in cols:
            return cols[n]
    return None

def rolling_smooth(y, window):
    """Pandas rolling mean with 'center=True'; handles edges with NaNs then linear interp."""
    s = pd.Series(y).rolling(window=window, min_periods=max(1, window//2), center=True).mean()
    # interpolate gaps at the ends for nicer minima detection
    s = s.interpolate(limit_direction="both")
    return s.values

def find_local_minima(x, y):
    """Return indices of simple local minima on y(x) via sign-change of first differences."""
    if len(y) < 3:
        return []
    dy = np.diff(y)
    # sign change from negative to positive indicates a local min
    mins = []
    for i in range(1, len(dy)):
        if dy[i-1] < 0 and dy[i] > 0:
            # choose the index centered on the "valley" (i)
            mins.append(i)
    return mins

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    # Strip spaces and unify column names
    df.columns = [str(c).strip() for c in df.columns]
    # Locate required columns
    ay_col = find_col(df, {"ay"})
    z_col  = find_col(df, {"z_pred", "z", "z_value"})
    if ay_col is None or z_col is None:
        print("[ERROR] Could not find required columns 'ay' and 'z_pred' in CSV.", file=sys.stderr)
        print("Columns found:", list(df.columns), file=sys.stderr)
        sys.exit(1)

    # Coerce numeric, drop rows without valid values
    df[ay_col] = pd.to_numeric(df[ay_col], errors="coerce")
    df[z_col]  = pd.to_numeric(df[z_col], errors="coerce")
    df = df.dropna(subset=[ay_col, z_col])

    # Sort by ay and, if duplicates, average z_pred at identical ay
    g = df.groupby(ay_col, as_index=False)[z_col].mean()
    g = g.sort_values(ay_col).reset_index(drop=True)

    ay = g[ay_col].values
    zz = g[z_col].values

    # Smooth (if enough points)
    if len(zz) >= args.smooth_min_pts and args.smooth_window >= 3 and args.smooth_window % 2 == 1:
        z_smooth = rolling_smooth(zz, args.smooth_window)
    else:
        z_smooth = zz.copy()

    # Find local minima on smoothed curve
    min_idx = find_local_minima(ay, z_smooth)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ay, zz, s=18, label="z_pred (samples)")
    ax.plot(ay, z_smooth, linewidth=2, label=f"smoothed (win={args.smooth_window})")

    # Optional target lines
    if args.zlines:
        for i, zl in enumerate(args.zlines, 1):
            lbl = f"target z{ i } = {zl:.6f}"
            ax.axhline(zl, linestyle="--", linewidth=1, alpha=0.9, label=lbl)

    # Annotate minima
    if min_idx:
        for i in min_idx:
            ax.plot(ay[i], z_smooth[i], marker="v", markersize=7)
            ax.annotate(f"min\nay={ay[i]:.6f}\nz={z_smooth[i]:.6f}",
                        (ay[i], z_smooth[i]),
                        textcoords="offset points", xytext=(6, -6), ha="left", va="top")
        # Print to stdout too
        print("\n[Local minima on smoothed curve]")
        for i in min_idx:
            print(f"  ay={ay[i]:.9f}  z={z_smooth[i]:.9f}")

    ax.set_xlabel("ay")
    ax.set_ylabel("z")
    ttl = args.title or "Neutrino sweep: z vs ay"
    if args.ax is not None:
        ttl += f" (ax={args.ax})"
    ax.set_title(ttl)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    plt.tight_layout()

    if args.out:
        out = args.out
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=200)
        print(f"\n[OK] Saved plot to: {out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()