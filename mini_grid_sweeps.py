# mini_grid_sweeps.py
# Validate each locked fermion by scanning a small grid around (ax, ay)
# and comparing z_pred to the target z. Windows/CMD friendly (no PowerShell needed).

import argparse
import csv
from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd


def fit_z_surface(latent_csv):
    """
    Fit z(ax, ay) = c0 + c1*ax + c2*ay + c3*ax^2 + c4*ax*ay + c5*ay^2
    from a latent CSV with columns: ax, ay, z (and possibly r).
    """
    df = pd.read_csv(latent_csv)
    # Clean and coerce
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    for col in ["ax", "ay", "z"]:
        if col not in df.columns:
            raise ValueError(f"latent CSV needs column '{col}'")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ax", "ay", "z"]).reset_index(drop=True)

    ax = df["ax"].to_numpy(dtype=float)
    ay = df["ay"].to_numpy(dtype=float)
    z  = df["z" ].to_numpy(dtype=float)

    # Design matrix (n, 6)
    X = np.column_stack([
        np.ones_like(ax),
        ax,
        ay,
        ax**2,
        ax*ay,
        ay**2
    ])

    coef, *_ = np.linalg.lstsq(X, z, rcond=None)

    # bounding box (for warnings)
    box = dict(
        ax_min=float(np.min(ax)),
        ax_max=float(np.max(ax)),
        ay_min=float(np.min(ay)),
        ay_max=float(np.max(ay)),
    )

    return coef, box


def z_pred_quad(coef, ax, ay):
    c0, c1, c2, c3, c4, c5 = coef
    return (c0 + c1*ax + c2*ay + c3*ax*ax + c4*ax*ay + c5*ay*ay)


def load_locked(locked_csv):
    """
    Expect columns: species, z_target, ax, ay
    (other columns are ignored).
    """
    df = pd.read_csv(locked_csv)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    required = ["species", "z_target", "ax", "ay"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"locked CSV missing columns: {missing}")

    df["species"]  = df["species"].astype(str)
    df["z_target"] = pd.to_numeric(df["z_target"], errors="coerce")
    df["ax"]       = pd.to_numeric(df["ax"], errors="coerce")
    df["ay"]       = pd.to_numeric(df["ay"], errors="coerce")
    df = df.dropna(subset=["species", "z_target", "ax", "ay"]).reset_index(drop=True)

    return df


def make_offsets(step, radius):
    """
    Build a symmetric 1D offset array: e.g., radius=2, step=0.004 -> [-.008, -.004, 0, .004, .008]
    """
    return np.arange(-radius, radius+1, dtype=int) * float(step)


def scan_one(species, ax0, ay0, z_target, coef, box, offsets_ax, offsets_ay, out_dir=None):
    """
    Evaluate the mini-grid around (ax0, ay0) and print a compact table.
    Optionally dump a per-species CSV to out_dir.
    """
    rows = []
    best_abs = math.inf
    best_idx = (-1, -1)
    center_idx = None

    # Build grid
    for i, da in enumerate(offsets_ax):
        for j, db in enumerate(offsets_ay):
            ax = ax0 + da
            ay = ay0 + db
            zp = z_pred_quad(coef, ax, ay)
            dz = zp - z_target

            # Track best |dz|
            adz = abs(dz)
            if adz < best_abs:
                best_abs = adz
                best_idx = (i, j)

            # Remember the center index
            if abs(da) < 1e-15 and abs(db) < 1e-15:
                center_idx = (i, j)

            rows.append({
                "species": species,
                "d_ax": float(da),
                "d_ay": float(db),
                "ax": float(ax),
                "ay": float(ay),
                "z_pred": float(zp),
                "dz": float(dz)
            })

    # Print header
    print(f"\n=== {species} ===")
    print(f"center: ax={ax0:.6f}  ay={ay0:.6f}  z_target={z_target:.9f}")
    print(f"latent box: ax∈[{box['ax_min']:.4f},{box['ax_max']:.4f}], ay∈[{box['ay_min']:.4f},{box['ay_max']:.4f}]")
    if not (box["ax_min"] <= ax0 <= box["ax_max"] and box["ay_min"] <= ay0 <= box["ay_max"]):
        print("  [!] center point lies outside the latent fit box — consider extending latent data there.")

    # Pretty-print a compact grid of dz (center in brackets)
    # Arrange as a matrix in the same order we computed (i=offsets_ax rows, j=offsets_ay cols)
    w = len(offsets_ay)
    matrix = [rows[r*w:(r+1)*w] for r in range(len(offsets_ax))]

    # Column header
    colhdr = "d_ay -> " + " ".join(f"{db:+.4f}" for db in offsets_ay)
    print(colhdr)
    for i, row in enumerate(matrix):
        label = f"d_ax={offsets_ax[i]:+.4f} : "
        line = []
        for j, cell in enumerate(row):
            val = cell["dz"]
            cell_txt = f"{val:+.6e}"
            if center_idx == (i, j):
                cell_txt = f"[{cell_txt}]"
            elif best_idx == (i, j):
                cell_txt = f"*{cell_txt}*"
            line.append(cell_txt)
        print(label + " ".join(line))

    # One-line summary
    ci, cj = center_idx
    dz_center = matrix[ci][cj]["dz"]
    best_da = offsets_ax[best_idx[0]]
    best_db = offsets_ay[best_idx[1]]
    print(f"center |dz| = {abs(dz_center):.3e}; best |dz| in grid = {best_abs:.3e} at d_ax={best_da:+.4f}, d_ay={best_db:+.4f}")

    # Optional CSV dump per species
    if out_dir:
        out_path = Path(out_dir) / f"mini_grid_{species}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[saved] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Mini-grid validation around locked fermion axes.")
    ap.add_argument("--latent", required=True, help="Path to latent_z_merged*.csv (must include ax, ay, z).")
    ap.add_argument("--locked", required=True, help="Path to fermions_locked.csv (must include species, z_target, ax, ay).")
    ap.add_argument("--step", type=float, default=0.004, help="Grid step for ax and ay (default 0.004).")
    ap.add_argument("--radius", type=int, default=2, help="How many steps on each side (2 => 5x5 grid).")
    ap.add_argument("--outdir", default="", help="Optional directory to write per-species mini-grid CSVs.")
    args = ap.parse_args()

    coef, box = fit_z_surface(args.latent)
    locked = load_locked(args.locked)

    offsets_ax = make_offsets(args.step, args.radius)
    offsets_ay = make_offsets(args.step, args.radius)

    print("[OK] z-surface fitted (deg=2).")
    print(f"     box: ax∈[{box['ax_min']:.4f},{box['ax_max']:.4f}], ay∈[{box['ay_min']:.4f},{box['ay_max']:.4f}]")
    print(f"     grid: size={(2*args.radius+1)}x{(2*args.radius+1)}, step={args.step:g}\n")

    out_dir = args.outdir if args.outdir.strip() else None

    for _, row in locked.iterrows():
        species = str(row["species"])
        ax0 = float(row["ax"])
        ay0 = float(row["ay"])
        zt  = float(row["z_target"])
        scan_one(species, ax0, ay0, zt, coef, box, offsets_ax, offsets_ay, out_dir=out_dir)


if __name__ == "__main__":
    main()