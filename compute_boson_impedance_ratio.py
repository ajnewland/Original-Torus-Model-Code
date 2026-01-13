# compute_boson_impedance_ratio.py
# Route 1: principal-curvature / impedance ratio on a z(ax,ay) grid
# Usage (example):
#   python compute_boson_impedance_ratio.py --grid "C:\path\to\grid_ax_ay_z.csv" --outdir "C:\path\to\out"

import argparse, os, sys, math
import numpy as np
import pandas as pd

def finite_diff_1d(arr, h):
    """
    Central difference with 1st order accurate edges set to NaN.
    Returns derivative of same shape as arr.
    """
    der = np.full_like(arr, np.nan, dtype=float)
    # interior central
    der[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / (2.0*h)
    # edges: leave NaN (we'll mask later)
    return der

def finite_diff_2nd_1d(arr, h):
    """
    Second derivative (Laplacian along axis) with central differences.
    Edges set to NaN.
    """
    d2 = np.full_like(arr, np.nan, dtype=float)
    d2[:, 1:-1] = (arr[:, 2:] - 2.0*arr[:, 1:-1] + arr[:, :-2]) / (h*h)
    return d2

def mixed_derivative(arr, hx, hy):
    """
    Mixed derivative z_xy using two-step central differences.
    Edges set to NaN.
    """
    # first derivative in x
    zx = np.full_like(arr, np.nan, dtype=float)
    zx[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / (2.0*hx)
    # then derivative of zx in y
    zxy = np.full_like(arr, np.nan, dtype=float)
    zxy[1:-1, :] = (zx[2:, :] - zx[:-2, :]) / (2.0*hy)
    # edges remain NaN
    return zxy

def main():
    ap = argparse.ArgumentParser(description="Compute geometric mixing ratio s* from z(ax,ay) grid.")
    ap.add_argument("--grid", required=True, help="Path to grid_ax_ay_z.csv (columns: ax,ay,z)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--png", action="store_true", help="If set, write a heatmap PNG (requires matplotlib)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load grid CSV
    df = pd.read_csv(args.grid)
    # Normalize column names defensively
    cols = {c.lower(): c for c in df.columns}
    need = ["ax","ay","z"]
    for k in need:
        if k not in cols and k not in df.columns:
            print(f"[ERROR] CSV must contain column '{k}'. Found: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
    ax_col = cols.get("ax","ax")
    ay_col = cols.get("ay","ay")
    z_col  = cols.get("z","z")

    # Extract unique axes and pivot to 2D arrays
    ax_vals = np.sort(df[ax_col].unique())
    ay_vals = np.sort(df[ay_col].unique())
    nx, ny = ax_vals.size, ay_vals.size
    # sanity
    if nx*ny != len(df):
        print("[ERROR] grid is not rectangular. Check input CSV.", file=sys.stderr)
        sys.exit(1)

    # Pivot to Z[iy, ix] where x->ax, y->ay
    Z = df.pivot(index=ay_col, columns=ax_col, values=z_col).sort_index(axis=0).sort_index(axis=1).to_numpy()
    # Build mesh for later output
    AX, AY = np.meshgrid(ax_vals, ay_vals)

    # Step sizes
    if nx < 3 or ny < 3:
        print("[ERROR] need at least 3x3 grid to compute derivatives.", file=sys.stderr)
        sys.exit(1)

    hx = float(ax_vals[1] - ax_vals[0])
    hy = float(ay_vals[1] - ay_vals[0])

    # First derivatives (z_x, z_y)
    # Note: our derivative helpers assume axis-1 is x; we transposed accordingly via pivot.
    Zx = finite_diff_1d(Z, hx)  # derivative along columns (x)
    Zy = finite_diff_1d(Z.T, hy).T  # derivative along rows (y); compute on transpose then transpose back

    # Second derivatives (z_xx, z_yy) and mixed (z_xy)
    Zxx = finite_diff_2nd_1d(Z, hx)
    Zyy = finite_diff_2nd_1d(Z.T, hy).T
    Zxy = mixed_derivative(Z, hx, hy)

    # Build Hessian and get principal curvatures (proxy: eigenvalues of Hessian)
    # For small slopes, eigenvalues of Hessian are proportional to principal curvatures of the graph.
    # We'll compute impedances: I1 = sqrt(1+Zx^2) * |k1|, I2 = sqrt(1+Zy^2) * |k2|
    # This aligns with the "impedance" idea (metric factor times curvature magnitude).
    s_ratio_pc = np.full_like(Z, np.nan, dtype=float)
    s_ratio_axis = np.full_like(Z, np.nan, dtype=float)

    for j in range(1, ny-1):
        for i in range(1, nx-1):
            zxx = Zxx[j, i]
            zyy = Zyy[j, i]
            zxy = Zxy[j, i]
            zx  = Zx[j, i]
            zy  = Zy[j, i]
            if np.any(np.isnan([zxx, zyy, zxy, zx, zy])):
                continue

            H = np.array([[zxx, zxy],
                          [zxy, zyy]], dtype=float)
            w, _ = np.linalg.eigh(H)  # eigenvalues sorted ascending
            k1, k2 = w[0], w[1]

            # principal-curvature impedance ratio
            I1 = math.sqrt(1.0 + zx*zx) * abs(k1)
            I2 = math.sqrt(1.0 + zy*zy) * abs(k2)
            denom = I1 + I2
            if denom > 0:
                s_ratio_pc[j, i] = I2 / denom

            # axis-aligned surrogate (may be noisier but useful cross-check)
            Ix = math.sqrt(1.0 + zx*zx) * abs(zxx)
            Iy = math.sqrt(1.0 + zy*zy) * abs(zyy)
            denom2 = Ix + Iy
            if denom2 > 0:
                s_ratio_axis[j, i] = Iy / denom2

    # Flatten valid interior samples
    def valid_stats(arr):
        v = arr[~np.isnan(arr)]
        if v.size == 0:
            return None
        return {
            "count": int(v.size),
            "mean": float(np.nanmean(v)),
            "std": float(np.nanstd(v)),
            "min": float(np.nanmin(v)),
            "q10": float(np.nanpercentile(v,10)),
            "median": float(np.nanmedian(v)),
            "q90": float(np.nanpercentile(v,90)),
            "max": float(np.nanmax(v)),
        }

    pc_stats   = valid_stats(s_ratio_pc)
    ax_stats   = valid_stats(s_ratio_axis)

    # Save map CSV
    out_map_csv = os.path.join(args.outdir, "impedance_ratio_map.csv")
    out_df = pd.DataFrame({
        "ax": AX.ravel(),
        "ay": AY.ravel(),
        "s_ratio_pc": s_ratio_pc.ravel(),
        "s_ratio_axis": s_ratio_axis.ravel()
    })
    out_df.to_csv(out_map_csv, index=False)

    # Save summary
    out_sum = os.path.join(args.outdir, "summary.txt")
    with open(out_sum, "w") as f:
        f.write("Impedance ratio s* = I2/(I1+I2)\n")
        f.write("s_ratio_pc = principal-curvature version (Hessian eigenvalues)\n")
        f.write("s_ratio_axis = axis-aligned surrogate (z_xx, z_yy)\n\n")

        if pc_stats:
            f.write("[s_ratio_pc stats]\n")
            for k,v in pc_stats.items():
                f.write(f"{k}: {v:.9f}\n")
            f.write("\n")
        else:
            f.write("[s_ratio_pc stats]\nno valid interior points\n\n")

        if ax_stats:
            f.write("[s_ratio_axis stats]\n")
            for k,v in ax_stats.items():
                f.write(f"{k}: {v:.9f}\n")
            f.write("\n")
        else:
            f.write("[s_ratio_axis stats]\nno valid interior points\n\n")

        f.write("Note: values are computed only on interior points (edges omitted).\n")

    print(f"[OK] wrote {out_map_csv}")
    print(f"[OK] wrote {out_sum}")

    # Optional heatmap
    if args.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(7,5))
            # plot principal-curvature s*; mask NaNs
            M = np.ma.masked_invalid(s_ratio_pc)
            im = plt.pcolormesh(ax_vals, ay_vals, M, shading="auto")
            plt.xlabel("ax")
            plt.ylabel("ay")
            plt.title("Impedance ratio s* (principal-curvature proxy)")
            cbar = plt.colorbar(im)
            cbar.set_label("s*")
            out_png = os.path.join(args.outdir, "impedance_ratio_heatmap.png")
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            print(f"[OK] wrote {out_png}")
        except Exception as e:
            print(f"[WARN] could not write PNG: {e}")

if __name__ == "__main__":
    main()