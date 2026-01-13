# compute_boson_impedance_ratio_v2.py
# Accepts z or z_pred (or a custom column via --zcol) and computes s* maps.

import argparse, os, sys, math
import numpy as np
import pandas as pd

def finite_diff_1d(arr, h):
    der = np.full_like(arr, np.nan, dtype=float)
    der[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / (2.0*h)
    return der

def finite_diff_2nd_1d(arr, h):
    d2 = np.full_like(arr, np.nan, dtype=float)
    d2[:, 1:-1] = (arr[:, 2:] - 2.0*arr[:, 1:-1] + arr[:, :-2]) / (h*h)
    return d2

def mixed_derivative(arr, hx, hy):
    zx = np.full_like(arr, np.nan, dtype=float)
    zx[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / (2.0*hx)
    zxy = np.full_like(arr, np.nan, dtype=float)
    zxy[1:-1, :] = (zx[2:, :] - zx[:-2, :]) / (2.0*hy)
    return zxy

def main():
    ap = argparse.ArgumentParser(description="Compute geometric mixing ratio s* from z(ax,ay) grid.")
    ap.add_argument("--grid", required=True, help="Path to grid CSV (ax, ay, z or z_pred)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--png", action="store_true", help="Write a heatmap PNG (requires matplotlib)")
    ap.add_argument("--zcol", default="", help="Override z column name (e.g. z_pred). Leave empty to auto-detect.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.grid)
    # case-insensitive lookup
    lower_to_real = {c.lower(): c for c in df.columns}
    # find ax/ay
    for need in ("ax","ay"):
        if need not in lower_to_real:
            print(f"[ERROR] CSV must contain '{need}'. Found: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
    ax_col = lower_to_real["ax"]
    ay_col = lower_to_real["ay"]

    # find z-like column
    z_candidates = []
    if args.zcol:
        if args.zcol in df.columns:
            z_col = args.zcol
        elif args.zcol.lower() in lower_to_real:
            z_col = lower_to_real[args.zcol.lower()]
        else:
            print(f"[ERROR] Requested z column '{args.zcol}' not found. Available: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
    else:
        for k in ("z","z_pred","Z","Z_PRED","zpred"):
            if k.lower() in lower_to_real:
                z_candidates.append(lower_to_real[k.lower()])
        if not z_candidates:
            print(f"[ERROR] Could not find a z column. Looked for z / z_pred. Found: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
        z_col = z_candidates[0]

    # build rectangular grid
    ax_vals = np.sort(df[ax_col].unique())
    ay_vals = np.sort(df[ay_col].unique())
    nx, ny = ax_vals.size, ay_vals.size
    if nx*ny != len(df):
        print("[ERROR] Grid is not rectangular. Check input CSV.", file=sys.stderr)
        sys.exit(1)

    Z = df.pivot(index=ay_col, columns=ax_col, values=z_col).sort_index(axis=0).sort_index(axis=1).to_numpy()
    AX, AY = np.meshgrid(ax_vals, ay_vals)

    if nx < 3 or ny < 3:
        print("[ERROR] Need at least 3x3 grid to compute derivatives.", file=sys.stderr)
        sys.exit(1)

    hx = float(ax_vals[1] - ax_vals[0])
    hy = float(ay_vals[1] - ay_vals[0])

    Zx  = finite_diff_1d(Z, hx)
    Zy  = finite_diff_1d(Z.T, hy).T
    Zxx = finite_diff_2nd_1d(Z, hx)
    Zyy = finite_diff_2nd_1d(Z.T, hy).T
    Zxy = mixed_derivative(Z, hx, hy)

    s_ratio_pc   = np.full_like(Z, np.nan, dtype=float)
    s_ratio_axis = np.full_like(Z, np.nan, dtype=float)

    for j in range(1, ny-1):
        for i in range(1, nx-1):
            zxx = Zxx[j, i]; zyy = Zyy[j, i]; zxy = Zxy[j, i]
            zx  = Zx[j, i];  zy  = Zy[j, i]
            if np.any(np.isnan([zxx, zyy, zxy, zx, zy])):
                continue

            H = np.array([[zxx, zxy],[zxy, zyy]], dtype=float)
            w, _ = np.linalg.eigh(H)
            k1, k2 = w[0], w[1]

            I1 = math.sqrt(1.0 + zx*zx) * abs(k1)
            I2 = math.sqrt(1.0 + zy*zy) * abs(k2)
            if I1 + I2 > 0:
                s_ratio_pc[j, i] = I2 / (I1 + I2)

            Ix = math.sqrt(1.0 + zx*zx) * abs(zxx)
            Iy = math.sqrt(1.0 + zy*zy) * abs(zyy)
            if Ix + Iy > 0:
                s_ratio_axis[j, i] = Iy / (Ix + Iy)

    def stats(arr):
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

    pc_stats = stats(s_ratio_pc)
    ax_stats = stats(s_ratio_axis)

    out_map_csv = os.path.join(args.outdir, "impedance_ratio_map.csv")
    pd.DataFrame({
        "ax": AX.ravel(),
        "ay": AY.ravel(),
        "s_ratio_pc": s_ratio_pc.ravel(),
        "s_ratio_axis": s_ratio_axis.ravel()
    }).to_csv(out_map_csv, index=False)

    out_sum = os.path.join(args.outdir, "summary.txt")
    with open(out_sum, "w") as f:
        f.write(f"Grid file : {args.grid}\n")
        f.write(f"Using zcol : {z_col}\n\n")

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

        f.write("Note: interior points only (edges omitted).\n")

    print(f"[OK] wrote {out_map_csv}")
    print(f"[OK] wrote {out_sum}")

    if args.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            M = np.ma.masked_invalid(s_ratio_pc)
            plt.figure(figsize=(7,5))
            im = plt.pcolormesh(ax_vals, ay_vals, M, shading="auto")
            plt.xlabel("ax"); plt.ylabel("ay"); plt.title("Impedance ratio s* (principal-curvature)")
            cbar = plt.colorbar(im); cbar.set_label("s*")
            out_png = os.path.join(args.outdir, "impedance_ratio_heatmap.png")
            plt.tight_layout(); plt.savefig(out_png, dpi=150)
            print(f"[OK] wrote {out_png}")
        except Exception as e:
            print(f"[WARN] PNG not written: {e}")

if __name__ == "__main__":
    main()