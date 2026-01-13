import argparse, sys, numpy as np, pandas as pd, os

def load_grid(path):
    df = pd.read_csv(path)
    zcol = "z" if "z" in df.columns else ("z_pred" if "z_pred" in df.columns else None)
    if zcol is None:
        raise RuntimeError("Grid CSV must contain 'z' or 'z_pred'. Found: %s" % list(df.columns))
    ax = np.array(sorted(df["ax"].unique()))
    ay = np.array(sorted(df["ay"].unique()))
    Z = np.full((ay.size, ax.size), np.nan)
    for _,row in df.iterrows():
        i = np.searchsorted(ax, row["ax"])
        j = np.searchsorted(ay, row["ay"])
        if 0 <= i < ax.size and 0 <= j < ay.size:
            Z[j,i] = float(row[zcol])
    if np.isnan(Z).any():
        raise RuntimeError("Grid appears incomplete or not rectilinear.")
    return ax, ay, Z

def center_index(arr, val):
    # choose the nearest index to the provided value
    return int(np.clip(np.abs(arr - val).argmin(), 1, len(arr)-2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="refined grid CSV (ax,ay,z or z_pred)")
    ap.add_argument("--Ly_over_Lx", type=float, default=None, help="Optional: pass Ly/Lx directly (from compute_cycle_ratio.py)")
    ap.add_argument("--ax0", type=float, default=None, help="Optional: choose evaluation ax (otherwise median)")
    ap.add_argument("--ay0", type=float, default=None, help="Optional: choose evaluation ay (otherwise median)")
    args = ap.parse_args()

    ax, ay, Z = load_grid(args.grid)
    # finite differences (central)
    dax = ax[1]-ax[0]
    day = ay[1]-ay[0]
    if dax <= 0 or day <= 0:
        print("[ERROR] non-positive grid steps.", file=sys.stderr); sys.exit(1)

    AX, AY = np.meshgrid(ax, ay, indexing="xy")
    # choose eval point
    ax0 = np.median(ax) if args.ax0 is None else args.ax0
    ay0 = np.median(ay) if args.ay0 is None else args.ay0
    i = center_index(ax, ax0)
    j = center_index(ay, ay0)

    # first derivatives
    Zx = (Z[j, i+1] - Z[j, i-1]) / (2*dax)
    Zy = (Z[j+1, i] - Z[j-1, i]) / (2*day)
    # second derivatives
    Zxx = (Z[j, i+1] - 2*Z[j, i] + Z[j, i-1]) / (dax*dax)
    Zyy = (Z[j+1, i] - 2*Z[j, i] + Z[j-1, i]) / (day*day)
    Zxy = (Z[j+1, i+1] - Z[j+1, i-1] - Z[j-1, i+1] + Z[j-1, i-1]) / (4*dax*day)

    # Hessian in chart coords
    H = np.array([[Zxx, Zxy],
                  [Zxy, Zyy]], dtype=float)
    evals, evecs = np.linalg.eigh(H)
    k1, k2 = np.abs(evals[0]), np.abs(evals[1])  # principal curvatures (magnitude)

    # metric scale factors (first fundamental form proxy)
    gxx = 1.0 + Zx*Zx
    gyy = 1.0 + Zy*Zy
    # local scale factors
    sx = np.sqrt(gxx)
    sy = np.sqrt(gyy)

    # cycle-length ratio
    if args.Ly_over_Lx is None:
        print("[WARN] Ly/Lx not provided; using local scale-factor ratio sy/sx as crude proxy.")
        Ly_over_Lx = sy / sx
    else:
        Ly_over_Lx = float(args.Ly_over_Lx)

    # impedance proportional to L * |k|
    Ix = (1.0) * k1  # Lx factor cancels in final ratio if we pass Ly/Lx explicitly
    Iy = Ly_over_Lx * k2

    s_star = Iy / (Ix + Iy)

    print("=== Impedance mixing at chosen point ===")
    print(f"ax0,ay0 = {ax[i]:.6f}, {ay[j]:.6f}")
    print(f"Zx,Zy   = {Zx:.6e}, {Zy:.6e}")
    print(f"Zxx,Zxy,Zyy = {Zxx:.6e}, {Zxy:.6e}, {Zyy:.6e}")
    print(f"|k1|,|k2| = {k1:.6e}, {k2:.6e}")
    print(f"Ly/Lx     = {Ly_over_Lx:.6g}")
    print(f"s_star    = {s_star:.6f}")

if __name__ == "__main__":
    main()