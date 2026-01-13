import argparse, sys, numpy as np, pandas as pd

def fit_quad(ax, ay, z):
    X = np.column_stack([np.ones_like(ax), ax, ay, ax*ax, ax*ay, ay*ay])
    coef, *_ = np.linalg.lstsq(X, z, rcond=None)
    return coef  # [c0,c1,c2,c3,c4,c5]

def eval_z_and_partials(coef, ax, ay):
    c0,c1,c2,c3,c4,c5 = coef
    z  = c0 + c1*ax + c2*ay + c3*ax*ax + c4*ax*ay + c5*ay*ay
    zx = c1 + 2*c3*ax + c4*ay
    zy = c2 + c4*ax + 2*c5*ay
    zxx= 2*c3
    zxy= c4
    zyy= 2*c5
    return z,zx,zy,zxx,zxy,zyy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True, help="latent_z_merged*.csv (must contain columns: ax, ay, z or z_pred)")
    ap.add_argument("--nx", type=int, default=101, help="sampling along ax")
    ap.add_argument("--ny", type=int, default=101, help="sampling along ay")
    args = ap.parse_args()

    df = pd.read_csv(args.latent)
    # tolerant column name for z
    zcol = "z" if "z" in df.columns else ("z_pred" if "z_pred" in df.columns else None)
    if zcol is None:
        print("[ERROR] latent CSV must have 'z' or 'z_pred' column.", file=sys.stderr)
        sys.exit(1)

    ax = df["ax"].to_numpy(dtype=float)
    ay = df["ay"].to_numpy(dtype=float)
    zz = df[zcol].to_numpy(dtype=float)

    coef = fit_quad(ax, ay, zz)

    ax_min, ax_max = float(ax.min()), float(ax.max())
    ay_min, ay_max = float(ay.min()), float(ay.max())
    axv = np.linspace(ax_min, ax_max, args.nx)
    ayv = np.linspace(ay_min, ay_max, args.ny)
    AX, AY = np.meshgrid(axv, ayv, indexing="xy")

    _, ZX, ZY, *_ = eval_z_and_partials(coef, AX, AY)

    gxx = 1.0 + ZX**2
    gyy = 1.0 + ZY**2

    Lx_eff = (ax_max - ax_min) * np.sqrt(gxx).mean()
    Ly_eff = (ay_max - ay_min) * np.sqrt(gyy).mean()
    ratio  = Ly_eff / Lx_eff

    print("=== Cycle lengths (effective) from latent surface ===")
    print(f"ax range: [{ax_min:.6f},{ax_max:.6f}]  -> Δax={ax_max-ax_min:.6f}")
    print(f"ay range: [{ay_min:.6f},{ay_max:.6f}]  -> Δay={ay_max-ay_min:.6f}")
    print(f"<sqrt(g_xx)> ≈ {np.sqrt(gxx).mean():.6g}")
    print(f"<sqrt(g_yy)> ≈ {np.sqrt(gyy).mean():.6g}")
    print(f"Lx_eff ∝ Δax * <sqrt(g_xx)> = {Lx_eff:.6g}")
    print(f"Ly_eff ∝ Δay * <sqrt(g_yy)> = {Ly_eff:.6g}")
    print(f"R_over_r ≡ Ly_eff/Lx_eff = {ratio:.6g}")

if __name__ == "__main__":
    main()