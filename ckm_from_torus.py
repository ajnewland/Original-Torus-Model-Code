# ckm_from_torus.py
import argparse, os, sys, math
import numpy as np
import pandas as pd

UP = ["u","c","t"]
DOWN = ["d","s","b"]

def load_species_csv(path):
    df = pd.read_csv(path)
    # Expected columns: species, ax, ay (others ignored)
    req = {"species","ax","ay"}
    miss = req - set(c.lower() for c in df.columns)
    # Map columns case-insensitively
    cols = {c.lower(): c for c in df.columns}
    if miss:
        raise ValueError(f"species CSV missing columns: {miss}. Got {list(df.columns)}")
    species = df[cols["species"]].astype(str).str.strip()
    ax = pd.to_numeric(df[cols["ax"]], errors="coerce")
    ay = pd.to_numeric(df[cols["ay"]], errors="coerce")
    out = pd.DataFrame({"species":species, "ax":ax, "ay":ay}).dropna()
    # Keep only needed 6 species for CKM
    out = out[out["species"].isin(UP+DOWN)].copy()
    if set(UP+DOWN) - set(out["species"]):
        missing = list((set(UP+DOWN) - set(out["species"])))
        raise ValueError(f"Missing species in table: {missing}")
    # Build ordered arrays
    up_df = out.set_index("species").loc[UP].reset_index()
    dn_df = out.set_index("species").loc[DOWN].reset_index()
    return up_df, dn_df

def distance_matrix(up_df, dn_df, alpha_x=1.0, alpha_y=1.0):
    U = up_df.shape[0]
    D = dn_df.shape[0]
    out = np.zeros((U,D), float)
    for i in range(U):
        for j in range(D):
            dx = up_df.loc[i,"ax"] - dn_df.loc[j,"ax"]
            dy = up_df.loc[i,"ay"] - dn_df.loc[j,"ay"]
            out[i,j] = math.sqrt(alpha_x*(dx*dx) + alpha_y*(dy*dy))
    return out

def inverse_distance_proxy(D):
    # avoid divide-by-zero: add tiny eps
    eps = 1e-12
    W = 1.0 / (D + eps)
    W = W / W.sum(axis=1, keepdims=True)
    return W

def softmax_proxy(D, tau=None, tau_rows=None):
    # energy = D^2 / tau_row
    U, V = D.shape
    if tau_rows is None:
        if tau is None:
            tau = 0.02
        tau_rows = np.array([tau]*U, float)
    E = D**2
    out = np.zeros_like(D, float)
    for i in range(U):
        t = tau_rows[i]
        S = np.exp(-E[i,:]/max(t,1e-12))
        out[i,:] = S / S.sum()
    return out

# ---------- Latent grid utilities (optional) ----------

def load_latent_grid(latent_csv):
    if latent_csv is None or not os.path.exists(latent_csv):
        return None
    df = pd.read_csv(latent_csv)
    # accept either z_pred or z
    zcol = None
    for c in df.columns:
        if c.lower() in ("z_pred","z"):
            zcol = c
            break
    if zcol is None:
        return None
    # try to pivot to a regular grid
    try:
        g = df.pivot_table(index="ay", columns="ax", values=zcol)
        ax_vals = np.array(g.columns.tolist(), float)
        ay_vals = np.array(g.index.tolist(), float)
        Z = g.values
        return ax_vals, ay_vals, Z
    except Exception:
        return None

def gradient_bilinear(ax_vals, ay_vals, Z):
    # central differences in interior, one-sided at edges
    dZ_dax = np.zeros_like(Z)
    dZ_day = np.zeros_like(Z)
    # spacings (assume sorted)
    for j in range(Z.shape[0]):
        for i in range(Z.shape[1]):
            # d/dax
            if 0 < i < Z.shape[1]-1:
                dx = ax_vals[i+1]-ax_vals[i-1]
                dZ_dax[j,i] = (Z[j,i+1]-Z[j,i-1])/(dx if dx!=0 else 1e-12)
            elif i==0:
                dx = ax_vals[1]-ax_vals[0]
                dZ_dax[j,i] = (Z[j,1]-Z[j,0])/(dx if dx!=0 else 1e-12)
            else:
                dx = ax_vals[-1]-ax_vals[-2]
                dZ_dax[j,i] = (Z[j,-1]-Z[j,-2])/(dx if dx!=0 else 1e-12)
            # d/day
            if 0 < j < Z.shape[0]-1:
                dy = ay_vals[j+1]-ay_vals[j-1]
                dZ_day[j,i] = (Z[j+1,i]-Z[j-1,i])/(dy if dy!=0 else 1e-12)
            elif j==0:
                dy = ay_vals[1]-ay_vals[0]
                dZ_day[j,i] = (Z[1,i]-Z[0,i])/(dy if dy!=0 else 1e-12)
            else:
                dy = ay_vals[-1]-ay_vals[-2]
                dZ_day[j,i] = (Z[-1,i]-Z[-2,i])/(dy if dy!=0 else 1e-12)
    return dZ_dax, dZ_day

def bilinear_interp(ax_vals, ay_vals, F, ax, ay):
    # find cell
    if ax < ax_vals[0] or ax > ax_vals[-1] or ay < ay_vals[0] or ay > ay_vals[-1]:
        return None
    i = np.searchsorted(ax_vals, ax) - 1
    j = np.searchsorted(ay_vals, ay) - 1
    i = max(0, min(i, len(ax_vals)-2))
    j = max(0, min(j, len(ay_vals)-2))
    x0, x1 = ax_vals[i], ax_vals[i+1]
    y0, y1 = ay_vals[j], ay_vals[j+1]
    tx = 0.0 if x1==x0 else (ax - x0)/(x1-x0)
    ty = 0.0 if y1==y0 else (ay - y0)/(y1-y0)
    f00 = F[j,i]; f10 = F[j,i+1]; f01 = F[j+1,i]; f11 = F[j+1,i+1]
    return (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11

def phase_kernel(ax_vals, ay_vals, dZ_dax, dZ_day, ax1, ay1, ax2, ay2, nseg=25):
    # integrate grad · tangent along straight segment
    tx = ax2-ax1; ty = ay2-ay1
    L = math.hypot(tx,ty)
    if L < 1e-15:
        return 0.0
    txn, tyn = tx/L, ty/L
    s_vals = np.linspace(0.0, 1.0, nseg)
    acc = 0.0
    for s in s_vals:
        ax = ax1 + s*tx
        ay = ay1 + s*ty
        gx = bilinear_interp(ax_vals, ay_vals, dZ_dax, ax, ay)
        gy = bilinear_interp(ax_vals, ay_vals, dZ_day, ax, ay)
        if gx is None or gy is None:
            continue
        acc += (gx*txn + gy*tyn) * (L/(nseg-1 if nseg>1 else 1))
    return acc

def with_phase(Vsoft, up_df, dn_df, ax_vals, ay_vals, dZ_dax, dZ_day,
               eps_u=0.06, eps_c=0.03, eps_t=0.01):
    eps_map = {"u":eps_u, "c":eps_c, "t":eps_t}
    U, D = Vsoft.shape
    Vphi = Vsoft.copy()
    for i in range(U):
        ui = up_df.loc[i,"species"]
        ax1 = up_df.loc[i,"ax"]; ay1 = up_df.loc[i,"ay"]
        eps = float(eps_map.get(ui, 0.0))
        if eps==0.0:
            continue
        for j in range(D):
            ax2 = dn_df.loc[j,"ax"]; ay2 = dn_df.loc[j,"ay"]
            phi = phase_kernel(ax_vals, ay_vals, dZ_dax, dZ_day, ax1, ay1, ax2, ay2, nseg=31)
            Vphi[i,j] = Vphi[i,j] * (1.0 + eps*math.cos(phi))
        # re-normalize row
        row = Vphi[i,:].clip(min=1e-18)
        Vphi[i,:] = row/row.sum()
    return Vphi

def save_matrix_csv(M, rows, cols, path):
    df = pd.DataFrame(M, index=rows, columns=cols)
    df.index.name = ""
    df.to_csv(path, float_format="%.6f")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species_csv", required=True, help="CSV with columns: species, ax, ay (your pasted table).")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--alpha_x", type=float, default=1.0)
    ap.add_argument("--alpha_y", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=None, help="single temperature for softmax.")
    ap.add_argument("--tau_u", type=float, default=None)
    ap.add_argument("--tau_c", type=float, default=None)
    ap.add_argument("--tau_t", type=float, default=None)
    ap.add_argument("--latent_csv", default=None, help="Optional latent grid CSV with ax, ay, z_pred (or z).")
    ap.add_argument("--phase_eps_u", type=float, default=0.06)
    ap.add_argument("--phase_eps_c", type=float, default=0.03)
    ap.add_argument("--phase_eps_t", type=float, default=0.01)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    up_df, dn_df = load_species_csv(args.species_csv)
    # 1) distances
    D = distance_matrix(up_df, dn_df, alpha_x=args.alpha_x, alpha_y=args.alpha_y)
    V_inv = inverse_distance_proxy(D)
    save_matrix_csv(D, UP, DOWN, os.path.join(args.outdir, "ckm_distance_matrix.csv"))
    save_matrix_csv(V_inv, UP, DOWN, os.path.join(args.outdir, "ckm_inverse_distance_proxy.csv"))

    # 2) softmax
    tau_rows = None
    if any(t is not None for t in [args.tau_u, args.tau_c, args.tau_t]):
        tr = [args.tau_u, args.tau_c, args.tau_t]
        tau_rows = np.array([t if (t is not None and t>0) else 0.02 for t in tr], float)
    V_soft = softmax_proxy(D, tau=args.tau, tau_rows=tau_rows)
    save_matrix_csv(V_soft, UP, DOWN, os.path.join(args.outdir, "ckm_softmax_proxy.csv"))

    # 3) optional phase (needs latent grid)
    latent = load_latent_grid(args.latent_csv) if args.latent_csv else None
    if latent is not None:
        ax_vals, ay_vals, Z = latent
        dZ_dax, dZ_day = gradient_bilinear(ax_vals, ay_vals, Z)
        V_phi = with_phase(V_soft, up_df, dn_df, ax_vals, ay_vals, dZ_dax, dZ_day,
                           eps_u=args.phase_eps_u, eps_c=args.phase_eps_c, eps_t=args.phase_eps_t)
        save_matrix_csv(V_phi, UP, DOWN, os.path.join(args.outdir, "ckm_phase_proxy.csv"))
    else:
        V_phi = None

    # Print key entries so you can eyeball quickly
    def pick(df, r, c): return df.loc[r,c]
    def as_df(M): return pd.DataFrame(M, index=UP, columns=DOWN)

    print("\n=== Quick look (row, col): (V_us, V_cb, V_tb) ===")
    print("Inverse-dist :",
          round(pick(as_df(V_inv), "u","s"),3),
          round(pick(as_df(V_inv), "c","b"),3),
          round(pick(as_df(V_inv), "t","b"),3))
    print("Softmax      :",
          round(pick(as_df(V_soft), "u","s"),3),
          round(pick(as_df(V_soft), "c","b"),3),
          round(pick(as_df(V_soft), "t","b"),3))
    if V_phi is not None:
        print("Phase-kernel :",
              round(pick(as_df(V_phi), "u","s"),3),
              round(pick(as_df(V_phi), "c","b"),3),
              round(pick(as_df(V_phi), "t","b"),3))
    print(f"\n[WROTE] {args.outdir}\\ckm_distance_matrix.csv")
    print(f"[WROTE] {args.outdir}\\ckm_inverse_distance_proxy.csv")
    print(f"[WROTE] {args.outdir}\\ckm_softmax_proxy.csv")
    if V_phi is not None:
        print(f"[WROTE] {args.outdir}\\ckm_phase_proxy.csv")

if __name__ == "__main__":
    main()