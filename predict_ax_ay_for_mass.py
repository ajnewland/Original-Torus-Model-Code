# -*- coding: utf-8 -*-
"""
Predict (ax, ay) for target PDG masses by matching the latent z-surface.
Robust CSV reader: auto-detects delimiter, trims whitespace/BOM, flexible column names.

Usage (CMD):
python "C:\...\predict_ax_ay_for_mass.py" ^
  --latent "C:\...\geom_fit_many\latent_z.csv" ^
  --iso    "C:\...\anchor_free_iso_full_many\anchor_free_isotonic_masses.csv" ^
  --sin2 0.231 --alpha 0.0647 --beta 0.5529 ^
  --masses "mu:0.10566,t:172.76,e:0.000511" ^
  --out "C:\...\Predicted Masses\predicted_ax_ay_for_masses.csv"
"""
import argparse, csv, math
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- I/O ----------------
def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _read_csv_robust(path: str):
    # Auto-detect delimiter (comma/semicolon/tab), keep strings initially
    df = pd.read_csv(path, sep=None, engine="python", dtype=str, na_filter=False)
    # Normalize column names: strip, lower, remove BOM if present
    norm = {c: c.encode('utf-8').decode('utf-8').strip().lstrip('\ufeff').lower()
            for c in df.columns}
    df.rename(columns=norm, inplace=True)
    # Strip whitespace from all cells
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def read_latent(path: str):
    df = _read_csv_robust(path)
    c_ax = _pick_col(df, ["ax","a_x"])
    c_ay = _pick_col(df, ["ay","a_y"])
    c_z  = _pick_col(df, ["z","latent_z","z_latent","z_rule"])

    if c_ax is None or c_ay is None or c_z is None:
        raise ValueError(f"[latent] Need ax/ay/z-like columns. Found: {list(df.columns)}")

    ax = pd.to_numeric(df[c_ax], errors="coerce").to_numpy()
    ay = pd.to_numeric(df[c_ay], errors="coerce").to_numpy()
    zz = pd.to_numeric(df[c_z],  errors="coerce").to_numpy()

    m = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(zz)
    ax, ay, zz = ax[m], ay[m], zz[m]
    if ax.size < 12:
        # Print a small debug to help
        preview = df.head(5).to_dict(orient="list")
        raise ValueError(
            f"Too few valid (ax,ay,z) rows to fit a surface (got {ax.size}).\n"
            f"Columns seen: {list(df.columns)}\n"
            f"Sample rows (first 5): {preview}"
        )
    return ax, ay, zz

def read_iso(path: str):
    df = _read_csv_robust(path)
    # Required columns (flexible naming)
    c_species = _pick_col(df, ["species"])
    c_z       = _pick_col(df, ["z","latent_z","z_latent","z_rule"])
    c_qiso    = _pick_col(df, ["q_iso","qiso","q"])
    c_logm    = _pick_col(df, ["logm_anchorfree","logm","logm_free"])
    c_mgev    = _pick_col(df, ["m_anchorfree_gev","mgev","mass_gev"])

    need = [c_species, c_z, c_qiso, c_logm, c_mgev]
    if any(c is None for c in need):
        raise ValueError(f"[iso] Missing required columns in {path}. Have: {list(df.columns)}")

    out = pd.DataFrame({
        "species": df[c_species],
        "z":       pd.to_numeric(df[c_z],    errors="coerce"),
        "q_iso":   pd.to_numeric(df[c_qiso], errors="coerce"),
        "logm_anchorfree": pd.to_numeric(df[c_logm], errors="coerce"),
        "m_anchorfree_GeV": pd.to_numeric(df[c_mgev], errors="coerce"),
    }).dropna().sort_values("z").reset_index(drop=True)
    if len(out) < 6:
        raise ValueError(f"[iso] Too few rows after cleaning: {len(out)}")
    return out

def parse_masses(spec: str):
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok: continue
        k,v = tok.split(":")
        out.append((k.strip(), float(v)))
    return out

# ---------------- fit z-surface ----------------
def poly_terms(ax, ay, deg=2):
    ax = np.asarray(ax); ay = np.asarray(ay)
    if deg == 2:
        return np.column_stack([np.ones_like(ax), ax, ay, ax*ay, ax*ax, ay*ay])
    elif deg == 3:
        return np.column_stack([
            np.ones_like(ax), ax, ay, ax*ay, ax*ax, ay*ay,
            ax*ax*ax, ay*ay*ay, (ax*ax)*ay, ax*(ay*ay)
        ])
    else:
        raise ValueError("--deg must be 2 or 3")

def fit_surface(ax, ay, zz, deg=2):
    X = poly_terms(ax, ay, deg=deg)
    coef, *_ = np.linalg.lstsq(X, zz, rcond=None)
    return coef

def z_hat_fun(coef, deg=2):
    def f(ax, ay):
        X = poly_terms(np.array([ax]), np.array([ay]), deg=deg)
        return float(X @ coef)
    return f

# ---------------- invert isotonic: mass -> z_target ----------------
def mass_to_z_target(df_iso, m_GeV: float):
    logm = math.log(m_GeV)
    z_arr  = df_iso["z"].to_numpy()
    q_arr  = df_iso["q_iso"].to_numpy()
    log_arr = df_iso["logm_anchorfree"].to_numpy()

    # logm -> q
    order = np.argsort(log_arr)
    log_arr2, q_arr2 = log_arr[order], q_arr[order]
    q_tgt = float(np.interp(logm, log_arr2, q_arr2, left=q_arr2[0], right=q_arr2[-1]))

    # q -> z
    order2 = np.argsort(q_arr)
    q_arr3, z_arr3 = q_arr[order2], z_arr[order2]
    z_tgt = float(np.interp(q_tgt, q_arr3, z_arr3, left=z_arr3[0], right=z_arr3[-1]))
    return logm, q_tgt, z_tgt

# ---------------- search (ax,ay) for z_target ----------------
def coarse_refine_search(zfun, z_target,
                         box, grid_n=81, refine_n=81,
                         tol_z=1e-6, max_expansions=6, expand_factor=1.35):
    ax_min, ax_max, ay_min, ay_max = box
    best = None

    for it in range(max_expansions+1):
        axs = np.linspace(ax_min, ax_max, grid_n)
        ays = np.linspace(ay_min, ay_max, grid_n)
        AX, AY = np.meshgrid(axs, ays, indexing="xy")
        Z = np.empty_like(AX)
        for i in range(AX.shape[0]):
            for j in range(AX.shape[1]):
                Z[i,j] = zfun(AX[i,j], AY[i,j])
        ERR = np.abs(Z - z_target)
        idx = np.unravel_index(np.argmin(ERR), ERR.shape)
        a0, b0, e0 = float(AX[idx]), float(AY[idx]), float(ERR[idx])

        half_ax = 0.15*(ax_max - ax_min)
        half_ay = 0.15*(ay_max - ay_min)
        r_ax_min = max(ax_min, a0 - half_ax); r_ax_max = min(ax_max, a0 + half_ax)
        r_ay_min = max(ay_min, b0 - half_ay); r_ay_max = min(ay_max, b0 + half_ay)

        raxs = np.linspace(r_ax_min, r_ax_max, refine_n)
        rays = np.linspace(r_ay_min, r_ay_max, refine_n)
        RAX, RAY = np.meshgrid(raxs, rays, indexing="xy")
        RZ = np.empty_like(RAX)
        for i in range(RAX.shape[0]):
            for j in range(RAX.shape[1]):
                RZ[i,j] = zfun(RAX[i,j], RAY[i,j])
        RERR = np.abs(RZ - z_target)
        ridx = np.unravel_index(np.argmin(RERR), RERR.shape)
        ax_best, ay_best, err_best = float(RAX[ridx]), float(RAY[ridx]), float(RERR[ridx])
        z_best = float(RZ[ridx])

        eps_b = 1e-12
        on_b = (abs(ax_best-ax_min) < eps_b or abs(ax_best-ax_max) < eps_b or
                abs(ay_best-ay_min) < eps_b or abs(ay_best-ay_max) < eps_b)

        best = dict(ax=ax_best, ay=ay_best, z_pred=z_best, abs_err=err_best,
                    on_boundary=bool(on_b), ok=(err_best<=tol_z),
                    expansions=it, note="")

        if best["ok"]:
            best["note"] = "ok"
            best["suggest"] = _neighbors(ax_best, ay_best, zfun, z_target)
            return best

        if not on_b:
            best["note"] = "local_min_interior"
            best["suggest"] = _neighbors(ax_best, ay_best, zfun, z_target)
            return best

        # expand box
        ax_c = 0.5*(ax_min + ax_max); ay_c = 0.5*(ay_min + ay_max)
        ax_half = 0.5*(ax_max - ax_min) * expand_factor
        ay_half = 0.5*(ay_max - ay_min) * expand_factor
        ax_min, ax_max = ax_c - ax_half, ax_c + ax_half
        ay_min, ay_max = ay_c - ay_half, ay_c + ay_half

    best["note"] = "no_bracket_after_expansion"
    best["suggest"] = _neighbors(best["ax"], best["ay"], zfun, z_target)
    return best

def _neighbors(ax, ay, zfun, z_target):
    ds = [(-0.06,0), (0.06,0), (0,-0.06), (0,0.06), (-0.06,-0.06), (0.06,0.06)]
    items=[]
    for dx,dy in ds:
        a,b = ax+dx, ay+dy
        z = zfun(a,b)
        items.append( (float(a), float(b), float(z), float(abs(z - z_target))) )
    items.sort(key=lambda t: t[3])
    return items

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True)
    ap.add_argument("--iso", required=True)
    ap.add_argument("--sin2", type=float, default=0.231)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--beta",  type=float, required=True)
    ap.add_argument("--masses", required=True, help='e.g. "mu:0.10566,t:172.76,e:0.000511"')
    ap.add_argument("--out", required=True)

    ap.add_argument("--deg", type=int, default=2, choices=[2,3])
    ap.add_argument("--tol_z", type=float, default=1e-6)
    ap.add_argument("--grid_n", type=int, default=81)
    ap.add_argument("--refine_n", type=int, default=81)
    ap.add_argument("--max_expansions", type=int, default=6)
    ap.add_argument("--expand_factor", type=float, default=1.35)

    args = ap.parse_args()

    ax_arr, ay_arr, z_arr = read_latent(args.latent)
    df_iso = read_iso(args.iso)
    masses = parse_masses(args.masses)

    coef = fit_surface(ax_arr, ay_arr, z_arr, deg=args.deg)
    zfun = z_hat_fun(coef, deg=args.deg)

    ax_min, ax_max = float(np.min(ax_arr)), float(np.max(ax_arr))
    ay_min, ay_max = float(np.min(ay_arr)), float(np.max(ay_arr))
    box0 = [ax_min, ax_max, ay_min, ay_max]

    print(f"[OK] fitted z-surface (deg={args.deg}) on {len(z_arr)} points.")
    print(f"     box0: ax∈[{ax_min:.4f},{ax_max:.4f}], ay∈[{ay_min:.4f},{ay_max:.4f}]")
    print(f"     sin2={args.sin2} (α={args.alpha}, β={args.beta})  [reference only]")

    rows=[]
    for sp, m in masses:
        logm, q_tgt, z_tgt = mass_to_z_target(df_iso, m)
        res = coarse_refine_search(
            zfun, z_tgt, box=box0,
            grid_n=args.grid_n, refine_n=args.refine_n,
            tol_z=args.tol_z, max_expansions=args.max_expansions,
            expand_factor=args.expand_factor
        )
        z_chk = zfun(res["ax"], res["ay"])
        rows.append({
            "species": sp,
            "m_GeV": m,
            "logm": logm,
            "q_target": q_tgt,
            "z_target": z_tgt,
            "ax": res["ax"],
            "ay": res["ay"],
            "z_pred": z_chk,
            "abs_err": res["abs_err"],
            "ok": res["ok"],
            "note": res["note"],
            "expansions": res["expansions"]
        })

        print(f"\n[{sp}] m={m:g} GeV  -> z_target={z_tgt:.9f}")
        print(f"     best: ax={res['ax']:.6f}  ay={res['ay']:.6f}  ẑ={z_chk:.9f}  |Δ|={res['abs_err']:.3e}  ok={res['ok']}  ({res['note']})")
        if not res["ok"]:
            print("     Suggest sweeps near:")
            for (a,b,zp,err) in res["suggest"][:6]:
                print(f"       (ax,ay)=({a:.4f},{b:.4f})  ẑ={zp:.9f}  |Δ|={err:.3e}")

    outp = Path(args.out)
    ensure_dir(outp)
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\n[OK] wrote {outp}")
    df_out = pd.DataFrame(rows)
    print("\n--- results ---")
    print(df_out.to_string(index=False))

if __name__ == "__main__":
    main()