#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
relationships_extended.py

One-stop audit for additional hidden relationships in the torus model.

It implements six tests (all optional, data-dependent):
  (T1) Band-slope universality across sectors (bootstrap CIs)
  (T2) Charge–curvature sign rule (Hessian / eigenvector alignment)   [requires per-species grids]
  (T3) Winding-fraction ladders (continued fractions vs generation)
  (T4) Mirror-centroid conservation (boson vs fermion)                [needs a combined locked CSV]
  (T5) CKM from geodesic triangles (MDS area ⇒ Jarlskog proxy)        [uses CKM distance CSV]
  (T6) Cycle-sum quasi-conservation along slices                      [needs latent grid]

Everything degrades gracefully if a file is absent.

Usage (examples)
---------------
python relationships_extended.py ^
  --outdir "C:\...\grand_audit_plus" ^
  --winding "C:\...\grand_audit_out\winding_rationals.csv" ^
  --sector_slopes "C:\...\grand_audit_out\sector_slopes.csv" ^
  --locked_csv "C:\...\all_particles_locked.csv" ^
  --ckm_dist "C:\...\grand_audit_out\ckm_distance_matrix_numeric.csv" ^
  --latent_grid "C:\...\geom_fit_many\latent_z_merged2.csv" ^
  --grid_glob "C:\...\Predicted Masses\**\grid_ax_ay_z.csv"

Notes
-----
- (T2) expects per-species refined grids with columns [ax, ay, z] (or z_pred). Pass via --grid_glob with a glob pattern.
- (T4) expects a tidy “locked” table with at least: species, ax, ay, sector (or can infer sector from species name).
- (T6) latent grid must have columns [ax, ay, z_pred] (or z).

Author: you+me
"""

import os, sys, glob, json, math
import argparse
import numpy as np
import pandas as pd

from pathlib import Path

# plotting (optional)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- small utils ----------

def safe_read_csv(path, numeric_only=False):
    try:
        df = pd.read_csv(path)
        if numeric_only:
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="ignore")
        return df
    except Exception as e:
        print(f"[SKIP] Could not read {path}: {e}")
        return None

def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def to_num(x):
    try:
        return float(x)
    except:
        return np.nan

def sector_of_species(s):
    s = s.strip().lower()
    if s in ["u","c","t"]: return "up"
    if s in ["d","s","b"]: return "down"
    if s in ["e","mu","tau","μ","tauon","electron","muon"]: return "leptons"
    if s.startswith("nu"): return "neutrinos"
    if s in ["w","z","h","photon","gluon"]: return "bosons"
    return "unknown"

# ---------- T1: Band-slope universality ----------

def test_band_slope_universality(sector_slopes_csv, outdir, n_boot=10000, seed=42):
    """
    Expect CSV with columns: sector, alpha, beta, R2, n
    Bootstraps CIs for alpha per sector; compares |α_ℓ-α_d|, etc.
    """
    if not sector_slopes_csv or not os.path.exists(sector_slopes_csv):
        print("[T1] [SKIP] sector_slopes.csv not provided.")
        return

    df = pd.read_csv(sector_slopes_csv)
    if "sector" not in df or "alpha" not in df:
        print("[T1] [SKIP] sector_slopes.csv missing required columns.")
        return

    rng = np.random.default_rng(seed)
    sectors = ["leptons","down","up"]
    rows = []
    for s in sectors:
        alphas = df.loc[df["sector"].str.lower()==s, "alpha"].astype(float).values
        if len(alphas)==0:
            continue
        boots = []
        for _ in range(n_boot):
            sample = rng.choice(alphas, size=len(alphas), replace=True)
            boots.append(np.median(sample))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"sector":s, "alpha_med":float(np.median(alphas)), "alpha_ci_low":float(lo), "alpha_ci_high":float(hi)})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(outdir, "T1_band_slope_universality.csv"), index=False)
    print("[T1] Wrote: T1_band_slope_universality.csv")

    # pairwise differences
    if len(out)>=2:
        pairs = []
        for i in range(len(out)):
            for j in range(i+1,len(out)):
                a_i = out.iloc[i]; a_j = out.iloc[j]
                pairs.append({
                    "pair": f"{a_i['sector']}-{a_j['sector']}",
                    "diff_med": float(abs(a_i["alpha_med"]-a_j["alpha_med"]))
                })
        pd.DataFrame(pairs).to_csv(os.path.join(outdir,"T1_alpha_pairwise_diffs.csv"), index=False)

# ---------- T2: Charge–curvature sign rule (Hessian / eigenvectors) ----------

def finite_diffs(ax, ay, z):
    """Centered diffs on regular grid arrays (meshgrid style). Return zx, zy, zxx, zyy, zxy."""
    # assume ax, ay are monotone 1D; z shaped (ny,nx)
    dax = np.gradient(ax)
    day = np.gradient(ay)
    zx  = np.gradient(z, axis=1) / dax[np.newaxis,:]
    zy  = np.gradient(z, axis=0) / day[:,np.newaxis]
    zxx = np.gradient(zx, axis=1) / dax[np.newaxis,:]
    zyy = np.gradient(zy, axis=0) / day[:,np.newaxis]
    zxy = np.gradient(zx, axis=0) / day[:,np.newaxis]
    return zx, zy, zxx, zyy, zxy

def grid_group_from_csv(path):
    """
    Reads a grid CSV with columns [ax, ay, z] or [ax, ay, z_pred].
    Returns unique sorted ax, ay and z array reshaped (ny,nx).
    """
    df = pd.read_csv(path)
    zcol = "z"
    if zcol not in df.columns and "z_pred" in df.columns:
        zcol = "z_pred"
    if not all(c in df.columns for c in ["ax","ay",zcol]):
        return None

    ax_vals = np.sort(df["ax"].unique())
    ay_vals = np.sort(df["ay"].unique())
    nx, ny = len(ax_vals), len(ay_vals)
    Z = np.full((ny, nx), np.nan)
    # build fast index
    ax_idx = {v:i for i,v in enumerate(ax_vals)}
    ay_idx = {v:i for i,v in enumerate(ay_vals)}
    for _,r in df.iterrows():
        i = ay_idx[r["ay"]]; j = ax_idx[r["ax"]]
        Z[i,j] = r[zcol]
    return ax_vals, ay_vals, Z

def species_from_path(p):
    base = os.path.basename(os.path.dirname(p))
    # try to parse species from parent folder; fallback to filename
    for key in ["_refined_","_mirror_","_push_"]:
        base = base.replace(key,"_")
    tokens = base.replace("boson_","").split("_")
    return tokens[-1] if tokens else base

def charge_for_species(sp):
    sp = sp.lower()
    if sp in ["e","mu","tau","electron","muon","tauon"]: return -1
    if sp in ["u","c","t"]: return +2/3
    if sp in ["d","s","b"]: return -1/3
    if sp in ["w+","w"]: return +1 # (sign ambiguous for W±; use +1 as proxy)
    if sp in ["z","h","photon","gluon"]: return 0
    if sp.startswith("nu"): return 0
    return np.nan

def test_charge_curvature(grid_glob, outdir):
    """
    For each grid, compute Hessian det sign and principal directions.
    Correlate sgn(det H) with electric charge Q (if species recognizable from path).
    """
    if not grid_glob:
        print("[T2] [SKIP] --grid_glob not given.")
        return

    files = sorted(glob.glob(grid_glob, recursive=True))
    if not files:
        print("[T2] [SKIP] glob matched nothing.")
        return

    rows = []
    for f in files:
        gg = grid_group_from_csv(f)
        if gg is None:
            continue
        ax, ay, Z = gg
        if np.isnan(Z).any():
            continue
        try:
            zx, zy, zxx, zyy, zxy = finite_diffs(ax, ay, Z)
            # evaluate at center point
            i = len(ay)//2; j = len(ax)//2
            H = np.array([[zxx[i,j], zxy[i,j]],[zxy[i,j], zyy[i,j]]])
            detH = np.linalg.det(H)
            w, v = np.linalg.eigh(H)
            # angle of max-curv eigenvector (principal)
            vec = v[:, np.argmax(w)]
            angle = math.degrees(math.atan2(vec[0], vec[1]))  # vs ay axis
            sp = species_from_path(f)
            Q  = charge_for_species(sp)
            rows.append({
                "file": f, "species_guess": sp, "Q": Q,
                "detH": float(detH), "sgn_detH": int(np.sign(detH)),
                "lambda_min": float(np.min(w)), "lambda_max": float(np.max(w)),
                "principal_angle_deg": float(angle)
            })
        except Exception as e:
            print(f"[T2] [WARN] {f}: {e}")

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(outdir, "T2_charge_curvature_sign.csv"), index=False)
        # small summary: mean sgn_detH per charge
        summ = out.groupby("Q")["sgn_detH"].mean().reset_index().rename(columns={"sgn_detH":"mean_sgn_detH"})
        summ.to_csv(os.path.join(outdir, "T2_charge_curvature_sign_summary.csv"), index=False)
        print("[T2] Wrote: T2_charge_curvature_sign*.csv")
    else:
        print("[T2] [SKIP] No usable grids parsed.")

# ---------- T3: Winding ladders ----------

def test_winding_ladders(winding_csv, outdir):
    """
    Uses winding_rationals.csv (species, ay_over_ax, p, q, p_over_q, abs_err)
    Checks monotone ladders by generation: (p+q) and mediants.
    """
    if not winding_csv or not os.path.exists(winding_csv):
        print("[T3] [SKIP] winding_rationals.csv not provided.")
        return

    df = pd.read_csv(winding_csv)
    if "species" not in df or "p" not in df or "q" not in df:
        print("[T3] [SKIP] winding CSV missing columns.")
        return

    df["pq_sum"] = df["p"] + df["q"]
    # rough generation index order via known species
    order = ["u","c","t","d","s","b","e","mu","tau","nu1","nu2","nu3","W","Z","H"]
    df["gen_order"] = df["species"].apply(lambda s: order.index(s) if s in order else 999)

    keep = df[["species","p","q","pq_sum","gen_order"]].sort_values("gen_order")
    keep.to_csv(os.path.join(outdir, "T3_winding_ladders.csv"), index=False)
    print("[T3] Wrote: T3_winding_ladders.csv")

# ---------- T4: Mirror-centroid conservation ----------

def test_mirror_centroids(locked_csv, outdir):
    """
    Expect a CSV with at least: species, ax, ay (and optionally sector).
    Compute fermion and boson centroids; if multiple runs (LOAO), groupby 'run_id' if present.
    """
    if not locked_csv or not os.path.exists(locked_csv):
        print("[T4] [SKIP] locked_csv not provided.")
        return

    df = pd.read_csv(locked_csv)
    if not set(["species","ax","ay"]).issubset(df.columns):
        print("[T4] [SKIP] locked_csv missing species/ax/ay.")
        return

    if "sector" not in df.columns:
        df["sector"] = df["species"].apply(sector_of_species)

    group_field = "run_id" if "run_id" in df.columns else None
    groups = [("all", df)] if group_field is None else df.groupby("run_id")

    rows = []
    for key, g in groups:
        ferm = g[g["sector"].isin(["up","down","leptons","neutrinos"])]
        bos  = g[g["sector"].isin(["bosons"])]
        if len(ferm)==0 or len(bos)==0:
            continue
        fc = ferm[["ax","ay"]].mean().values
        bc = bos[["ax","ay"]].mean().values
        d  = float(np.linalg.norm(fc-bc))
        rows.append({
            "group": key,
            "fermion_cx": float(fc[0]), "fermion_cy": float(fc[1]),
            "boson_cx": float(bc[0]), "boson_cy": float(bc[1]),
            "centroid_distance": d
        })

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(outdir,"T4_mirror_centroids.csv"), index=False)
        print("[T4] Wrote: T4_mirror_centroids.csv")
    else:
        print("[T4] [SKIP] not enough species to form both centroids.")

# ---------- T5: CKM triangles via MDS ----------

def classical_mds(D, dim=2, eps=1e-12):
    """Classical MDS from distance matrix D (nxn) → coordinates (n,dim)."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n,n))/n
    B = -0.5 * J @ (D**2) @ J
    w, v = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w  = w[idx]
    v  = v[:,idx]
    w[w<eps] = eps
    L = np.diag(np.sqrt(w[:dim]))
    X = v[:,:dim] @ L
    return X

def triangle_area(pts):
    """Area of triangle from 3 points in 2D (rows)."""
    a,b,c = pts
    return 0.5*abs(np.cross(b-a, c-a))

def test_ckm_jarlskog(ckm_dist_csv, outdir):
    """
    Input: 3x3 distance matrix between (rows: up u,c,t; cols: down d,s,b).
    We embed rows and cols separately via MDS on pseudo-distance matrices and form
    areas Au, Ad; proxy Jgeo ∝ Au * Ad.
    """
    if not ckm_dist_csv or not os.path.exists(ckm_dist_csv):
        print("[T5] [SKIP] ckm_distance_matrix_numeric.csv not provided.")
        return

    df = pd.read_csv(ckm_dist_csv, index_col=0)
    M = df.values.astype(float)
    if M.shape != (3,3):
        print("[T5] [SKIP] CKM distance must be 3x3.")
        return

    # Build symmetric pseudo-dist for up sector (rows) and down sector (cols)
    # Use row-wise distances: D_ij = ||row_i - row_j||
    Dr = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            Dr[i,j] = np.linalg.norm(M[i,:]-M[j,:])

    Dc = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            Dc[i,j] = np.linalg.norm(M[:,i]-M[:,j])

    Xu = classical_mds(Dr, dim=2)  # u,c,t
    Xd = classical_mds(Dc, dim=2)  # d,s,b

    Au = triangle_area(Xu)
    Ad = triangle_area(Xd)
    Jgeo = Au*Ad

    out = pd.DataFrame([{"Au":float(Au),"Ad":float(Ad),"Jgeo_proxy":float(Jgeo)}])
    out.to_csv(os.path.join(outdir, "T5_ckm_jarlskog_proxy.csv"), index=False)
    print("[T5] Wrote: T5_ckm_jarlskog_proxy.csv")

    # quick plots
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].scatter(Xu[:,0],Xu[:,1]); ax[0].set_title("Up (u,c,t) MDS")
    for i,n in enumerate(["u","c","t"]): ax[0].annotate(n,(Xu[i,0],Xu[i,1]))
    ax[1].scatter(Xd[:,0],Xd[:,1]); ax[1].set_title("Down (d,s,b) MDS")
    for i,n in enumerate(["d","s","b"]): ax[1].annotate(n,(Xd[i,0],Xd[i,1]))
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "T5_ckm_mds.png"), dpi=160)
    plt.close(fig)

# ---------- T6: Cycle-sum quasi conservation ----------

def test_cycle_sum(latent_grid_csv, locked_csv, outdir, n_steps=25):
    """
    Along fixed ax (or ay) slices passing through family neighborhoods,
    check variation of sum(z) over species in that family.
    - latent grid CSV: [ax, ay, z_pred] (or z)
    - locked CSV: to identify slice centers around families (species, ax, ay)
    """
    if not latent_grid_csv or not os.path.exists(latent_grid_csv):
        print("[T6] [SKIP] latent grid CSV missing.")
        return
    dfL = pd.read_csv(latent_grid_csv)
    zcol = "z_pred" if "z_pred" in dfL.columns else ("z" if "z" in dfL.columns else None)
    if zcol is None or not set(["ax","ay"]).issubset(dfL.columns):
        print("[T6] [SKIP] latent grid missing ax/ay/z.")
        return

    if not locked_csv or not os.path.exists(locked_csv):
        print("[T6] [SKIP] need locked_csv to get family centers.")
        return

    dfP = pd.read_csv(locked_csv)
    if not set(["species","ax","ay"]).issubset(dfP.columns):
        print("[T6] [SKIP] locked_csv missing species/ax/ay.")
        return

    # family groups
    families = {
        "up_quarks": ["u","c","t"],
        "down_quarks": ["d","s","b"],
        "leptons": ["e","mu","tau"],
        "neutrinos": ["nu1","nu2","nu3"]
    }

    results = []
    # helper: nearest points along a vertical slice through mean(ax) of the family
    for fam, sp_list in families.items():
        G = dfP[dfP["species"].isin(sp_list)]
        if len(G) < 3:
            continue
        ax0 = G["ax"].mean()
        # take n_steps closest distinct ay rows near each species ay and sum z across the nearest ax column
        # pick the column in latent grid closest to ax0
        ax_vals = np.sort(dfL["ax"].unique())
        j = int(np.argmin(np.abs(ax_vals-ax0)))
        # build slice
        slice_df = dfL[dfL["ax"]==ax_vals[j]].sort_values("ay")
        ys = slice_df["ay"].values
        zs = slice_df[zcol].values
        # sum z at species ay indices (nearest)
        idxs = []
        for _,r in G.iterrows():
            k = int(np.argmin(np.abs(ys - r["ay"])))
            idxs.append(k)
        idxs = sorted(set(idxs))
        zsum = zs[idxs].sum()
        # local window variation around each idx
        window = max(1, len(ys)//n_steps)
        zsum_series = []
        for shift in range(-window, window+1):
            kshift = [min(max(0, i+shift), len(ys)-1) for i in idxs]
            zsum_series.append(zs[kshift].sum())
        zsum_series = np.array(zsum_series)
        rel_var = (zsum_series.max()-zsum_series.min())/max(1e-9, abs(zsum))
        results.append({"family":fam, "ax_slice":float(ax_vals[j]),
                        "zsum_base":float(zsum), "rel_var_window":float(rel_var)})

    if results:
        out = pd.DataFrame(results)
        out.to_csv(os.path.join(outdir,"T6_cycle_sum_quasi_conservation.csv"), index=False)
        print("[T6] Wrote: T6_cycle_sum_quasi_conservation.csv")
    else:
        print("[T6] [SKIP] not enough data to form family sums.")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--winding", default=None, help="winding_rationals.csv")
    ap.add_argument("--sector_slopes", default=None, help="sector_slopes.csv")
    ap.add_argument("--locked_csv", default=None, help="all_particles_locked.csv (or similar)")
    ap.add_argument("--ckm_dist", default=None, help="ckm_distance_matrix_numeric.csv (3x3)")
    ap.add_argument("--latent_grid", default=None, help="latent_z_merged2.csv (or similar)")
    ap.add_argument("--grid_glob", default=None, help="glob for per-species grids **/grid_ax_ay_z.csv")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap draws (T1)")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # T1
    test_band_slope_universality(args.sector_slopes, args.outdir, n_boot=args.boot)

    # T2
    test_charge_curvature(args.grid_glob, args.outdir)

    # T3
    test_winding_ladders(args.winding, args.outdir)

    # T4
    test_mirror_centroids(args.locked_csv, args.outdir)

    # T5
    test_ckm_jarlskog(args.ckm_dist, args.outdir)

    # T6
    test_cycle_sum(args.latent_grid, args.locked_csv, args.outdir)

    # Provenance
    prov = {
        "cmd": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "inputs": {
            "winding": args.winding,
            "sector_slopes": args.sector_slopes,
            "locked_csv": args.locked_csv,
            "ckm_dist": args.ckm_dist,
            "latent_grid": args.latent_grid,
            "grid_glob": args.grid_glob
        }
    }
    with open(os.path.join(args.outdir, "relationships_extended_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print(f"[DONE] Extended audit complete. See: {args.outdir}")

if __name__ == "__main__":
    main()