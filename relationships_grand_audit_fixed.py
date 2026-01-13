#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grand Audit (robust) — relationships_grand_audit_fixed.py

Covers:
- Winding-number approximants for rho = ay/ax
- Sectoral slopes of log(mass) vs z, universality check
- CKM from distances (inverse-distance + normalization), optional PDG compare
- Koide-like Q_z per family
- Mirror centroid separation (fermions vs bosons)
- Charge-weighted z balance
- No-anchors ordering (Spearman rank z vs logm)
- Boson band width diagnostic
- Optional: grid sweep minima count (dark-sector nulls) from grid_ax_ay_z.csv files

Schema tolerant: auto-detects column names for mass/z/ax/ay, etc.

Author: you
"""

import argparse, os, sys, glob, math, csv, json, textwrap
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd

# ------------------------------- helpers

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def pick_col(cols, candidates, required=False, error_hint=""):
    """Pick first existing column from candidates. Return None if not found and required==False."""
    for c in candidates:
        if c in cols: return c
    if required:
        raise KeyError(f"Required column not found. Tried {candidates}. Seen {list(cols)}. {error_hint}")
    return None

def safe_log(x):
    try:
        xv = float(x)
        if xv <= 0: return np.nan
        return math.log(xv)
    except:
        return np.nan

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.nansum((y_true - y_pred)**2)
    ss_tot = np.nansum((y_true - np.nanmean(y_true))**2)
    return float('nan') if ss_tot == 0 else 1.0 - ss_res/ss_tot

def linear_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan, np.nan, np.nan  # slope, intercept, r2
    m, b = np.polyfit(x[ok], y[ok], deg=1)
    r2 = r2_score(y[ok], m*x[ok] + b)
    return m, b, r2

def species_sector_map():
    # Basic sector tagging
    return {
        # leptons
        "e":"lepton","mu":"lepton","tau":"lepton",
        # neutrinos
        "nu":"neutrino","nu1":"neutrino","nu2":"neutrino","nu3":"neutrino","ve":"neutrino","vmu":"neutrino","vtau":"neutrino",
        # up quarks
        "u":"up","c":"up","t":"up",
        # down quarks
        "d":"down","s":"down","b":"down",
        # bosons
        "W":"boson","Z":"boson","H":"boson","h":"boson",
        # optional
        "photon":"boson","gamma":"boson","g":"boson","gluon":"boson"
    }

def electric_charge_map():
    Q = {
        "e":-1,"mu":-1,"tau":-1,
        "nu":0,"nu1":0,"nu2":0,"nu3":0,"ve":0,"vmu":0,"vtau":0,
        "u":2/3,"c":2/3,"t":2/3,
        "d":-1/3,"s":-1/3,"b":-1/3,
        "W+":+1,"W-":-1,"W":0,   # W label may be neutral placeholder; most CSVs use "W"
        "Z":0,"H":0,"h":0,
        "gamma":0,"photon":0,
    }
    return Q

def canonical_species(s):
    s = str(s).strip()
    return s.replace("ν","nu").replace("v_e","ve").replace("v_mu","vmu").replace("v_tau","vtau")

def compute_rho(df, ax_col, ay_col):
    # Create ay_over_ax if missing
    if "ay_over_ax" not in df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            df["ay_over_ax"] = df[ay_col].astype(float) / df[ax_col].astype(float)
    return df

def best_rational(x, max_den=80):
    try:
        frac = Fraction(x).limit_denominator(max_den)
        return frac.numerator, frac.denominator, float(frac), abs(x - float(frac))
    except Exception:
        return None, None, None, None

def load_locked(locked_csv: Path):
    df = pd.read_csv(locked_csv)
    cols = df.columns

    # species
    sp_col = pick_col(cols, ["species","name","label","id"], required=True)

    # ax, ay
    ax_col = pick_col(cols, ["ax","a_x","alpha_x","ax_coord"], required=True, error_hint="Need torus coordinates.")
    ay_col = pick_col(cols, ["ay","a_y","alpha_y","ay_coord"], required=True, error_hint="Need torus coordinates.")

    # z values
    z_col = pick_col(cols, ["z","z_pred","z_target","z_hat","z_est"], required=True, error_hint="Need z (pred/target).")

    # mass (GeV preferred)
    mass_col = pick_col(cols, ["mass_GeV","m_GeV","mass","massGeV","mgev"])
    logm_col = pick_col(cols, ["logm","log_mass","lnm","log_mass_GeV"])
    if mass_col is None and logm_col is None:
        raise KeyError(f"No mass or log-mass column found in {locked_csv}.")

    # Build canonical columns
    df["species"] = df[sp_col].map(canonical_species)
    df["ax"] = pd.to_numeric(df[ax_col], errors="coerce")
    df["ay"] = pd.to_numeric(df[ay_col], errors="coerce")
    df["z"]  = pd.to_numeric(df[z_col], errors="coerce")

    if mass_col is not None:
        df["mass_GeV"] = pd.to_numeric(df[mass_col], errors="coerce")
        # derive logm
        df["logm"] = np.log(df["mass_GeV"].replace({0:np.nan}))
    else:
        # derive mass from logm (assume log is natural log)
        df["logm"] = pd.to_numeric(df[logm_col], errors="coerce")
        df["mass_GeV"] = np.exp(df["logm"])

    # Drop rows with missing essentials
    df = df.dropna(subset=["species","ax","ay","z","logm"])
    df = compute_rho(df, "ax", "ay")
    return df

def write_csv(path, rows, header):
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def save_json(path, obj):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ------------------------------- audits

def audit_winding(df, outdir: Path, max_den=80):
    rows = []
    for _, r in df.iterrows():
        sp = r["species"]
        rho = float(r["ay_over_ax"])
        p,q,approx_val,err = best_rational(rho, max_den=max_den)
        rows.append([sp, rho, p, q, approx_val, err])
    out = outdir / "winding" / "winding_rationals_auto.csv"
    write_csv(out, rows, ["species","ay_over_ax","p","q","p_over_q","abs_err"])
    return str(out)

def sector_of(species):
    s = species.lower()
    secmap = species_sector_map()
    return secmap.get(s, "unknown")

def is_fermion(species):
    return sector_of(species) in {"lepton","neutrino","up","down"}

def is_boson(species):
    return sector_of(species) == "boson"

def audit_sector_slopes(df, outdir: Path):
    # group by sectors and fit logm ~ alpha * z + beta
    sectors = ["lepton","up","down","neutrino"]
    rows_sum = []
    for sec in sectors:
        d = df[[s==sec for s in df["species"].map(sector_of)]]
        if len(d) >= 2:
            m, b, r2 = linear_fit(d["z"], d["logm"])
            rows_sum.append([sec, m, b, r2, len(d)])
    out = outdir / "slopes_universality" / "alpha_universality_summary.csv"
    write_csv(out, rows_sum, ["sector","alpha","beta","r2","n"])
    return str(out)

def audit_ckm_from_dist(dist_csv: Path, outdir: Path, pdg_csv: Path = None, eps=1e-6):
    # Expect a 3x3 distance matrix with row labels u,c,t and col labels d,s,b
    dist_df = pd.read_csv(dist_csv, index_col=0)
    # inverse + row normalize
    V = dist_df.copy()
    V = 1.0 / (V + eps)
    V = V.div(V.sum(axis=1), axis=0)
    inv_out = outdir / "ckm" / "ckm_inverse_distance_normalized.csv"
    ensure_dir(inv_out.parent)
    V.to_csv(inv_out)

    summary = {"ckm_inverse_out": str(inv_out)}
    if pdg_csv is not None and Path(pdg_csv).exists():
        pdg = pd.read_csv(pdg_csv, index_col=0)
        # align indexes
        try:
            pdg = pdg.loc[V.index, V.columns]
            diff = (V - pdg).abs()
            mae = diff.values.mean()
            rmse = math.sqrt((diff.values**2).mean())
            summary.update({"MAE": float(mae), "RMSE": float(rmse)})
            (outdir / "ckm").mkdir(parents=True, exist_ok=True)
            diff.to_csv(outdir / "ckm" / "ckm_abs_error.csv")
        except Exception as e:
            summary.update({"warning": f"Could not align PDG matrix: {e}"})
    save_json(outdir / "ckm" / "summary.json", summary)
    return summary

def audit_koide_z(df, outdir: Path):
    # Compute Q_z = (sum sqrt(z_i))^2 / (3 sum z_i)
    # Triplets by sector families
    fams = {
        "leptons": ["e","mu","tau"],
        "up": ["u","c","t"],
        "down": ["d","s","b"]
    }
    rows=[]
    for name, members in fams.items():
        dd = df[df["species"].isin(members)]
        if len(dd) == 3:
            z = dd.set_index("species")["z"].reindex(members).values
            if np.all(np.isfinite(z)) and np.all(z>0):
                Q = (np.sum(np.sqrt(z))**2) / (3*np.sum(z))
            else:
                Q = np.nan
            rows.append([name, *members, *z, Q])
        else:
            rows.append([name, *members, *(["NA"]*3), np.nan])
    out = outdir / "koide" / "koide_Qz.csv"
    write_csv(out, rows, ["family","s1","s2","s3","z1","z2","z3","Qz"])
    return str(out)

def audit_centroids(df, outdir: Path):
    ferm = df[[is_fermion(s) for s in df["species"]]]
    bos  = df[[is_boson(s) for s in df["species"]]]
    res = {}
    if len(ferm):
        res["fermion_centroid"] = [float(ferm["ax"].mean()), float(ferm["ay"].mean())]
    if len(bos):
        res["boson_centroid"]   = [float(bos["ax"].mean()), float(bos["ay"].mean())]
    if "fermion_centroid" in res and "boson_centroid" in res:
        fa = np.array(res["fermion_centroid"]); ba = np.array(res["boson_centroid"])
        res["centroid_distance"] = float(np.linalg.norm(fa - ba))
    save_json(outdir / "centroids" / "centroids.json", res)
    return res

def audit_charge_weighted_z(df, outdir: Path):
    Qmap = electric_charge_map()
    df["Q"] = df["species"].map(lambda s: Qmap.get(s, 0.0))
    total = float(np.nansum(df["Q"] * df["z"]))
    save_json(outdir / "charge_balance" / "charge_weighted_z.json", {"sum_Qz": total})
    return total

def audit_spearman_ordering(df, outdir: Path):
    # Spearman between z and logm by sector and overall
    def spearman(x,y):
        # pandas rank corr
        s = pd.Series(x).corr(pd.Series(y), method="spearman")
        return float(s) if pd.notna(s) else np.nan

    res = {}
    overall = spearman(df["z"], df["logm"])
    res["overall"] = overall

    for sec in ["lepton","up","down","neutrino"]:
        d = df[[s==sec for s in df["species"].map(sector_of)]]
        if len(d) >= 2:
            res[sec] = spearman(d["z"], d["logm"])
    save_json(outdir / "ordering" / "spearman_z_logm.json", res)
    return res

def audit_boson_band(df, outdir: Path):
    d = df[df["species"].isin(["W","Z","H","h"])]
    res={}
    if len(d) >= 2:
        zmin, zmax = float(d["z"].min()), float(d["z"].max())
        width = zmax - zmin
        res = {"z_min": zmin, "z_max": zmax, "band_width": width}
    save_json(outdir / "boson_band" / "boson_band_width.json", res)
    return res

def local_minima_from_grid(df_grid, z_col="z"):
    # df_grid with columns ax, ay, z on a rectangular grid
    # Simple 2D local minima finder using neighborhood comparison.
    # Assumes grid sampling is dense and (ax,ay) form a mesh.
    ax_vals = np.sort(df_grid["ax"].unique())
    ay_vals = np.sort(df_grid["ay"].unique())
    ax_idx = {v:i for i,v in enumerate(ax_vals)}
    ay_idx = {v:i for i,v in enumerate(ay_vals)}
    Z = np.full((len(ax_vals), len(ay_vals)), np.nan)
    for _,r in df_grid.iterrows():
        i = ax_idx[r["ax"]]; j = ay_idx[r["ay"]]
        Z[i,j] = r[z_col]

    mins = []
    for i in range(1,len(ax_vals)-1):
        for j in range(1,len(ay_vals)-1):
            val = Z[i,j]
            if not np.isfinite(val): continue
            nbrs = Z[i-1:i+2, j-1:j+2].copy()
            nbrs[1,1] = np.inf
            if val < np.nanmin(nbrs):
                mins.append((ax_vals[i], ay_vals[j], float(val)))
    return mins

def audit_grid_minima(grid_glob, outdir: Path):
    files = glob.glob(grid_glob, recursive=True)
    found = []
    for fp in files:
        try:
            g = pd.read_csv(fp)
            # schema tolerance
            cols = g.columns
            ax_col = pick_col(cols, ["ax","a_x","alpha_x"], required=True)
            ay_col = pick_col(cols, ["ay","a_y","alpha_y"], required=True)
            z_col  = pick_col(cols, ["z","z_pred","z_target","z_hat"], required=True)
            g = g.rename(columns={ax_col:"ax", ay_col:"ay", z_col:"z"})
            mins = local_minima_from_grid(g, z_col="z")
            for (ax,ay,z) in mins:
                found.append([Path(fp).name, ax, ay, z])
        except Exception as e:
            # skip file but record error
            found.append([Path(fp).name, "ERROR", str(e), ""])
    out = outdir / "grid_minima" / "minima_found.csv"
    write_csv(out, found, ["grid_file","ax","ay","z"])
    return str(out)

# ------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Grand Audit (robust) for torus-based SM reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python relationships_grand_audit_fixed.py ^
            --locked-csv "C:\\...\\Predicted Masses\\all_particles_locked.csv" ^
            --outdir "C:\\...\\grand_audit_plus" ^
            --ckm-dist "C:\\...\\ckm_distance_matrix_numeric.csv" ^
            --pdg-ckm  "C:\\...\\pdg_ckm_abs_3x3.csv" ^
            --grid-glob "C:\\...\\Predicted Masses\\**\\grid_ax_ay_z.csv" ^
            --max-den 80
        """)
    )
    ap.add_argument("--locked-csv", required=True, type=str, help="Path to all_particles_locked.csv (any schema; auto-detected)")
    ap.add_argument("--outdir", required=True, type=str, help="Output directory")
    ap.add_argument("--ckm-dist", type=str, default=None, help="3x3 up-vs-down distance CSV")
    ap.add_argument("--pdg-ckm", type=str, default=None, help="Optional PDG |Vij| CSV to compare")
    ap.add_argument("--grid-glob", type=str, default=None, help="Optional glob to grid_ax_ay_z.csv files for minima count")
    ap.add_argument("--max-den", type=int, default=80, help="Max denominator for rational approximants")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Load main locked CSV with schema tolerance
    df = load_locked(Path(args.locked_csv))

    # Audits
    winding_csv = audit_winding(df, outdir, max_den=args.max_den)
    slopes_csv  = audit_sector_slopes(df, outdir)
    koide_csv   = audit_koide_z(df, outdir)
    centroids   = audit_centroids(df, outdir)
    Qz_sum      = audit_charge_weighted_z(df, outdir)
    order_js    = audit_spearman_ordering(df, outdir)
    boson_band  = audit_boson_band(df, outdir)

    ckm_summary = None
    if args.ckm_dist:
        ckm_summary = audit_ckm_from_dist(Path(args.ckm_dist), outdir, Path(args.pdg_ckm) if args.pdg_ckm else None)

    grid_minima_csv = None
    if args.grid_glob:
        grid_minima_csv = audit_grid_minima(args.grid_glob, outdir)

    # Master summary
    summary = {
        "inputs": {
            "locked_csv": {"path": args.locked_csv},
            "ckm_dist": {"path": args.ckm_dist} if args.ckm_dist else None,
            "pdg_ckm": {"path": args.pdg_ckm} if args.pdg_ckm else None,
            "grid_glob": args.grid_glob,
        },
        "outputs": {
            "winding_csv": winding_csv,
            "slopes_universality_csv": slopes_csv,
            "koide_csv": koide_csv,
            "centroids": centroids,
            "sum_Qz": Qz_sum,
            "spearman": order_js,
            "boson_band": boson_band,
            "ckm_summary": ckm_summary,
            "grid_minima_csv": grid_minima_csv
        }
    }
    save_json(outdir / "grand_audit_summary.json", summary)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()