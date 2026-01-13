# relationships_extended_v2.py
# Robust extended relationships audit (Windows-friendly)
# Usage example:
#   python relationships_extended_v2.py ^
#     --outdir "C:\...\grand_audit_plus" ^
#     --winding "C:\...\grand_audit_out\winding_rationals.csv" ^
#     --sector_slopes "C:\...\grand_audit_out\sector_slopes.csv" ^
#     --locked_csv "C:\...\Predicted Masses\all_particles_locked.csv" ^
#     --ckm_dist "C:\...\grand_audit_out\ckm_distance_matrix_numeric.csv" ^
#     --ckm_inv  "C:\...\grand_audit_out\ckm_inverse_distance_proxy_numeric.csv"

import argparse
import os, sys, json, glob, math, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from textwrap import dedent
from scipy.stats import spearmanr


# ------------------------ utils ------------------------

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def savefig_tight(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def read_csv_smart(p):
    df = pd.read_csv(p)
    # strip stray whitespace in all object cols
    objcols = df.select_dtypes(include=["object"]).columns
    if len(objcols):
        for c in objcols:
            df[c] = df[c].astype(str).str.strip()
    return df

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ---------------- coercers / cleaners ------------------

def coerce_sector_slopes(df):
    # sector as lowercase string
    if "sector" in df.columns:
        df["sector"] = df["sector"].astype(str).str.strip().str.lower()
    # numeric columns (optional presence)
    for c in ("alpha", "beta", "R2"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # drop fully empty rows
    return df.dropna(how="all")

def coerce_winding(df):
    # expected cols (flexible names)
    # try to harmonize common variants
    rename_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("species","name"): rename_map[c] = "species"
        if lc in ("ay_over_ax","ay/ax","ratio"): rename_map[c] = "ay_over_ax"
        if lc in ("p","num"): rename_map[c] = "p"
        if lc in ("q","den"): rename_map[c] = "q"
        if lc in ("p_over_q","p/q"): rename_map[c] = "p_over_q"
        if lc in ("abs_err","error","err"): rename_map[c] = "abs_err"
    if rename_map:
        df = df.rename(columns=rename_map)
    # dtypes
    if "species" in df.columns:
        df["species"] = df["species"].astype(str).str.strip()
    for c in ("ay_over_ax","p_over_q","abs_err"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("p","q"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")
    # compute abs_err if missing
    if "abs_err" not in df.columns and {"ay_over_ax","p_over_q"} <= set(df.columns):
        df["abs_err"] = (df["ay_over_ax"] - df["p_over_q"]).abs()
    return df.dropna(how="all")

def coerce_numeric_matrix(df):
    # drop non-numeric first column if it looks like labels (u/c/t etc.)
    # try converting everything to numeric except a possible index col
    def is_all_numeric(s):
        try:
            pd.to_numeric(s, errors="raise")
            return True
        except Exception:
            return False

    if df.shape[1] >= 2:
        # test first column
        if not is_all_numeric(df.iloc[:,0]):
            # keep as labels; numeric on the rest
            M = df.iloc[:,1:].apply(pd.to_numeric, errors="coerce")
            labels = df.iloc[:,0].astype(str).tolist()
            return M, labels
    # else: all numeric
    M = df.apply(pd.to_numeric, errors="coerce")
    return M, None

def coerce_locked(df):
    # Try to find z and mass columns in a forgiving way
    rename = {}
    lcmap = {c: c.lower().strip() for c in df.columns}
    inv = {v:k for k,v in lcmap.items()}
    # z
    zcol = None
    for cand in ("z","z_pred","zval","latent_z"):
        if cand in lcmap.values():
            zcol = inv[cand]; break
    # mass
    mcol = None
    for cand in ("mass","m","pdg_mass","mass_mev","mass_gev","m_pdq","m_pdg"):
        if cand in lcmap.values():
            mcol = inv[cand]; break
    # species
    scol = None
    for cand in ("species","name","label","particle"):
        if cand in lcmap.values():
            scol = inv[cand]; break

    if zcol is None or mcol is None:
        raise ValueError(f"Could not find z/mass columns. Seen: {list(df.columns)}")

    # type coercions
    df[zcol] = pd.to_numeric(df[zcol], errors="coerce")
    df[mcol] = pd.to_numeric(df[mcol], errors="coerce")
    if scol:
        df[scol] = df[scol].astype(str).str.strip()
    return df, zcol, mcol, scol


# ---------------- analyses -----------------------------

def test_band_slope_universality(csv_path, outdir, n_boot=5000, rng_seed=123):
    outdir = ensure_dir(os.path.join(outdir, "slopes_universality"))
    df = read_csv_smart(csv_path)
    df = coerce_sector_slopes(df)

    # Determine sectors from file; fall back to canonical set
    sectors = sorted(df["sector"].dropna().unique().tolist()) if "sector" in df.columns else []
    if not sectors:
        sectors = ["up","down","leptons","bosons","neutrinos"]

    rng = np.random.default_rng(rng_seed)

    rows = []
    for s in sectors:
        mask = (df["sector"] == s) if "sector" in df.columns else np.array([False]*len(df))
        alphas = df.loc[mask, "alpha"].dropna().astype(float).values if "alpha" in df.columns else np.array([])
        if alphas.size == 0:
            continue
        mean_alpha = np.mean(alphas)
        # bootstrap CI
        bs = []
        for _ in range(n_boot):
            bs.append(np.mean(rng.choice(alphas, size=alphas.size, replace=True)))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"sector": s, "alpha_mean": mean_alpha, "alpha_ci_lo": lo, "alpha_ci_hi": hi, "n": alphas.size})

    out_csv = os.path.join(outdir, "alpha_universality_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # bar plot
    if rows:
        X = np.arange(len(rows))
        means = [r["alpha_mean"] for r in rows]
        yerr = np.array([[r["alpha_mean"]-r["alpha_ci_lo"] for r in rows],
                         [r["alpha_ci_hi"]-r["alpha_mean"] for r in rows]])
        plt.figure(figsize=(7,4))
        plt.bar(X, means)
        plt.errorbar(X, means, yerr=yerr, fmt="none", capsize=4)
        plt.xticks(X, [r["sector"] for r in rows], rotation=0)
        plt.ylabel("alpha (slope of log m vs z)")
        plt.title("Band-slope universality (mean ± 95% CI)")
        savefig_tight(os.path.join(outdir, "alpha_universality.png"))

    return out_csv


def plot_ckm(ckm_dist_path, ckm_inv_path, outdir):
    outdir = ensure_dir(os.path.join(outdir, "ckm"))
    summary = {}

    if ckm_dist_path and os.path.exists(ckm_dist_path):
        df = read_csv_smart(ckm_dist_path)
        M, labels = coerce_numeric_matrix(df)
        summary["ckm_dist_shape"] = list(M.shape)
        # heatmap
        plt.figure(figsize=(4,3))
        plt.imshow(M.values, aspect="auto")
        plt.colorbar()
        plt.title("CKM distance matrix (geom)")
        if labels:
            plt.yticks(np.arange(len(labels)), labels)
        savefig_tight(os.path.join(outdir, "ckm_distance_matrix.png"))
        # save numeric copy we used
        M.to_csv(os.path.join(outdir, "ckm_distance_matrix_numeric_used.csv"), index=False)

    if ckm_inv_path and os.path.exists(ckm_inv_path):
        df = read_csv_smart(ckm_inv_path)
        M, labels = coerce_numeric_matrix(df)
        summary["ckm_inv_shape"] = list(M.shape)
        # heatmap
        plt.figure(figsize=(4,3))
        plt.imshow(M.values, aspect="auto")
        plt.colorbar()
        plt.title("Inverse-distance proxy (|V|)")
        if labels:
            plt.yticks(np.arange(len(labels)), labels)
        savefig_tight(os.path.join(outdir, "ckm_inverse_distance_proxy.png"))
        # row-normalize to sum=1
        arr = M.values.astype(float)
        row_sums = arr.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.where(row_sums>0, arr/row_sums, 0.0)
        pd.DataFrame(R).to_csv(os.path.join(outdir, "ckm_inverse_row_normalized.csv"), index=False)
        # row argmax (which down-type pairs best with which up-type)
        argmax_cols = np.argmax(R, axis=1).tolist()
        summary["ckm_row_argmax"] = argmax_cols
        # tiny text summary
        with open(os.path.join(outdir, "ckm_summary.txt"), "w") as f:
            f.write("Row-normalized inverse-distance |V| proxy (rows up; cols down)\n")
            f.write("Row argmax column indices (0-based): " + ", ".join(map(str,argmax_cols)) + "\n")

    write_json(os.path.join(outdir, "ckm_meta.json"), summary)
    return summary


def winding_report(winding_path, outdir):
    outdir = ensure_dir(os.path.join(outdir, "winding"))
    df = read_csv_smart(winding_path)
    df = coerce_winding(df)

    # absolute error stats
    if "abs_err" in df.columns:
        stats = {
            "n": int(df["abs_err"].notna().sum()),
            "mean_abs_err": float(df["abs_err"].mean()),
            "median_abs_err": float(df["abs_err"].median()),
            "min_abs_err": float(df["abs_err"].min()),
            "max_abs_err": float(df["abs_err"].max()),
        }
        write_json(os.path.join(outdir, "winding_stats.json"), stats)

        # histogram plot
        plt.figure(figsize=(5,3))
        vals = df["abs_err"].dropna().values
        bins = max(10, min(60, int(np.sqrt(len(vals)+1))))
        plt.hist(vals, bins=bins)
        plt.xlabel("|ay/ax - p/q|")
        plt.ylabel("count")
        plt.title("Winding rational approximation errors")
        savefig_tight(os.path.join(outdir, "winding_abs_err_hist.png"))

    # write a cleaned CSV for record
    df.to_csv(os.path.join(outdir, "winding_rationals_checked.csv"), index=False)
    return True


def rank_only_geometry_mass(locked_csv_path, outdir):
    outdir = ensure_dir(os.path.join(outdir, "rank_only"))
    df = read_csv_smart(locked_csv_path)
    df, zcol, mcol, scol = coerce_locked(df)

    # Use log mass; require positive
    dff = df[[zcol, mcol] + ([scol] if scol else [])].dropna()
    dff = dff[dff[mcol] > 0].copy()
    dff["logm"] = np.log(dff[mcol].astype(float))
    dff["z"] = dff[zcol].astype(float)

    rho, p = spearmanr(dff["z"].values, dff["logm"].values)

    out = {
        "n": int(len(dff)),
        "spearman_rho": float(rho),
        "p_value": float(p),
        "z_col": zcol,
        "mass_col": mcol,
        "species_col": scol
    }
    write_json(os.path.join(outdir, "rank_only_spearman.json"), out)

    # scatter (rank only visual)
    plt.figure(figsize=(5,4))
    plt.scatter(dff["z"], dff["logm"], s=20)
    plt.xlabel("z (geometry only)")
    plt.ylabel("log(mass)")
    plt.title(f"Rank-only geometry→mass: Spearman ρ={rho:.3f} (n={len(dff)})")
    savefig_tight(os.path.join(outdir, "rank_only_scatter.png"))

    # dump the points we used
    dff.to_csv(os.path.join(outdir, "rank_only_points.csv"), index=False)
    return out


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Extended relationships audit (robust).")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--winding", default=None, help="winding_rationals.csv")
    ap.add_argument("--sector_slopes", default=None, help="sector_slopes.csv")
    ap.add_argument("--locked_csv", default=None, help="all_particles_locked.csv")
    ap.add_argument("--ckm_dist", default=None, help="ckm_distance_matrix*.csv")
    ap.add_argument("--ckm_inv", default=None, help="ckm_inverse_distance_proxy*.csv")
    ap.add_argument("--boot", type=int, default=5000, help="bootstrap resamples for slope CIs")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    meta = {"inputs": {}, "results": {}}

    # record input hashes if present
    for key in ["winding","sector_slopes","locked_csv","ckm_dist","ckm_inv"]:
        p = getattr(args, key)
        if p and os.path.exists(p):
            meta["inputs"][key] = {"path": p, "sha256": sha256_file(p)}
        elif p:
            meta["inputs"][key] = {"path": p, "sha256": None, "note": "missing"}

    # 1) slope universality
    if args.sector_slopes and os.path.exists(args.sector_slopes):
        try:
            out_csv = test_band_slope_universality(args.sector_slopes, args.outdir, n_boot=args.boot)
            meta["results"]["slopes_universality_csv"] = out_csv
        except Exception as e:
            meta["results"]["slopes_universality_error"] = str(e)

    # 2) CKM heatmaps
    if (args.ckm_dist and os.path.exists(args.ckm_dist)) or (args.ckm_inv and os.path.exists(args.ckm_inv)):
        try:
            ckms = plot_ckm(args.ckm_dist, args.ckm_inv, args.outdir)
            meta["results"]["ckm"] = ckms
        except Exception as e:
            meta["results"]["ckm_error"] = str(e)

    # 3) winding rational checks
    if args.winding and os.path.exists(args.winding):
        try:
            winding_report(args.winding, args.outdir)
            meta["results"]["winding_ok"] = True
        except Exception as e:
            meta["results"]["winding_error"] = str(e)

    # 4) rank-only geometry→mass (no anchors)
    if args.locked_csv and os.path.exists(args.locked_csv):
        try:
            ro = rank_only_geometry_mass(args.locked_csv, args.outdir)
            meta["results"]["rank_only"] = ro
        except Exception as e:
            meta["results"]["rank_only_error"] = str(e)

    # write meta
    write_json(os.path.join(args.outdir, "MASTER_META.json"), meta)
    print(f"[DONE] Extended audit complete. See: {args.outdir}")


if __name__ == "__main__":
    main()