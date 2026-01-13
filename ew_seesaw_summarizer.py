import os, argparse, json, numpy as np, pandas as pd

def load_combined(path):
    df = pd.read_csv(path)
    # Expected columns in your combined file (from your examples):
    # ay, A, B, cA, cB, r, S_raw, w_star, S_star, err_star
    need = []
    for c in ["ay","A","B","S_raw","w_star","S_star"]:
        if c in df.columns: need.append(c)
    if "ay" not in need or ("A" not in need and "B" not in need):
        raise ValueError(f"Combined file is missing required columns. Saw: {list(df.columns)}")
    return df

def summarize(df):
    out = {}
    # Global stats on S_star and S_raw
    for col in ["S_star","S_raw","A","B","w_star","r","err_star"]:
        if col in df.columns:
            s = df[col].dropna()
            out[col] = {
                "count": int(s.shape[0]),
                "mean": float(s.mean()),
                "std":  float(s.std(ddof=1)) if len(s)>1 else 0.0,
                "min":  float(s.min()) if len(s)>0 else None,
                "max":  float(s.max()) if len(s)>0 else None,
                "q05":  float(s.quantile(0.05)) if len(s)>0 else None,
                "q95":  float(s.quantile(0.95)) if len(s)>0 else None,
            }
    # Plateau closeness to 0.231
    if "S_star" in df.columns:
        target = 0.231
        s = (df["S_star"] - target).abs().dropna()
        out["plateau_0p231"] = {
            "mean_abs_err": float(s.mean()) if len(s)>0 else None,
            "median_abs_err": float(s.median()) if len(s)>0 else None,
            "q95_abs_err": float(s.quantile(0.95)) if len(s)>0 else None
        }
    # Cross-check: where A ~ B (seesaw crossing)
    if "A" in df.columns and "B" in df.columns:
        s = (df["A"] - df["B"]).abs().dropna()
        out["A_minus_B"] = {
            "mean_abs": float(s.mean()) if len(s)>0 else None,
            "median_abs": float(s.median()) if len(s)>0 else None,
            "q05_abs": float(s.quantile(0.05)) if len(s)>0 else None
        }
    return out

def bin_by_ay(df, nbins=20):
    if "ay" not in df.columns: return None
    ay = df["ay"].dropna().values
    if ay.size == 0: return None
    bins = np.linspace(np.min(ay), np.max(ay), nbins+1)
    bx = pd.cut(df["ay"], bins, include_lowest=True)
    agg = df.groupby(bx).agg({
        "ay":"mean",
        "S_star":"mean",
        "S_raw":"mean",
        "A":"mean",
        "B":"mean",
        "w_star":"mean",
        "r":"mean"
    }).reset_index(drop=True)
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", required=True, help="Path to all_cycle_rows_combined.csv")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_combined(args.combined)

    # 1) Global summary
    summary = summarize(df)
    with open(os.path.join(args.outdir, "ew_seesaw_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 2) binned-by-ay trends (for plotting)
    ay_bins = bin_by_ay(df, nbins=24)
    if ay_bins is not None:
        ay_bins.to_csv(os.path.join(args.outdir, "ew_seesaw_by_ay.csv"), index=False)

    # 3) Export lean table for paper figs
    keep = [c for c in ["ay","A","B","S_raw","w_star","S_star","r","err_star"] if c in df.columns]
    df[keep].to_csv(os.path.join(args.outdir, "ew_seesaw_clean.csv"), index=False)

    # Console hints
    print("=== EW seesaw summary ===")
    print(json.dumps(summary, indent=2))
    print("Wrote:", os.path.join(args.outdir, "ew_seesaw_summary.json"))
    if ay_bins is not None:
        print("Wrote:", os.path.join(args.outdir, "ew_seesaw_by_ay.csv"))
    print("Wrote:", os.path.join(args.outdir, "ew_seesaw_clean.csv"))

if __name__ == "__main__":
    main()