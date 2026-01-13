import argparse, os, sys, textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_df(path, expect=None):
    df = pd.read_csv(path)
    # strip strings and whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def numeric_matrix_from_df(df):
    """
    Return a pure numeric 2D array from a DF that may contain a label
    row/col. Strategy:
      1) Try to convert all cells to numeric (coerce errors to NaN)
      2) Drop fully-NaN rows/cols (likely labels)
      3) If first col or row is non-numeric labels, drop it
      4) Return .values as float array
    """
    df_num = df.apply(pd.to_numeric, errors="coerce")
    # Drop rows/cols that are all NaN (these are labels)
    df_num = df_num.dropna(axis=0, how="all")
    df_num = df_num.dropna(axis=1, how="all")

    # Sometimes a label column remains (e.g. first col is species names)
    # If a column has any NaN left and others don't, try dropping it.
    if df_num.shape[1] > 0:
        # If any column still has NaNs but dropping it resolves, do so.
        bad_cols = [c for c in df_num.columns if df_num[c].isna().any()]
        # Only drop if there is at least one fully numeric column left
        if bad_cols and len(bad_cols) < df_num.shape[1]:
            df_try = df_num.drop(columns=bad_cols, errors="ignore")
            if df_try.shape[1] > 0 and not df_try.isna().any().any():
                df_num = df_try

    # Same idea for rows (just in case a label row sneaks in)
    if df_num.shape[0] > 0 and df_num.isna().any().any():
        bad_rows = [i for i in df_num.index if df_num.loc[i].isna().any()]
        if bad_rows and len(bad_rows) < df_num.shape[0]:
            df_try = df_num.drop(index=bad_rows, errors="ignore")
            if df_try.shape[0] > 0 and not df_try.isna().any().any():
                df_num = df_try

    # Final fallback: fill remaining NaNs with 0 (shouldn't happen for CKM matrices)
    df_num = df_num.fillna(0.0)

    arr = df_num.to_numpy(dtype=float)
    return arr, df_num.index.tolist(), df_num.columns.tolist()

def save_heatmap(arr, out_png, title="", xlabels=None, ylabels=None):
    plt.figure(figsize=(4.8, 4.2), dpi=140)
    im = plt.imshow(arr, aspect="auto")
    plt.title(title)
    if xlabels is not None and len(xlabels) == arr.shape[1]:
        plt.xticks(np.arange(arr.shape[1]), xlabels)
    else:
        plt.xticks(np.arange(arr.shape[1]))
    if ylabels is not None and len(ylabels) == arr.shape[0]:
        plt.yticks(np.arange(arr.shape[0]), ylabels)
    else:
        plt.yticks(np.arange(arr.shape[0]))
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    ap = argparse.ArgumentParser(
        description="Robust relationships audit (v3) – safe numeric loading for CKM/Koide/etc."
    )
    ap.add_argument("--winding", help="winding_rationals.csv")
    ap.add_argument("--ckm_dist", help="ckm_distance_matrix.csv")
    ap.add_argument("--ckm_inv", help="ckm_inverse_distance_proxy.csv")
    ap.add_argument("--koide", help="koide_z_shift_invariant.csv")
    ap.add_argument("--slopes", help="sector_slopes.csv")
    ap.add_argument("--noanchors", help="noanchors_ordering.csv")
    ap.add_argument("--outdir", default="audit_master_out")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    summary_lines = ["# Master Relationships Audit (v3)\n"]

    # 1) Winding rationals table (just copy to outdir)
    if args.winding and os.path.exists(args.winding):
        try:
            df_w = load_df(args.winding)
            df_w.to_csv(os.path.join(args.outdir, "winding_rationals.csv"), index=False)
            summary_lines.append("**Winding rationals:** loaded and copied.")
        except Exception as e:
            summary_lines.append(f"**Winding rationals:** ERROR {e}")
    else:
        summary_lines.append("**Winding rationals:** missing.")

    # 2) CKM distance heatmap
    if args.ckm_dist and os.path.exists(args.ckm_dist):
        try:
            df_cd = load_df(args.ckm_dist)
            arr, rlbl, clbl = numeric_matrix_from_df(df_cd)
            save_heatmap(arr,
                         os.path.join(args.outdir, "ckm_distance_matrix.png"),
                         title="CKM distance matrix (geometry proxy)",
                         xlabels=clbl, ylabels=rlbl)
            # Also write a cleaned numeric CSV
            clean_csv = os.path.join(args.outdir, "ckm_distance_matrix_numeric.csv")
            pd.DataFrame(arr, index=rlbl, columns=clbl).to_csv(clean_csv)
            summary_lines.append(f"**CKM distance:** plotted {arr.shape} and wrote cleaned CSV.")
        except Exception as e:
            summary_lines.append(f"**CKM distance:** ERROR {e}")
    else:
        summary_lines.append("**CKM distance:** missing.")

    # 3) CKM inverse-distance heatmap
    if args.ckm_inv and os.path.exists(args.ckm_inv):
        try:
            df_ci = load_df(args.ckm_inv)
            arr, rlbl, clbl = numeric_matrix_from_df(df_ci)
            save_heatmap(arr,
                         os.path.join(args.outdir, "ckm_inverse_distance_proxy.png"),
                         title="CKM inverse-distance proxy (|V| approx.)",
                         xlabels=clbl, ylabels=rlbl)
            clean_csv = os.path.join(args.outdir, "ckm_inverse_distance_proxy_numeric.csv")
            pd.DataFrame(arr, index=rlbl, columns=clbl).to_csv(clean_csv)
            summary_lines.append(f"**CKM inverse-distance:** plotted {arr.shape} and wrote cleaned CSV.")
        except Exception as e:
            summary_lines.append(f"**CKM inverse-distance:** ERROR {e}")
    else:
        summary_lines.append("**CKM inverse-distance:** missing.")

    # 4) Koide-like z table (just copy to outdir)
    if args.koide and os.path.exists(args.koide):
        try:
            df_k = load_df(args.koide)
            df_k.to_csv(os.path.join(args.outdir, "koide_z_shift_invariant.csv"), index=False)
            summary_lines.append("**Koide-like z:** loaded and copied.")
        except Exception as e:
            summary_lines.append(f"**Koide-like z:** ERROR {e}")
    else:
        summary_lines.append("**Koide-like z:** missing.")

    # 5) Sector slopes (copy + quick text summary)
    if args.slopes and os.path.exists(args.slopes):
        try:
            df_s = load_df(args.slopes)
            df_s.to_csv(os.path.join(args.outdir, "sector_slopes.csv"), index=False)
            # produce a small markdown summary table
            md = df_s.to_markdown(index=False)
            with open(os.path.join(args.outdir, "sector_slopes.md"), "w", encoding="utf-8") as f:
                f.write("# Sector slopes (log m vs z)\n\n")
                f.write(md + "\n")
            summary_lines.append("**Sector slopes:** loaded; summary markdown written.")
        except Exception as e:
            summary_lines.append(f"**Sector slopes:** ERROR {e}")
    else:
        summary_lines.append("**Sector slopes:** missing.")

    # 6) No-anchors ordering (copy)
    if args.noanchors and os.path.exists(args.noanchors):
        try:
            df_n = load_df(args.noanchors)
            df_n.to_csv(os.path.join(args.outdir, "noanchors_ordering.csv"), index=False)
            summary_lines.append("**No-anchors ordering:** loaded and copied.")
        except Exception as e:
            summary_lines.append(f"**No-anchors ordering:** ERROR {e}")
    else:
        summary_lines.append("**No-anchors ordering:** missing.")

    # Write SUMMARY.md
    with open(os.path.join(args.outdir, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("[DONE] Audit complete. See:", args.outdir)

if __name__ == "__main__":
    main()