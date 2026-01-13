# relationships_master_audit_v2.py
import argparse, os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def resolve_one(pattern_or_path):
    # Accept exact path or a glob (Windows CMD doesn't expand *, Python will)
    matches = glob.glob(pattern_or_path)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # pick the newest by mtime
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return matches[0]
    return None

def load_named(name, path_or_glob):
    if not path_or_glob:
        print(f"[INFO] {name}: not provided.")
        return None, None
    p = resolve_one(path_or_glob)
    if p is None:
        print(f"[WARN] {name}: no file matched: {path_or_glob}")
        return None, None
    try:
        df = pd.read_csv(p)
        print(f"[OK] {name}: loaded {p} ({df.shape[0]}×{df.shape[1]})")
        return df, p
    except Exception as e:
        print(f"[WARN] {name}: failed to read {p}: {e}")
        return None, None

def save_fig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Master relationships audit (paths or globs).")
    ap.add_argument("--winding", help="winding_rationals.csv path or glob")
    ap.add_argument("--ckm_dist", help="ckm_distance_matrix.csv path or glob")
    ap.add_argument("--ckm_inv", help="ckm_inverse_distance_proxy.csv path or glob")
    ap.add_argument("--koide", help="koide_z_shift_invariant.csv path or glob")
    ap.add_argument("--slopes", help="sector_slopes.csv path or glob")
    ap.add_argument("--noanchors", help="noanchors_ordering.csv path or glob")
    ap.add_argument("--outdir", default="audit_master_out", help="output directory")
    args = ap.parse_args()

    od = args.outdir
    os.makedirs(od, exist_ok=True)

    # 1) Winding rationals
    df_wind, p_wind = load_named("winding_rationals", args.winding)
    if df_wind is not None:
        df_wind.to_csv(os.path.join(od, "winding_rationals_checked.csv"), index=False)

    # 2) CKM distance
    df_ckm_dist, p_ckm_dist = load_named("ckm_distance_matrix", args.ckm_dist)
    if df_ckm_dist is not None:
        plt.imshow(df_ckm_dist.values, aspect="auto")
        plt.title("CKM distance matrix (geom)")
        plt.colorbar(label="distance")
        save_fig(os.path.join(od, "ckm_distance_heatmap.png"))
        df_ckm_dist.to_csv(os.path.join(od, "ckm_distance_matrix_checked.csv"), index=False)

    # 3) CKM inverse-distance proxy
    df_ckm_inv, p_ckm_inv = load_named("ckm_inverse_distance_proxy", args.ckm_inv)
    if df_ckm_inv is not None:
        plt.imshow(df_ckm_inv.values, aspect="auto")
        plt.title("CKM inverse-distance proxy |V|")
        plt.colorbar(label="proxy")
        save_fig(os.path.join(od, "ckm_inverse_heatmap.png"))
        df_ckm_inv.to_csv(os.path.join(od, "ckm_inverse_distance_proxy_checked.csv"), index=False)

    # 4) Koide-like
    df_koide, p_koide = load_named("koide_z_shift_invariant", args.koide)
    if df_koide is not None:
        df_koide.to_csv(os.path.join(od, "koide_z_shift_invariant_checked.csv"), index=False)

    # 5) Sector slopes
    df_slopes, p_slopes = load_named("sector_slopes", args.slopes)
    if df_slopes is not None:
        # simple bar of alpha by sector
        if {"sector", "alpha"}.issubset(df_slopes.columns):
            plt.figure()
            df_slopes.sort_values("sector").plot(kind="bar", x="sector", y="alpha", legend=False)
            plt.ylabel("alpha (slope of log m vs z)")
            plt.title("Sector slopes (alpha)")
            save_fig(os.path.join(od, "sector_slopes_alpha.png"))
        df_slopes.to_csv(os.path.join(od, "sector_slopes_checked.csv"), index=False)

    # 6) No-anchors ordering
    df_noa, p_noa = load_named("noanchors_ordering", args.noanchors)
    if df_noa is not None:
        df_noa.to_csv(os.path.join(od, "noanchors_ordering_checked.csv"), index=False)

    # Master summary
    rows = []
    for nm, df, p in [
        ("winding_rationals", df_wind, p_wind),
        ("ckm_distance_matrix", df_ckm_dist, p_ckm_dist),
        ("ckm_inverse_distance_proxy", df_ckm_inv, p_ckm_inv),
        ("koide_z_shift_invariant", df_koide, p_koide),
        ("sector_slopes", df_slopes, p_slopes),
        ("noanchors_ordering", df_noa, p_noa),
    ]:
        if df is None:
            rows.append({"dataset": nm, "status": "missing", "path": ""})
        else:
            rows.append({"dataset": nm, "status": "ok", "path": p, "rows": df.shape[0], "cols": df.shape[1]})
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(os.path.join(od, "MASTER_SUMMARY.csv"), index=False)

    with open(os.path.join(od, "MASTER_SUMMARY.md"), "w") as f:
        f.write("# Master Relationships Audit\n\n")
        for r in rows:
            if r["status"] == "ok":
                f.write(f"- {r['dataset']}: OK — {r['rows']}×{r['cols']} from `{r['path']}`\n")
            else:
                f.write(f"- {r['dataset']}: MISSING\n")

    print(f"\n[DONE] Audit complete. See: {od}")

if __name__ == "__main__":
    main()