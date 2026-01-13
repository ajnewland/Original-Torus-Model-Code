# relationships_master_audit.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None

def main(outdir="audit_master_out"):
    os.makedirs(outdir, exist_ok=True)

    # === 1) Winding rationals ===
    df_wind = load_csv("winding_rationals.csv")
    if df_wind is not None:
        print("\n=== Winding rationals ===")
        print(df_wind.head())
        df_wind.to_csv(os.path.join(outdir, "winding_rationals_checked.csv"), index=False)

    # === 2–3) CKM matrices ===
    df_ckm_dist = load_csv("ckm_distance_matrix.csv")
    df_ckm_inv  = load_csv("ckm_inverse_distance_proxy.csv")
    if df_ckm_dist is not None:
        plt.imshow(df_ckm_dist.values, cmap="viridis")
        plt.colorbar(label="distance")
        plt.title("CKM distance matrix (geom)")
        plt.savefig(os.path.join(outdir, "ckm_distance_heatmap.png"))
        plt.close()
    if df_ckm_inv is not None:
        plt.imshow(df_ckm_inv.values, cmap="plasma")
        plt.colorbar(label="|V| proxy")
        plt.title("CKM inverse-distance proxy")
        plt.savefig(os.path.join(outdir, "ckm_inverse_heatmap.png"))
        plt.close()

    # === 4) Koide-like ===
    df_koide = load_csv("koide_z_shift_invariant.csv")
    if df_koide is not None:
        print("\n=== Koide-like relation ===")
        print(df_koide)
        df_koide.to_csv(os.path.join(outdir, "koide_check.csv"), index=False)

    # === 5–8) Sector slopes ===
    df_slopes = load_csv("sector_slopes.csv")
    if df_slopes is not None:
        print("\n=== Sector slopes ===")
        print(df_slopes)
        df_slopes.to_csv(os.path.join(outdir, "sector_slopes_checked.csv"), index=False)

    # === 9) No-anchors ordering ===
    df_noanc = load_csv("noanchors_ordering.csv")
    if df_noanc is not None:
        print("\n=== No-anchors ordering ===")
        print(df_noanc)
        df_noanc.to_csv(os.path.join(outdir, "noanchors_checked.csv"), index=False)

    # === 10) Master summary ===
    summary_lines = []
    for name, df in [
        ("winding_rationals", df_wind),
        ("ckm_distance_matrix", df_ckm_dist),
        ("ckm_inverse_proxy", df_ckm_inv),
        ("koide_z_shift", df_koide),
        ("sector_slopes", df_slopes),
        ("noanchors_ordering", df_noanc),
    ]:
        if df is not None:
            summary_lines.append(f"- {name}: loaded {df.shape[0]} rows × {df.shape[1]} cols")
        else:
            summary_lines.append(f"- {name}: [missing]")

    with open(os.path.join(outdir, "MASTER_SUMMARY.md"), "w") as f:
        f.write("# Master Relationships Audit\n\n")
        f.write("\n".join(summary_lines))

    print("\n[DONE] Audit complete. See:", outdir)

if __name__ == "__main__":
    main()