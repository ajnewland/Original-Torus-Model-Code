import os, glob, pandas as pd

# EDIT THIS to your "New Results" root
ROOT = r"C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results"
OUT  = r"C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\all_cycle_rows_merged.csv"

rows = []
paths = glob.glob(os.path.join(ROOT, "**", "cycle_rows_r.csv"), recursive=True)

for p in paths:
    try:
        df = pd.read_csv(p)
    except Exception:
        continue

    # Prefer z_pred if present, else z; skip if neither exists
    zcol = "z_pred" if "z_pred" in df.columns else ("z" if "z" in df.columns else None)
    if zcol is None:
        continue
    if "ax" not in df.columns or "ay" not in df.columns:
        continue

    use = df[["ax","ay", zcol]].copy()
    use.rename(columns={zcol: "z"}, inplace=True)
    use["src"] = p
    rows.append(use)

if not rows:
    raise SystemExit("No cycle_rows_r.csv files found under ROOT. Check ROOT path.")

merged = pd.concat(rows, ignore_index=True).dropna(subset=["ax","ay","z"])
# Drop exact duplicates (sometimes the same sweep gets copied around)
merged = merged.drop_duplicates()

merged.to_csv(OUT, index=False)
print(f"Saved merged latent CSV with {len(merged)} rows to:\n  {OUT}")
print("Example rows:")
print(merged.head(5).to_string(index=False))