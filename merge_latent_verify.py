# merge_latent_verify.py  (handles missing 'z' by computing from kbd.json)
import argparse, json
from pathlib import Path
import pandas as pd

def read_mean_rz(csv_path: str):
    df = pd.read_csv(csv_path)
    # r is always present
    r = pd.to_numeric(df["r"], errors="coerce").mean()
    # z may or may not be present
    z = None
    for zname in ("z","z_latent","latent_z"):
        if zname in df.columns:
            z = pd.to_numeric(df[zname], errors="coerce").mean()
            break
    return float(r), (None if z is None or pd.isna(z) else float(z))

def parse_add_arg(s: str):
    # Format: path,ax,ay  (commas separate the last two numbers)
    parts = s.rsplit(",", 2)
    if len(parts) != 3:
        raise ValueError(f"--add must look like 'C:\\path\\to\\cycle_rows_r.csv,2.520,0.733' (got: {s})")
    path, ax, ay = parts[0].strip(), float(parts[1]), float(parts[2])
    return path, ax, ay

def main():
    ap = argparse.ArgumentParser(description="Merge verify cycle_rows_r.csv into latent_z CSV")
    ap.add_argument("--base", required=True, help="Existing latent_z CSV (columns: src,ax,ay,r,z)")
    ap.add_argument("--kbd",  required=True, help="kbd.json from the geometry fit (must contain r_mean and r_std)")
    ap.add_argument("--add", action="append", default=[], help="Triplet 'path,ax,ay' (repeatable)")
    ap.add_argument("--out", required=True, help="Output merged latent CSV")
    args = ap.parse_args()

    base = Path(args.base)
    if not base.exists(): raise SystemExit(f"[ERR] base not found: {base}")

    kbd_path = Path(args.kbd)
    if not kbd_path.exists(): raise SystemExit(f"[ERR] kbd.json not found: {kbd_path}")
    kbd = json.loads(kbd_path.read_text(encoding="utf-8"))
    # Expect r_mean / r_std; fall back to compute from base if absent
    r_mean = kbd.get("r_mean", None)
    r_std  = kbd.get("r_std", None)

    df_base = pd.read_csv(base)
    for col in ["ax","ay","r","z"]:
        if col in df_base.columns:
            df_base[col] = pd.to_numeric(df_base[col], errors="coerce")

    # If r_mean/r_std missing in kbd, estimate from base rows that have z:
    if (r_mean is None or r_std is None) and {"r","z"}.issubset(df_base.columns):
        # invert z = -(r - r_mean)/r_std  =>  r = r_mean - r_std*z
        # Use linear regression r ~ a + b*(-z) to estimate (a=r_mean, b=r_std)
        tmp = df_base.dropna(subset=["r","z"])
        if len(tmp) >= 3:
            import numpy as np
            X = -np.asarray(tmp["z"], float)
            Y =  np.asarray(tmp["r"], float)
            A = np.column_stack([np.ones_like(X), X])
            coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
            r_mean, r_std = float(coef[0]), float(coef[1])

    if r_mean is None or r_std in (None, 0.0):
        raise SystemExit("[ERR] Need r_mean and r_std to compute z from r (not found in kbd.json and could not infer).")

    rows = []
    for item in args.add:
        path, ax, ay = parse_add_arg(item)
        r, z = read_mean_rz(path)
        if z is None:
            z = -(r - r_mean) / r_std
        rows.append({"src": path, "ax": ax, "ay": ay, "r": r, "z": z})

    df_new = pd.DataFrame(rows, columns=["src","ax","ay","r","z"])
    df_merged = pd.concat([df_base, df_new], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=["ax","ay"], keep="last")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(out, index=False)
    print(f"[OK] wrote {out}  (rows: {len(df_merged)})  using r_mean={r_mean:.9f} r_std={r_std:.9f}")

if __name__ == "__main__":
    main()