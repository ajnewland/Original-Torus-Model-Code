import argparse, pandas as pd, numpy as np
from pathlib import Path
import glob

def fit_z_surface(latent_csv):
    df = pd.read_csv(latent_csv)
    for col in ("ax","ay","z"):
        if col not in df.columns:
            raise ValueError(f"latent CSV must have columns ax, ay, z (missing {col})")
    X = np.column_stack([
        np.ones(len(df)),
        df["ax"].values,
        df["ay"].values,
        df["ax"].values**2,
        (df["ax"]*df["ay"]).values,
        df["ay"].values**2,
    ])
    y = df["z"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    def z_fun(ax, ay):
        ax = np.asarray(ax); ay = np.asarray(ay)
        return (coef[0] + coef[1]*ax + coef[2]*ay
                + coef[3]*ax*ax + coef[4]*ax*ay + coef[5]*ay*ay)
    return z_fun

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True, help="latent_z_merged2.csv (with ax,ay,z)")
    # choose ONE of the following:
    ap.add_argument("--inputs", nargs="+", default=None,
                    help="one or more CSVs to process")
    ap.add_argument("--glob", default=None,
                    help="glob pattern for many CSVs, e.g. C:\\path\\*\\cycle_rows_r.csv")
    ap.add_argument("--ax_const", type=float, default=None,
                    help="use this ax for all rows if input files lack an 'ax' column")
    ap.add_argument("--out_dir", required=True,
                    help="directory to write *_with_z.csv files")
    ap.add_argument("--merge_out", default=None,
                    help="optional single CSV with all rows merged")
    args = ap.parse_args()

    # collect files
    files = []
    if args.inputs: files.extend(args.inputs)
    if args.glob:   files.extend(glob.glob(args.glob))
    if not files:
        raise SystemExit("Provide --inputs <file1 file2 ...> or --glob <pattern>")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    z_fun = fit_z_surface(args.latent)

    merged_rows = []
    for f in files:
        fpath = Path(f)
        if not fpath.exists():
            print(f"[skip] not found: {f}")
            continue
        df = pd.read_csv(fpath)

        if "ay" not in df.columns:
            print(f"[skip] {f} has no 'ay' column"); continue

        if "ax" in df.columns:
            ax = df["ax"].values
        else:
            if args.ax_const is None:
                print(f"[skip] {f} has no 'ax' column and no --ax_const given"); continue
            ax = np.full(len(df), args.ax_const)

        ay = df["ay"].values
        df["z_pred"] = z_fun(ax, ay)

        out_file = out_dir / (fpath.stem + "_with_z.csv")
        df.to_csv(out_file, index=False)
        print(f"[OK] wrote {out_file}")

        # keep a few useful identifiers for merge
        df["_source"] = str(fpath)
        merged_rows.append(df)

    if args.merge_out and merged_rows:
        merged = pd.concat(merged_rows, ignore_index=True)
        Path(args.merge_out).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.merge_out, index=False)
        print(f"[OK] wrote merged table: {args.merge_out}")

if __name__ == "__main__":
    main()