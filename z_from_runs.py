import argparse, pandas as pd, numpy as np
from pathlib import Path

def fit_z_surface(latent_csv):
    # expects columns: ax, ay, z  (your latent_z_merged2.csv has these)
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
        return (coef[0]
                + coef[1]*ax + coef[2]*ay
                + coef[3]*ax*ax + coef[4]*ax*ay + coef[5]*ay*ay)
    return z_fun

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True,
                    help="path to latent_z_merged2.csv (with ax,ay,z samples)")
    ap.add_argument("--in_csv", required=True,
                    help="your sweep CSV (must contain 'ay', and optionally 'ax')")
    ap.add_argument("--out_csv", required=True,
                    help="where to write the same table + z_pred")
    ap.add_argument("--ax_const", type=float, default=None,
                    help="use this ax for all rows if the input file has no 'ax' column")
    args = ap.parse_args()

    z_fun = fit_z_surface(args.latent)

    df = pd.read_csv(args.in_csv)
    if "ay" not in df.columns:
        raise ValueError("input CSV must contain an 'ay' column")

    if "ax" in df.columns:
        ax = df["ax"].values
    else:
        if args.ax_const is None:
            raise ValueError("No 'ax' column found. Provide --ax_const=<value>.")
        ax = np.full(len(df), args.ax_const)

    ay = df["ay"].values
    df["z_pred"] = z_fun(ax, ay)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv}")
    print("First few z_pred:", df["z_pred"].head().to_list())

if __name__ == "__main__":
    main()