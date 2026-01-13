import sys, pandas as pd, json

def load_band(path):
    df = pd.read_csv(path)
    # detect usable column for comparison
    for cand in ["r", "z", "S_star"]:
        if cand in df.columns:
            col_used = cand
            break
    else:
        raise ValueError(f"No usable column found in {path}. Need one of: r, z, S_star")

    return df, col_used

def main():
    if len(sys.argv) != 4:
        print("Usage: python strong_probe.py <ew_band.csv> <other_band.csv> <out.txt>")
        sys.exit(1)

    ew_file, other_file, out_file = sys.argv[1:4]

    ew, col_ew = load_band(ew_file)
    other, col_other = load_band(other_file)

    n = min(len(ew), len(other))
    ew, other = ew.iloc[:n].reset_index(drop=True), other.iloc[:n].reset_index(drop=True)

    ew_vals = ew[col_ew].astype(float)
    other_vals = other[col_other].astype(float)

    # sanity check on z-like column
    z_col_ew = "z" if "z" in ew.columns else ("S_star" if "S_star" in ew.columns else None)
    z_col_other = "z" if "z" in other.columns else ("S_star" if "S_star" in other.columns else None)

    ew_z = ew[z_col_ew].astype(float) if z_col_ew else None
    other_z = other[z_col_other].astype(float) if z_col_other else None

    # stats
    mean_ew = ew_vals.mean()
    mean_other = other_vals.mean()
    delta_means = mean_ew - mean_other

    z_delta = None
    if ew_z is not None and other_z is not None:
        z_delta = ew_z.mean() - other_z.mean()

    # affine fit y ≈ A + B*x
    import numpy as np
    X = np.vstack([np.ones(n), other_vals]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, ew_vals, rcond=None)
    A, B = coeffs
    mse = float(((ew_vals - (A + B*other_vals))**2).mean())

    report = {
        "counts": {"ew": len(ew), "other": len(other), "paired": n},
        "columns_used": {"ew": col_ew, "other": col_other},
        "r_means": {"ew": mean_ew, "other": mean_other, "delta": delta_means},
        "z_means": {
            "ew": float(ew_z.mean()) if ew_z is not None else None,
            "other": float(other_z.mean()) if other_z is not None else None,
            "delta": z_delta,
        },
        "affine_map_other_to_ew": {"A": float(A), "B": float(B), "mse": mse},
    }

    text = []
    text.append("=== Strong↔Other probe report ===")
    text.append(f"EW file     : {ew_file}")
    text.append(f"Other file  : {other_file}")
    text.append(f"Samples (EW / OTHER / paired): {len(ew)} / {len(other)} / {n}\n")
    text.append(f"Columns used: EW={col_ew}, OTHER={col_other}\n")
    text.append("r-statistics")
    text.append(f"  mean(r)_EW     = {mean_ew}")
    text.append(f"  mean(r)_OTHER  = {mean_other}")
    text.append(f"  delta means    = {delta_means}\n")
    if z_delta is not None:
        text.append("z-statistics (sanity check)")
        text.append(f"  mean(z)_EW     = {ew_z.mean()}")
        text.append(f"  mean(z)_OTHER  = {other_z.mean()}")
        text.append(f"  delta means    = {z_delta}\n")
    text.append("Affine map y_EW ≈ A + B * x_OTHER")
    text.append(f"  A = {A}")
    text.append(f"  B = {B}")
    text.append(f"  MSE = {mse}\n")
    text.append("JSON:")
    text.append(json.dumps(report, indent=2))

    with open(out_file, "w") as f:
        f.write("\n".join(text))

    print(f"Report written to {out_file}")

if __name__ == "__main__":
    main()