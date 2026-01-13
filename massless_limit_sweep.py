import argparse, subprocess, sys, csv, math, tempfile, os
from pathlib import Path

def call_predict(predict_script, latent, iso, sin2, alpha, beta, mass_GeV, label, tmp_out):
    masses_arg = f"{label}:{mass_GeV:.16g}"
    cmd = [
        sys.executable, predict_script,
        "--latent", latent,
        "--iso", iso,
        "--sin2", str(sin2),
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--masses", masses_arg,
        "--out", tmp_out
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("\n[ERROR] Predictor returned non-zero exit status.")
        print("Command:", " ".join(cmd))
        print("\n--- predictor output (stdout+stderr) ---")
        print(e.output)
        print("--- end predictor output ---\n")
        raise

    # Read the CSV we asked the predictor to write
    with open(tmp_out, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Predictor wrote an empty CSV (no rows).")
    row = rows[0]
    ax = float(row["ax"]); ay = float(row["ay"])
    zt = float(row["z_target"]); zp = float(row["z_pred"]); err = float(row["abs_err"])
    return ax, ay, zt, zp, err, out

def main():
    ap = argparse.ArgumentParser(description="Sweep mass -> 0 and record (ax,ay) convergence.")
    ap.add_argument("--predict_script", required=True)
    ap.add_argument("--latent", required=True)
    ap.add_argument("--iso", required=True)
    ap.add_argument("--sin2", type=float, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--start_mass", type=float, default=1e-6, help="GeV")
    ap.add_argument("--ratio", type=float, default=0.3, help="multiply mass by this each step (0<ratio<1)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--label", default="nu1", help="species label to pass to predictor (use a known label like nu1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep_tmp", action="store_true", help="keep per-step tmp CSVs next to --out")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Where to write predictor tmp CSVs
    if args.keep_tmp:
        tmp_dir = out_path.parent / (out_path.stem + "_tmp")
        tmp_dir.mkdir(exist_ok=True)
    else:
        tmp_dir = Path(tempfile.gettempdir())

    rows = []
    m = args.start_mass
    last_ax = last_ay = None
    for k in range(args.steps):
        tmp_csv = str(tmp_dir / f"mls_tmp_step{k}.csv")
        ax, ay, zt, zp, err, _ = call_predict(
            args.predict_script, args.latent, args.iso,
            args.sin2, args.alpha, args.beta, m, args.label, tmp_csv
        )
        dax = "" if last_ax is None else f"{(ax - last_ax):.9e}"
        day = "" if last_ay is None else f"{(ay - last_ay):.9e}"
        rows.append({
            "step": k,
            "mass_GeV": f"{m:.16g}",
            "ax": f"{ax:.9f}",
            "ay": f"{ay:.9f}",
            "dax": dax,
            "day": day,
            "z_target": f"{zt:.9f}",
            "z_pred": f"{zp:.9f}",
            "abs_err": f"{err:.3e}",
        })
        last_ax, last_ay = ax, ay
        m *= args.ratio

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] wrote {out_path}")
    print("limit ~ (ax, ay) =", rows[-1]["ax"], rows[-1]["ay"], "at mass =", rows[-1]["mass_GeV"], "GeV")
    if args.keep_tmp:
        print(f"[info] kept step CSVs in: {tmp_dir}")

if __name__ == "__main__":
    main()