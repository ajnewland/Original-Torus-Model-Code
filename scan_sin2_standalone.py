import argparse, os, sys, csv, subprocess, tempfile
from pathlib import Path
import pandas as pd

def frange(start, stop, step):
    vals = []
    x = start
    # include stop within floating tolerance
    while x <= stop + 1e-12:
        vals.append(round(x, 6))
        x += step
    return vals

def run_predict(python_exe, predict_script, latent, iso, sin2, alpha, beta, masses, out_csv):
    cmd = [
        python_exe, predict_script,
        "--latent", latent,
        "--iso", iso,
        "--sin2", str(sin2),
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--masses", masses,
        "--out", out_csv
    ]
    env = os.environ.copy()
    # Force UTF-8 so any unicode printing in the child won't crash on Windows
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--python_exe", default=sys.executable)
    p.add_argument("--predict_script", required=True)
    p.add_argument("--latent", required=True)
    p.add_argument("--iso", required=True)
    p.add_argument("--alpha", type=float, default=0.0647)
    p.add_argument("--beta",  type=float, default=0.5529)
    p.add_argument("--masses", required=True,
                   help='e.g. "u:0.0022,d:0.0047,s:0.095,e:0.000511,mu:0.10566,c:1.27,tau:1.77686,b:4.18,t:172.76,H:125.1,W:80.4,Z:91.2"')
    p.add_argument("--sin2_min", type=float, required=True)
    p.add_argument("--sin2_max", type=float, required=True)
    p.add_argument("--sin2_step", type=float, required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tmpdir = outdir / "tmp"
    tmpdir.mkdir(exist_ok=True)

    rows = []
    trials = frange(args.sin2_min, args.sin2_max, args.sin2_step)

    print(f"[INFO] Running {len(trials)} sin2 trials from {args.sin2_min} to {args.sin2_max} step {args.sin2_step}")

    for s in trials:
        tmp_csv = str(tmpdir / f"trial_sin2_{s:.6f}.csv")
        rc, stdout, stderr = run_predict(
            args.python_exe, args.predict_script,
            args.latent, args.iso, s, args.alpha, args.beta,
            args.masses, tmp_csv
        )
        if rc != 0 or not os.path.exists(tmp_csv) or os.path.getsize(tmp_csv) == 0:
            # log failure but keep going
            fail_log = outdir / "failures.log"
            with open(fail_log, "a", encoding="utf-8") as flog:
                flog.write(f"\n=== sin2={s:.6f} FAILED (rc={rc}) ===\n")
                flog.write(stdout or "")
                flog.write("\n--- STDERR ---\n")
                flog.write(stderr or "")
                flog.write("\n")
            print(f"[WARN] sin2={s:.6f} failed; see {fail_log.name}")
            continue

        try:
            df = pd.read_csv(tmp_csv)
            if "abs_err" not in df.columns:
                raise ValueError("abs_err column missing in predictor output")
            total_abs_err = float(df["abs_err"].abs().sum())
            best_abs_err = float(df["abs_err"].abs().min())
            max_abs_err  = float(df["abs_err"].abs().max())
            n_ok = int(df["ok"].astype(str).str.lower().eq("true").sum()) if "ok" in df.columns else None
            n_rows = int(len(df))
            rows.append({
                "sin2": s,
                "total_abs_err": f"{total_abs_err:.12g}",
                "best_abs_err": f"{best_abs_err:.12g}",
                "max_abs_err":  f"{max_abs_err:.12g}",
                "n_ok": n_ok if n_ok is not None else "",
                "n_rows": n_rows
            })
            print(f"[OK] sin2={s:.6f} total_abs_err={total_abs_err:.6g} n_rows={n_rows}")
        except Exception as e:
            fail_log = outdir / "failures.log"
            with open(fail_log, "a", encoding="utf-8") as flog:
                flog.write(f"\n=== sin2={s:.6f} PARSE ERROR === {e}\n")
            print(f"[WARN] sin2={s:.6f} parse error; see {fail_log.name}")

    summary_csv = outdir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sin2","total_abs_err","best_abs_err","max_abs_err","n_ok","n_rows"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if rows:
        best = min(rows, key=lambda r: float(r["total_abs_err"]))
        print(f"\n[DONE] Wrote {summary_csv}")
        print(f"[BEST] sin2={best['sin2']:.6f} with total_abs_err={best['total_abs_err']}")
    else:
        print("\n[END] No successful trials. Check failures.log for details.")

if __name__ == "__main__":
    main()