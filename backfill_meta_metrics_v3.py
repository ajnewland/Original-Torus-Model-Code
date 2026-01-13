import argparse, json, os, glob
from pathlib import Path
import numpy as np
import pandas as pd

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def robust_linfit(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan
    A = np.vstack([x[m], np.ones(m.sum())]).T
    coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    slope, intercept = coef
    yhat = A @ coef
    ss_res = np.sum((y[m] - yhat) ** 2)
    ss_tot = np.sum((y[m] - y[m].mean()) ** 2)
    R2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return float(slope), float(R2)

def estimate_xi_pixels(corr_df):
    # columns: r, C (any order)
    r = corr_df.get("r", corr_df.iloc[:, 0]).to_numpy()
    C = corr_df.get("C", corr_df.iloc[:, 1]).to_numpy()
    m = np.isfinite(r) & np.isfinite(C) & (r > 0) & (C > 0)
    r, C = r[m], C[m]
    if len(r) < 10:
        return np.nan, np.nan
    # fit tail: log C ~ -r/xi
    tail_len = max(10, len(r) // 3)
    r_tail = r[-tail_len:]
    C_tail = C[-tail_len:]
    slope, R2 = robust_linfit(r_tail, -np.log(C_tail))
    if not np.isfinite(slope) or slope <= 0:
        return np.nan, R2
    xi = 1.0 / slope
    return float(xi), float(R2)

def spectral_dimension_from_heat_trace(ht_df, tau_min, tau_max, windowsize):
    # columns: tau, K (any order)
    tau = ht_df.get("tau", ht_df.iloc[:, 0]).to_numpy()
    K   = ht_df.get("K",   ht_df.iloc[:, 1]).to_numpy()
    m = np.isfinite(tau) & np.isfinite(K) & (tau > 0) & (K > 0)
    tau, K = tau[m], K[m]
    m2 = (tau >= tau_min) & (tau <= tau_max)
    tau, K = tau[m2], K[m2]
    if len(tau) < max(windowsize + 2, 12):
        return dict(ds_plateau_mean=np.nan, ds_plateau_spread=np.nan,
                    ds_max=np.nan, ds_median=np.nan)
    ln_tau = np.log(tau)
    ln_K   = np.log(K)
    ds_vals = []
    half = max(1, windowsize // 2)
    for i in range(half, len(tau) - half):
        x = ln_tau[i - half:i + half + 1]
        y = ln_K[i - half:i + half + 1]
        if len(x) < 3:
            continue
        slope, _R2 = robust_linfit(x, y)
        if np.isfinite(slope):
            ds_vals.append(-2.0 * slope)
    if not ds_vals:
        return dict(ds_plateau_mean=np.nan, ds_plateau_spread=np.nan,
                    ds_max=np.nan, ds_median=np.nan)
    ds = np.array(ds_vals, dtype=float)
    return dict(
        ds_plateau_mean   = float(np.mean(ds)),
        ds_plateau_spread = float(np.std(ds)),
        ds_max            = float(np.max(ds)),
        ds_median         = float(np.median(ds)),
    )

def find_first(patterns, folder: Path):
    for pat in patterns:
        hits = sorted(glob.glob(str(folder / pat)))
        if hits:
            return Path(hits[0])
    return None

def process_run(run_dir: Path, windowsize, tau_min, tau_max, verbose=False):
    js = list(run_dir.glob("meta_summary.json"))
    if not js:
        return False, "no_json"
    jpath = js[0]
    meta = load_json(jpath)

    # locate correlation & heat trace csv (be flexible)
    corr_path = find_first(["corr_vs_distance*.csv"], run_dir)
    heat_path = find_first(["meta_heat_trace*.csv"], run_dir)

    changed = False

    if corr_path and corr_path.exists():
        try:
            corr_df = pd.read_csv(corr_path)
            xi, xiR2 = estimate_xi_pixels(corr_df)
            meta["xi_pixels"]     = None if not np.isfinite(xi)   else float(xi)
            meta["xi_logfit_R2"]  = None if not np.isfinite(xiR2) else float(xiR2)
            changed = True
        except Exception as e:
            if verbose: print(f"[{run_dir.name}] xi fail: {e}")

    if heat_path and heat_path.exists():
        try:
            ht_df = pd.read_csv(heat_path)
            ds_stats = spectral_dimension_from_heat_trace(ht_df, tau_min, tau_max, windowsize)
            meta.update(ds_stats)
            meta["windowsize"] = int(windowsize)
            changed = True
        except Exception as e:
            if verbose: print(f"[{run_dir.name}] ds fail: {e}")

    if changed:
        save_json(jpath, meta)
        return True, "updated"
    else:
        return False, "no_inputs"

def main():
    ap = argparse.ArgumentParser(description="Backfill xi and ds metrics into ANY meta_summary.json found under roots.")
    ap.add_argument("--roots", nargs="+", required=True, help="Root folders to scan recursively.")
    ap.add_argument("--windowsize", type=int, default=32)
    ap.add_argument("--ds_tau_min", type=float, default=1e-3)
    ap.add_argument("--ds_tau_max", type=float, default=1e+1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    total_dirs = 0
    found_runs = 0
    updated = 0
    skipped = 0

    for root in args.roots:
        rootp = Path(root)
        if not rootp.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(rootp):
            total_dirs += 1
            if "meta_summary.json" in filenames:
                found_runs += 1
                ok, msg = process_run(Path(dirpath), args.windowsize, args.ds_tau_min, args.ds_tau_max, args.verbose)
                if ok:
                    updated += 1
                else:
                    skipped += 1

    print(f"[DONE] walked {total_dirs} folders; found {found_runs} runs; updated {updated}, skipped {skipped}")

if __name__ == "__main__":
    main()