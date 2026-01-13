#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit species masses from a geometric scalar z, using anchored (z, PDG mass) points.
Enhancements:
  - Extrapolation control: --extrapolate {clip,linear}
  - Global floors/ceilings: --mass-floor, --mass-ceil (GeV)
  - Per-species bounds CSV: --bounds <csv>  (columns: sp,min_GeV,max_GeV)
  - Anchors validation: --check-anchors

Inputs (same as before):
  1) species CSV (e.g., norm_top.csv). Must contain either:
       - column 'z'  (directly), OR
       - columns 'Seff','Seff_sigma' (then we compute z = (Seff-0.231)/Seff_sigma)
     Optional columns used only for plotting/error bars:
       - 'm_GeV', 'm_sigma_GeV'
  2) anchors CSV (z, PDG_GeV[, Delta, w, sigma_GeV])  -- at least 3 anchors

Outputs:
  - predicted_masses.csv
  - z_vs_logmass_fit.png
  - mass_barplot.png

Examples
--------
# Default (clip)
python fit_species_masses.py norm_top.csv anchors.csv

# Linear extrapolation and a global floor at 1 MeV, neutrino ceiling 0.2 eV:
python fit_species_masses.py norm_top.csv anchors.csv --extrapolate linear --mass-floor 0.001 --mass-ceil 200e-9

# Per-species bounds
python fit_species_masses.py norm_top.csv anchors.csv --bounds bounds.csv

# Just validate anchors
python fit_species_masses.py dummy.csv anchors.csv --check-anchors
"""
import argparse, csv, math, os, sys
from collections import defaultdict

TARGET = 0.231  # sin^2(theta_W) target – used only if we must compute z from Seff
EPS = 1e-12

# ---------- small utils ----------
def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k,v in r.items()})
    return rows

def to_float(x, default=None):
    try:
        if x is None or x=="":
            return default
        return float(x)
    except Exception:
        return default

def has_col(rows, col):
    return rows and col in rows[0]

def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        # union of keys
        ks = set()
        for r in rows: ks |= set(r.keys())
        fieldnames = list(ks)
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

def safe_log(x):
    return math.log(max(x, EPS))

def lin_interp(x, x0, y0, x1, y1):
    if abs(x1-x0) < EPS:
        return 0.5*(y0+y1)
    t = (x - x0)/(x1 - x0)
    return (1.0-t)*y0 + t*y1

# ---------- anchors validator ----------
def validate_anchors(anchors_rows, verbose=True):
    problems = []
    need = ["sp", "z", "PDG_GeV"]
    for col in need:
        if not has_col(anchors_rows, col):
            problems.append(f"Missing required column '{col}' in anchors CSV.")
    parsed = []
    for i,row in enumerate(anchors_rows):
        sp = row.get("sp","").strip()
        z  = to_float(row.get("z"))
        m  = to_float(row.get("PDG_GeV"))
        if not sp: problems.append(f"Row {i+1}: empty 'sp'.")
        if z is None: problems.append(f"Row {i+1} ({sp}): invalid 'z'.")
        if m is None or m<=0: problems.append(f"Row {i+1} ({sp}): invalid 'PDG_GeV' (>0 required).")
        parsed.append((sp,z,m))
    # check duplicates
    seen = set()
    for sp,_,_ in parsed:
        if sp in seen:
            problems.append(f"Duplicate anchor species '{sp}'.")
        seen.add(sp)
    # check monotonic order (advisory): log mass should increase with z if you intend isotonic
    zs  = [z for _,z,_ in parsed if z is not None]
    lms = [safe_log(m) for _,_,m in parsed if m is not None and m>0]
    if len(zs)==len(lms) and len(zs)>=3:
        pairs = sorted(zip(zs,lms), key=lambda t:t[0])
        nonmono = False
        for i in range(1,len(pairs)):
            if pairs[i][1] < pairs[i-1][1] - 1e-9:
                nonmono = True; break
        if nonmono:
            problems.append("Anchors (z, log(PDG)) are not monotone. Isotonic fit will enforce monotonicity, "
                            "but consider revising anchors or using linear interpolation through sorted points.")
    ok = (len(problems)==0)
    if verbose:
        if ok:
            print(f"Anchors OK: {len(anchors_rows)} rows, columns {list(anchors_rows[0].keys())}")
        else:
            print("Anchor validation problems:")
            for p in problems:
                print(" -", p)
    return ok, problems

# ---------- fit map: z -> log(mass) ----------
def build_monotone_map(anchors_rows, fit="isotonic", use_sklearn=True):
    """
    Returns a dict with:
      'z_sorted', 'lm_sorted' (log-mass), 'mode' in {"isotonic","piecewise"}
    If sklearn is unavailable or anchors are <3, falls back to piecewise-linear.
    """
    zs, lms = [], []
    for r in anchors_rows:
        z = to_float(r.get("z"))
        m = to_float(r.get("PDG_GeV"))
        if z is None or m is None or m<=0: continue
        zs.append(z); lms.append(safe_log(m))
    order = sorted(range(len(zs)), key=lambda i: zs[i])
    z_sorted  = [zs[i] for i in order]
    lm_sorted = [lms[i] for i in order]
    mode = "piecewise"
    if fit=="isotonic" and use_sklearn:
        try:
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
            lm_fit = ir.fit_transform(z_sorted, lm_sorted)
            lm_sorted = list(lm_fit)
            mode = "isotonic"
        except Exception:
            mode = "piecewise"
    return {"z_sorted": z_sorted, "lm_sorted": lm_sorted, "mode": mode}

def eval_map(z, mp, extrapolate="clip"):
    z_sorted  = mp["z_sorted"]
    lm_sorted = mp["lm_sorted"]
    if not z_sorted:
        return None
    # in-range: interpolate between nearest neighbors
    if z <= z_sorted[0]:
        if extrapolate=="linear" and len(z_sorted)>=2:
            return lin_interp(z, z_sorted[0], lm_sorted[0], z_sorted[1], lm_sorted[1])
        return lm_sorted[0]
    if z >= z_sorted[-1]:
        if extrapolate=="linear" and len(z_sorted)>=2:
            return lin_interp(z, z_sorted[-2], lm_sorted[-2], z_sorted[-1], lm_sorted[-1])
        return lm_sorted[-1]
    # find bracket
    lo, hi = 0, len(z_sorted)-1
    while hi - lo > 1:
        mid = (lo+hi)//2
        if z_sorted[mid] <= z: lo = mid
        else: hi = mid
    return lin_interp(z, z_sorted[lo], lm_sorted[lo], z_sorted[hi], lm_sorted[hi])

# ---------- floors/ceilings ----------
def load_bounds(bounds_csv):
    if not bounds_csv: return {}
    rows = read_csv(bounds_csv)
    out = {}
    for r in rows:
        sp = r.get("sp","").strip()
        if not sp: continue
        mn = to_float(r.get("min_GeV"))
        mx = to_float(r.get("max_GeV"))
        out[sp] = (mn, mx)
    return out

def apply_bounds(sp, mass, global_floor, global_ceil, per_sp_bounds):
    mn, mx = global_floor, global_ceil
    if sp in per_sp_bounds:
        sp_mn, sp_mx = per_sp_bounds[sp]
        if sp_mn is not None: mn = sp_mn
        if sp_mx is not None: mx = sp_mx
    if mn is not None: mass = max(mass, mn)
    if mx is not None: mass = min(mass, mx)
    return mass

# ---------- z from species CSV (if not provided) ----------
def compute_z_from_species(row):
    # Preferred: direct z
    z = to_float(row.get("z"))
    if z is not None:
        return z
    # Fallback: z-score from Seff relative to TARGET using Seff_sigma
    seff = to_float(row.get("Seff"))
    seff_sig = to_float(row.get("Seff_sigma"))
    if seff is not None and (seff_sig is not None and seff_sig>0):
        return (seff - TARGET)/seff_sig
    # Last resort: try y (dimensionless Yukawa-like) with its sigma (rare)
    y = to_float(row.get("y"))
    ysig = to_float(row.get("y_sigma"))
    if y is not None and (ysig is not None and ysig>0):
        return (y-1.0)/ysig
    return None

# ---------- plotting ----------
def try_plot_isotonic(mp, anchors_rows, out_png):
    try:
        import matplotlib.pyplot as plt
        zs  = [to_float(r["z"]) for r in anchors_rows]
        lms = [safe_log(to_float(r["PDG_GeV"])) for r in anchors_rows]
        pts = sorted(zip(zs,lms), key=lambda t:t[0])
        xs  = [p[0] for p in pts]
        ys  = [p[1] for p in pts]
        # draw fit curve
        xs_dense = []
        ys_dense = []
        if xs:
            x0, x1 = xs[0], xs[-1]
            N = 200
            for i in range(N+1):
                x = x0 + (x1-x0)*i/N
                xs_dense.append(x)
                ys_dense.append(eval_map(x, mp, extrapolate="clip"))
        plt.figure(figsize=(7,5))
        if xs_dense:
            plt.plot(xs_dense, ys_dense, label="isotonic fit (scaled)")
        plt.scatter(xs, ys, label="anchors")
        plt.xlabel("z score"); plt.ylabel("log mass")
        plt.title("Isotonic fit (anchors)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=160); plt.close()
        print(f"Saved: {out_png}")
    except Exception as e:
        print(f"(Plot skipped: {e})")

def try_plot_bars(pred_rows, out_png):
    try:
        import matplotlib.pyplot as plt
        names = [r["sp"] for r in pred_rows]
        vals  = [to_float(r["m_pred_GeV"]) for r in pred_rows]
        sigs  = [to_float(r.get("sigma_GeV")) or 0.0 for r in pred_rows]
        xs = list(range(len(names)))
        plt.figure(figsize=(10,6))
        plt.bar(xs, vals, yerr=sigs, capsize=3)
        plt.xticks(xs, names, rotation=0)
        plt.ylabel("mass [GeV]")
        plt.title("Predicted masses (with ~1σ bands)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=160); plt.close()
        print(f"Saved: {out_png}")
    except Exception as e:
        print(f"(Plot skipped: {e})")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species_csv", help="species table (norm_top.csv or similar)")
    ap.add_argument("anchors_csv", help="anchors table (sp,z,PDG_GeV[,sigma])")
    ap.add_argument("--extrapolate", choices=["clip","linear"], default="clip",
                    help="behavior outside anchor z-range (default: clip)")
    ap.add_argument("--mass-floor", type=float, default=None, dest="mass_floor",
                    help="global minimum mass in GeV (applied after mapping)")
    ap.add_argument("--mass-ceil", type=float, default=None, dest="mass_ceil",
                    help="global maximum mass in GeV (applied after mapping)")
    ap.add_argument("--bounds", type=str, default=None,
                    help="CSV with per-species min/max bounds: sp,min_GeV,max_GeV")
    ap.add_argument("--check-anchors", action="store_true",
                    help="validate anchors and exit")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    anchors = read_csv(args.anchors_csv)
    ok, probs = validate_anchors(anchors, verbose=True)
    if args.check_anchors:
        return 0 if ok else 3
    if not ok:
        print("Refusing to run fit until anchors issues are fixed.")
        return 3

    species = read_csv(args.species_csv)
    if not species:
        print("No species rows.")
        return 2

    # Build monotone map z -> log(mass)
    mp = build_monotone_map(anchors, fit="isotonic", use_sklearn=True)
    print(f"Fit mode: {mp['mode']}  (points={len(mp['z_sorted'])})")

    # Optional per-species bounds
    per_bounds = load_bounds(args.bounds)

    # Predict
    preds = []
    for r in species:
        sp = r.get("species") or r.get("sp") or r.get("name") or ""
        sp = sp.strip()
        if not sp: continue
        z = compute_z_from_species(r)
        if z is None:
            print(f"Warning: skipping '{sp}' (no z and not enough info to compute).")
            continue
        lm = eval_map(z, mp, extrapolate=args.extrapolate)
        if lm is None:
            print(f"Warning: map failed for '{sp}'.")
            continue
        m = math.exp(lm)
        # naive 1σ mass band: reuse species m_sigma_GeV if present, else ~3% of m
        sigma = to_float(r.get("m_sigma_GeV"), default=None)
        if sigma is None:
            sigma = 0.033*m
        # apply floors/ceilings
        m_bounded = apply_bounds(sp, m, args.mass_floor, args.mass_ceil, per_bounds)
        # if clipped, adjust sigma to not exceed box (conservative)
        if args.mass_floor is not None or args.mass_ceil is not None or sp in per_bounds:
            if args.mass_floor is not None and m - sigma < args.mass_floor:
                sigma = max(0.0, m_bounded - args.mass_floor)
            if args.mass_ceil is not None and m + sigma > args.mass_ceil:
                sigma = max(0.0, args.mass_ceil - m_bounded)

        # If PDG ref exists in species CSV, include it (for quick ratios on anchors)
        pdg = to_float(r.get("PDG_GeV"), default=None)
        ratio = (m_bounded/pdg) if (pdg and pdg>0) else None

        preds.append({
            "sp": sp,
            "z": f"{z:.6f}",
            "m_pred_GeV": f"{m_bounded:.6f}",
            "sigma_GeV": f"{sigma:.6f}",
            "PDG_GeV": (f"{pdg:.6f}" if pdg else ""),
            "ratio": (f"{ratio:.3f}" if ratio else "")
        })

    # Order nicely: anchors first (if present in species), then others by z
    anchor_names = [r["sp"] for r in anchors]
    anchor_set = set(anchor_names)
    preds_anchors = [p for p in preds if p["sp"] in anchor_set]
    preds_others  = [p for p in preds if p["sp"] not in anchor_set]
    preds_others.sort(key=lambda r: float(r["z"]))
    out_rows = preds_anchors + preds_others

    out_csv = os.path.join(os.path.dirname(args.species_csv) or ".", "predicted_masses.csv")
    write_csv(out_csv, out_rows,
              fieldnames=["sp","z","m_pred_GeV","sigma_GeV","PDG_GeV","ratio"])
    print(f"Wrote: {out_csv}\n")

    # Pretty print table
    print(f"{'sp':<6}{'z':>12}{'m_pred[GeV]':>15}{'±sigma':>10}{'PDG[GeV]':>12}{'ratio':>10}")
    print("-"*64)
    for r in out_rows:
        print(f"{r['sp']:<6}{float(r['z']):12.6f}{float(r['m_pred_GeV']):15.6f}"
              f"{float(r['sigma_GeV']):10.6f}{(r['PDG_GeV'] or ''):>12}{(r['ratio'] or ''):>10}")

    if not args.no-plot:
        base = os.path.dirname(args.species_csv) or "."
        try_plot_isotonic(mp, anchors, os.path.join(base, "z_vs_logmass_fit.png"))
        try_plot_bars(out_rows,     os.path.join(base, "mass_barplot.png"))

    return 0

if __name__ == "__main__":
    sys.exit(main())