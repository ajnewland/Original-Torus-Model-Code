#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fit SM fermion masses from geometric 'norm_top.csv' using inline anchors,
optionally allowing non-monotone anchors.

Inputs
------
1) norm_top.csv (required): columns must include
   sp, Seff, Seff_sigma   (optionally: w, Abar, Bbar ...)
   z is computed as (Seff - 0.231)/Seff_sigma

2) anchors.csv (optional): headers exactly
   sp,z,PDG_GeV,PDG_sigma_GeV
   If omitted, use --use-default-anchors to take built-ins.

Flags
-----
--use-default-anchors    use six built-in anchors (e, mu, c, tau, b, t)
--allow-nonmonotone      do not abort if (z, log m) not monotone
--extrapolate {isotonic,linear}   default: isotonic for fit, linear for extrap
--bounds path/to/bounds.csv        (optional per-species min/max_GeV)

Outputs
-------
- predicted_masses.csv
- z_vs_logmass_fit.png
- mass_barplot.png
"""

import argparse, math, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TARGET = 0.231

# ---------- helpers ----------

def load_norm_csv(path):
    df = pd.read_csv(path)
    needed = {"sp","Seff","Seff_sigma"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"norm_top CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["z"] = (df["Seff"] - TARGET) / df["Seff_sigma"]
    return df

def default_anchors_from_norm(df_norm):
    """Use your 'Norm top results' logic: the anchor species are taken from the
    norm file to get their z; PDG masses are hard-coded below.
    """
    # PDG masses (illustrative 2024-ish values; tweak if you prefer)
    PDG = {
        "e":   (0.000511, 9e-09),
        "mu":  (0.105658, 0.002804),
        "c":   (1.27, 0.014),
        "tau": (1.77686, 0.012),
        "b":   (4.18, 0.16),
        "t":   (172.76, 5.7),
    }
    rows = []
    for sp, (m, s) in PDG.items():
        if sp not in set(df_norm["sp"]):
            raise ValueError(f"default anchor '{sp}' not found in norm_top.csv")
        z = float(df_norm.loc[df_norm["sp"]==sp, "z"].iloc[0])
        rows.append({"sp": sp, "z": z, "PDG_GeV": m, "PDG_sigma_GeV": s})
    return pd.DataFrame(rows)

def load_anchors_csv(path):
    df = pd.read_csv(path)
    needed = {"sp","z","PDG_GeV"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"anchors CSV missing columns: {sorted(missing)}")
    if "PDG_sigma_GeV" not in df.columns:
        df["PDG_sigma_GeV"] = np.nan
    return df[["sp","z","PDG_GeV","PDG_sigma_GeV"]].copy()

def load_bounds(path):
    b = pd.read_csv(path)
    if not {"sp","min_GeV","max_GeV"} <= set(b.columns):
        raise ValueError("bounds CSV needs columns: sp,min_GeV,max_GeV")
    return {r["sp"]:(float(r["min_GeV"]), float(r["max_GeV"])) for _,r in b.iterrows()}

def check_monotone(z, logm):
    # strictly increasing in z, nondecreasing in logm
    dz = np.diff(z)
    dm = np.diff(logm)
    monotone = np.all(dz > 0) and np.all(dm >= 0)
    return monotone

def isotonic_fit(xs, ys):
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        return iso.fit(xs, ys)
    except Exception as e:
        # Minimal fallback PAV (pool-adjacent-violators) for increasing fit
        x = np.asarray(xs, float)
        y = np.asarray(ys, float)
        order = np.argsort(x)
        x = x[order]; y = y[order]
        # pav
        yfit = y.copy()
        w = np.ones_like(yfit)
        i = 0
        while i < len(yfit)-1:
            if yfit[i] > yfit[i+1]:
                tot = yfit[i]*w[i] + yfit[i+1]*w[i+1]
                wt = w[i]+w[i+1]
                yfit[i] = yfit[i+1] = tot/wt
                w[i] = w[i+1] = wt
                j = i-1
                while j>=0 and yfit[j] > yfit[j+1]:
                    tot = yfit[j]*w[j] + yfit[j+1]*w[j+1]
                    wt = w[j]+w[j+1]
                    yfit[j] = yfit[j+1] = tot/wt
                    w[j] = w[j+1] = wt
                    j -= 1
                i = max(j,0)
            else:
                i += 1
        class _Iso:
            def __init__(self, x, yfit):
                self.x = x; self.yfit = yfit
            def predict(self, xq):
                xq = np.asarray(xq, float)
                yq = np.interp(xq, self.x, self.yfit)
                return yq
        return _Iso(x, yfit)

def linear_interp(xs, ys):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    order = np.argsort(xs)
    xs = xs[order]; ys = ys[order]
    class _Lin:
        def predict(self, xq):
            xq = np.asarray(xq, float)
            return np.interp(xq, xs, ys, left=ys[0], right=ys[-1])
    return _Lin()

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("norm_csv", help="norm_top.csv")
    ap.add_argument("anchors_csv", nargs="?", default=None, help="(optional) anchors.csv")
    ap.add_argument("--use-default-anchors", action="store_true")
    ap.add_argument("--allow-nonmonotone", action="store_true")
    ap.add_argument("--extrapolate", choices=["isotonic","linear"], default="linear",
                    help="interpolator for z→log m (fit is isotonic; this only sets extrap behavior)")
    ap.add_argument("--bounds", default=None, help="bounds.csv with sp,min_GeV,max_GeV")
    args = ap.parse_args()

    df_norm = load_norm_csv(args.norm_csv)

    if args.anchors_csv:
        anchors = load_anchors_csv(args.anchors_csv)
    elif args.use_default_anchors:
        anchors = default_anchors_from_norm(df_norm)
    else:
        raise SystemExit("Provide anchors.csv or --use-default-anchors")

    # sort anchors by z
    anchors = anchors.sort_values("z").reset_index(drop=True)
    z_a = anchors["z"].to_numpy(float)
    logm_a = np.log(anchors["PDG_GeV"].to_numpy(float) + 0.0)

    if not check_monotone(z_a, logm_a) and not args.allow_nonmonotone:
        raise SystemExit("Anchors (z, log m) are not monotone; re-run with --allow-nonmonotone "
                         "or revise anchors.")

    # Fit isotonic over anchors
    iso = isotonic_fit(z_a, logm_a)

    # For extrapolation behavior, wrap predictor
    if args.extrapolate == "linear":
        lin = linear_interp(z_a, logm_a)
        def predict_logm(zq):
            zq = np.asarray(zq, float)
            inside = (zq >= z_a[0]) & (zq <= z_a[-1])
            out = np.empty_like(zq, float)
            out[inside] = iso.predict(zq[inside])
            out[~inside] = lin.predict(zq[~inside])
            return out
    else:  # isotonic clip at ends
        def predict_logm(zq):
            return iso.predict(zq)

    # Optional bounds
    bounds = load_bounds(args.bounds) if args.bounds else {}

    # Predict for all species in norm file
    rows = []
    for _, r in df_norm.iterrows():
        sp = r["sp"]
        z = float(r["z"])
        logm = float(predict_logm([z])[0])
        m = float(np.exp(logm))
        # crude 1σ from z uncertainty propagated via local slope (finite diff)
        dz = 0.1
        m_hi = float(np.exp(predict_logm([z+dz])[0]))
        m_lo = float(np.exp(predict_logm([z-dz])[0]))
        sigma = abs(m_hi - m_lo)/2.0

        # apply bounds if provided
        if sp in bounds:
            mn, mx = bounds[sp]
            m = min(max(m, mn), mx)

        pdg = anchors.loc[anchors["sp"]==sp, "PDG_GeV"].iloc[0] if sp in set(anchors["sp"]) else np.nan
        ratio = m/pdg if pdg>0 else np.nan

        rows.append({"sp": sp, "z": z, "m_pred_GeV": m, "sigma_GeV": sigma,
                     "PDG_GeV": pdg, "ratio": ratio})

    df_out = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(args.norm_csv)) or "."
    out_csv = os.path.join(out_dir, "predicted_masses.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}\n")

    # Console table
    print("sp".ljust(6),"z".rjust(10),"m_pred[GeV]".rjust(14),"±sigma".rjust(10),"PDG[GeV]".rjust(12),"ratio".rjust(9))
    print("-"*66)
    for _,r in df_out.iterrows():
        print(f"{r['sp']:<6}{r['z']:>10.6f}{r['m_pred_GeV']:>14.6f}{r['sigma_GeV']:>10.6f}"
              f"{(r['PDG_GeV'] if not math.isnan(r['PDG_GeV']) else 0):>12.6f}"
              f"{(r['ratio'] if not math.isnan(r['ratio']) else 0):>9.2f}")

    # Plots
    # 1) z vs log mass anchors + fitted curve
    zz = np.linspace(min(df_norm["z"]), max(df_norm["z"]), 200)
    yy = predict_logm(zz)
    plt.figure(figsize=(7,5))
    plt.plot(zz, yy, label="isotonic fit (scaled)")
    plt.scatter(anchors["z"], np.log(anchors["PDG_GeV"]), label="anchors")
    plt.xlabel("z score")
    plt.ylabel("log mass")
    plt.title("Isotonic fit (anchors)")
    plt.legend()
    out_png1 = os.path.join(out_dir, "z_vs_logmass_fit.png")
    plt.tight_layout(); plt.savefig(out_png1, dpi=160); plt.close()
    print(f"Saved: {out_png1}")

    # 2) bar chart of predicted masses (with crude sigma)
    plt.figure(figsize=(9,6))
    order = list(df_out["sp"])
    plt.bar(order, df_out["m_pred_GeV"], yerr=df_out["sigma_GeV"], capsize=3)
    plt.ylabel("mass [GeV]")
    plt.title("Predicted masses (with ~1σ bands)")
    out_png2 = os.path.join(out_dir, "mass_barplot.png")
    plt.tight_layout(); plt.savefig(out_png2, dpi=160); plt.close()
    print(f"Saved: {out_png2}")

if __name__ == "__main__":
    main()