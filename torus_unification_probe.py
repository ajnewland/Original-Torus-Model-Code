#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, sys
import numpy as np
import pandas as pd

# -------------------------------
# Utilities
# -------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def col_or_fail(df: pd.DataFrame, colnames, want_label):
    for c in colnames:
        if c in df.columns:
            return df[c]
    raise ValueError(f"latent CSV must contain column '{want_label}' (looked for {colnames}); "
                     f"found columns: {list(df.columns)}")

def infer_mass_from_locks(locks_df: pd.DataFrame, species_name: str, mass_col_candidates=("m_GeV","mass_GeV","mass")):
    if "species" not in locks_df.columns:
        raise ValueError("locks CSV must have a 'species' column")

    row = locks_df.loc[locks_df["species"].str.lower() == species_name.lower()]
    if row.empty:
        return None

    for c in mass_col_candidates:
        if c in locks_df.columns:
            val = float(row.iloc[0][c])
            return val
    # fallback: try to parse from any numeric column if obvious
    numcols = [c for c in locks_df.columns if np.issubdtype(locks_df[c].dtype, np.number)]
    if numcols:
        return float(row.iloc[0][numcols[0]])
    return None

def make_interior_mask(ax, ay, z, mask_band, policy="percentile", erosion=0.01, min_interior=100):
    """Robust interior selection. Falls back to 'all' if too few points."""
    axb = ax[mask_band]; ayb = ay[mask_band]; zb = z[mask_band]

    if mask_band.sum() == 0:
        return mask_band

    if policy == "all":
        mask_interior = mask_band.copy()

    elif policy == "bbox":
        # strict bounding-box interior
        eps = 1e-15
        ax_lo, ax_hi = axb.min(), axb.max()
        ay_lo, ay_hi = ayb.min(), ayb.max()
        z_lo,  z_hi  = zb.min(),  zb.max()
        mask_interior = (mask_band &
                         (ax > ax_lo + eps) & (ax < ax_hi - eps) &
                         (ay > ay_lo + eps) & (ay < ay_hi - eps) &
                         (z  > z_lo  + eps) & (z  < z_hi  - eps))

    elif policy == "percentile":
        lo = 100 * erosion
        hi = 100 * (1 - erosion)
        ax_lo, ax_hi = np.percentile(axb, [lo, hi])
        ay_lo, ay_hi = np.percentile(ayb, [lo, hi])
        z_lo,  z_hi  = np.percentile(zb,  [lo, hi])
        mask_interior = (mask_band &
                         (ax > ax_lo) & (ax < ax_hi) &
                         (ay > ay_lo) & (ay < ay_hi) &
                         (z  > z_lo)  & (z  < z_hi))
    else:
        raise ValueError(f"Unknown interior policy: {policy}")

    if mask_interior.sum() < min_interior:
        print(f"[warn] interior points={int(mask_interior.sum())} < {min_interior}; "
              f"falling back to all band points (policy={policy})")
        mask_interior = mask_band.copy()

    return mask_interior

def linear_surface_fit(ax, ay, z, mask):
    """Fit z ≈ a0 + a1*ax + a2*ay over mask (simple least squares)."""
    X = np.column_stack([np.ones_like(ax[mask]), ax[mask], ay[mask]])
    y = z[mask]
    # least squares
    coeff, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0, a1, a2 = coeff.tolist()
    # residuals
    zhat = (a0 + a1 * ax[mask] + a2 * ay[mask])
    resid = y - zhat
    mse = float(np.mean(resid**2))
    return dict(a0=a0, a1=a1, a2=a2, mse=mse)

def curvature_proxy(ax, ay, z, mask):
    """
    Very benign curvature proxy: fit a quadratic form
    z ≈ c0 + c1*ax + c2*ay + c3*ax^2 + c4*ay^2 + c5*ax*ay
    and report the fraction of positive curvature directions.
    This is a PROXY, not a physical spin-2 measurement.
    """
    X = np.column_stack([
        np.ones_like(ax[mask]),
        ax[mask],
        ay[mask],
        ax[mask]**2,
        ay[mask]**2,
        ax[mask]*ay[mask],
    ])
    y = z[mask]
    try:
        coeff, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return dict(note="quadratic fit failed", fraction_positive=0.0, eigenvalues=[])

    # Approximate Hessian from quadratic terms:
    # H ≈ [[2*c3, c5],[c5, 2*c4]]
    c3 = float(coeff[3]); c4 = float(coeff[4]); c5 = float(coeff[5])
    H = np.array([[2*c3, c5],
                  [c5,  2*c4]], dtype=float)
    evals = np.linalg.eigvalsh(H)
    frac_pos = float(np.mean(evals > 0))
    return dict(note="quadratic curvature proxy only",
                fraction_positive=frac_pos,
                eigenvalues=evals.tolist(),
                c3=c3, c4=c4, c5=c5)

def compute_sin2_from_masses(mW, mZ):
    """Tree-level relation sin^2 θ_W = 1 - (mW/mZ)^2"""
    return 1.0 - (mW / mZ) ** 2

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="EW band probe with robust interior selection.")
    ap.add_argument("--latent-csv", required=True, help="CSV with at least ax, ay, z, r columns")
    ap.add_argument("--locks", required=True, help="CSV with species masses (needs W and Z)")
    ap.add_argument("--outdir", required=True)

    # Column names
    ap.add_argument("--ax-col", default="ax")
    ap.add_argument("--ay-col", default="ay")
    ap.add_argument("--z-col",  default="S_star", help="Use S_star for EW band z")
    ap.add_argument("--r-col",  default="r")

    # Interior policy
    ap.add_argument("--interior-policy",
                    choices=["bbox", "percentile", "all"],
                    default="percentile",
                    help="Interior definition: 'bbox' strict, 'percentile' trimmed, 'all' = disable interior test.")
    ap.add_argument("--erosion", type=float, default=0.01,
                    help="For 'percentile' policy: fraction of range trimmed at each side (0.01 = 1%).")
    ap.add_argument("--min-interior", type=int, default=100,
                    help="If interior points fewer than this, fall back to all band points.")

    # Optional additional band filters (usually not needed if you prefilter)
    ap.add_argument("--aymin", type=float, default=None)
    ap.add_argument("--aymax", type=float, default=None)
    ap.add_argument("--zmin",  type=float, default=None)
    ap.add_argument("--zmax",  type=float, default=None)

    args = ap.parse_args()
    ensure_dir(args.outdir)

    # Load data
    df = load_csv(args.latent_csv)
    locks = load_csv(args.locks)

    # Map columns
    ax = col_or_fail(df, [args.ax_col, "ax", "a_x", "alpha_x"], "ax").to_numpy(dtype=float)
    ay = col_or_fail(df, [args.ay_col, "ay", "a_y", "alpha_y"], "ay").to_numpy(dtype=float)
    z  = col_or_fail(df, [args.z_col,  "S_star", "z", "z_pred", "z_target"], "z").to_numpy(dtype=float)
    r  = col_or_fail(df, [args.r_col,  "r", "AminusB", "A_minus_B"], "r").to_numpy(dtype=float)

    # Band selection (if user provided extra bounds)
    mask_band = np.ones(len(df), dtype=bool)
    if args.aymin is not None:
        mask_band &= (ay >= args.aymin)
    if args.aymax is not None:
        mask_band &= (ay <= args.aymax)
    if args.zmin is not None:
        mask_band &= (z  >= args.zmin)
    if args.zmax is not None:
        mask_band &= (z  <= args.zmax)

    if mask_band.sum() == 0:
        raise ValueError("No rows survive band filtering. Remove ay/z bounds or widen them.")

    # Interior selection (robust)
    mask_interior = make_interior_mask(
        ax, ay, z, mask_band,
        policy=args.interior_policy,
        erosion=args.erosion,
        min_interior=args.min_interior
    )

    # Fit a simple linear EW field over the interior (for a stable diagnostic)
    lin = linear_surface_fit(ax, ay, z, mask_interior)
    ew_struct_path = os.path.join(args.outdir, "ew_structure_constants.csv")
    pd.DataFrame([lin]).to_csv(ew_struct_path, index=False)

    # Curvature proxy (clearly labeled as proxy)
    curv = curvature_proxy(ax, ay, z, mask_interior)
    spin2_path = os.path.join(args.outdir, "spin2_fraction_summary.json")
    with open(spin2_path, "w") as f:
        json.dump(curv, f, indent=2)

    # sin^2 θ_W from masses (locks) and from band geometry
    mW = infer_mass_from_locks(locks, "W")
    mZ = infer_mass_from_locks(locks, "Z")
    sin2_mass = compute_sin2_from_masses(mW, mZ) if (mW and mZ) else None
    sin2_geo  = float(np.mean(z[mask_interior]))
    sin2_delta = None if sin2_mass is None else float(sin2_geo - sin2_mass)

    sin2_summary = dict(
        mW_GeV=mW, mZ_GeV=mZ,
        sin2_mass=sin2_mass,
        sin2_geo=sin2_geo,
        delta_sin2=sin2_delta,
        interior_policy=args.interior_policy,
        erosion=args.erosion,
        min_interior=args.min_interior,
        n_band=int(mask_band.sum()),
        n_interior=int(mask_interior.sum()),
        z_mean=float(np.mean(z[mask_interior])),
        z_std=float(np.std(z[mask_interior]))
    )
    sin2_path = os.path.join(args.outdir, "sin2_summary.json")
    with open(sin2_path, "w") as f:
        json.dump(sin2_summary, f, indent=2)

    # Also dump a residuals map for inspection
    df_out = pd.DataFrame({
        "ax": ax[mask_interior],
        "ay": ay[mask_interior],
        "z":  z[mask_interior],
        "r":  r[mask_interior]
    })
    # Linear residuals relative to the fit:
    zhat = lin["a0"] + lin["a1"] * df_out["ax"] + lin["a2"] * df_out["ay"]
    df_out["z_resid"] = df_out["z"] - zhat
    df_out.to_csv(os.path.join(args.outdir, "ew_residuals_map.csv"), index=False)

    print("=== Probe complete ===")
    print(f"Interior policy: {args.interior_policy}  |  interior points: {mask_interior.sum()}  / band: {mask_band.sum()}")
    print(f"Linear field fit: a0={lin['a0']:.6g}, a1={lin['a1']:.6g}, a2={lin['a2']:.6g}, mse={lin['mse']:.3g}")
    if sin2_mass is not None:
        print(f"sin^2θ_W mass = {sin2_mass:.6f} | geo(band mean) = {sin2_geo:.6f} | Δ = {sin2_geo - sin2_mass:+.6f}")
    else:
        print(f"sin^2θ_W geo(band mean) = {sin2_geo:.6f} (no W/Z masses found in locks)")
    print(f"Wrote:\n  - {ew_struct_path}\n  - {spin2_path}\n  - {sin2_path}\n  - {os.path.join(args.outdir, 'ew_residuals_map.csv')}")

if __name__ == "__main__":
    main()