#!/usr/bin/env python3
# calibrate_mass_scale_noanchors.py
import argparse, json, math, os
import numpy as np
import pandas as pd

HBAR = 6.582119569e-25   # GeV*s
C    = 2.99792458e8      # m/s
ELL_P = 1.616255e-35     # m

def load_latent(path):
    df = pd.read_csv(path)
    # expected columns: ax, ay, z (at least)
    for col in ["ax","ay","z"]:
        if col not in df.columns:
            raise ValueError(f"latent CSV missing required column: {col}")
    # strip/clean
    df["ax"]=pd.to_numeric(df["ax"], errors="coerce")
    df["ay"]=pd.to_numeric(df["ay"], errors="coerce")
    df["z"] =pd.to_numeric(df["z"],  errors="coerce")
    df = df.dropna(subset=["ax","ay","z"]).copy()
    return df

def load_sector_slopes(path):
    # expects columns: sector, alpha, beta, R2, n
    df = pd.read_csv(path)
    if "alpha" not in df.columns:
        raise ValueError("sector_slopes.csv missing 'alpha'")
    # prefer fermion sectors; fallback to overall mean
    candidates = df[df["sector"].str.lower().isin(["up","down","leptons"])] if "sector" in df.columns else df
    if len(candidates)==0: candidates = df
    alpha = float(candidates["alpha"].astype(float).mean())
    return alpha, df

def load_locked(path):
    # expects: species, ax, ay (at least); z_pred or z
    df = pd.read_csv(path)
    want = ["species","ax","ay"]
    for w in want:
        if w not in df.columns:
            raise ValueError(f"locked CSV missing required column: {w}")
    zcol = "z_pred" if "z_pred" in df.columns else ("z" if "z" in df.columns else None)
    if zcol is None:
        raise ValueError("locked CSV needs either 'z_pred' or 'z'")
    df["z_use"] = pd.to_numeric(df[zcol], errors="coerce")
    df = df.dropna(subset=["z_use"])
    return df, zcol

def estimate_torus_area(df_latent, force_full=False):
    axu = np.unique(np.round(df_latent["ax"].values, 10))
    ayu = np.unique(np.round(df_latent["ay"].values, 10))
    if len(axu)<2 or len(ayu)<2:
        raise ValueError("latent grid needs >=2 unique ax and ay values to estimate spacings")
    dax = float(np.min(np.diff(np.sort(axu))))
    day = float(np.min(np.diff(np.sort(ayu))))
    Ax = dax * len(axu)
    Ay = day * len(ayu)
    A_grid = Ax * Ay
    A_full = (2*math.pi)*(2*math.pi)
    return A_full if force_full else A_grid, dict(Nx=len(axu), Ny=len(ayu), dax=dax, day=day, Ax=Ax, Ay=Ay,
                                                  A_grid=A_grid, A_full=A_full)

def fft_eigs_2d(Nx, Ny, dax, day):
    # Discrete Laplacian eigenvalues on a periodic Nx x Ny grid (torus)
    m = np.arange(Nx)
    n = np.arange(Ny)
    sinx2 = np.sin(np.pi*m/Nx)**2
    siny2 = np.sin(np.pi*n/Ny)**2
    lam = np.zeros((Nx,Ny))
    for i in range(Nx):
        lam[i,:] = 4.0*( sinx2[i]/(dax**2) + siny2/(day**2) )
    return lam

def fit_t_from_z_and_lambda(z_vals, lam_vals):
    # crude moment match: align mean and variance of z ~ -t*lambda
    lam = lam_vals.flatten()
    lam = lam[np.isfinite(lam)]
    z   = z_vals[np.isfinite(z_vals)]
    # remove zero mode lambda=0 to avoid bias of constant mode
    lam = lam[lam>1e-15]
    mu_l = np.mean(lam); var_l = np.var(lam)
    mu_z = np.mean(z);   var_z = np.var(z)
    # Best t (least squares on moments): solve [-t*mu_l ≈ mu_z,  t^2*var_l ≈ var_z]
    # Average the two estimates for robustness:
    t1 = -mu_z / mu_l if abs(mu_l)>1e-15 else np.nan
    t2 = math.sqrt(max(var_z,0.0)/var_l) if var_l>1e-15 and var_z>=0 else np.nan
    # choose geometric mean when both positive; else fallback to finite one
    cand=[]
    if np.isfinite(t1) and t1>0: cand.append(t1)
    if np.isfinite(t2) and t2>0: cand.append(t2)
    if len(cand)==0:
        raise ValueError("could not determine a positive t_* from moments")
    t_star = float(np.exp(np.mean(np.log(np.array(cand))))) if len(cand)==2 else float(cand[0])
    return t_star, dict(t1=t1, t2=t2)

def planck_locked_m0(A_eff):
    # m0 = (ħ / (c ℓP)) * sqrt(4π / A_eff)
    return (HBAR/(C*ELL_P)) * math.sqrt(4*math.pi / A_eff)

def heattrace_locked_m0(t_star, A_eff):
    # Same dimensional form as Planck-locked but with t_* in place of ℓP^2 scaling:
    # Treat t_* as an effective (length)^2 and set m0 ~ ħ/(c sqrt{t_*}) scaled by area
    return (HBAR/C) * math.sqrt((4*math.pi / A_eff) / max(t_star,1e-30))

def higgs_locked_m0(z_H, alpha, mH_GeV):
    # mH = m0 * exp(alpha * z_H) -> m0 = mH * exp(-alpha z_H)
    return float(mH_GeV) * math.exp(-alpha * float(z_H))

def main():
    ap = argparse.ArgumentParser(description="Anchor-free (and optional anchored) absolute mass calibration from torus latent geometry.")
    ap.add_argument("--latent", required=True, help="latent_z_merged*.csv with columns ax, ay, z")
    ap.add_argument("--locked", required=True, help="all_particles_locked.csv (needs species, ax, ay, and z_pred or z)")
    ap.add_argument("--slopes", required=True, help="sector_slopes.csv (to get alpha)")
    ap.add_argument("--mode", choices=["heattrace","planck","higgs"], default="heattrace",
                    help="heattrace (anchor-free), planck (anchor-free), or higgs (anchored)")
    ap.add_argument("--force_full_area", action="store_true", help="use (2π)^2 for torus area even if latent grid is a window")
    ap.add_argument("--higgs_mass", type=float, default=125.1, help="GeV (only for --mode higgs)")
    ap.add_argument("--species_H", default="H", help="row label for Higgs in locked CSV (only for --mode higgs)")
    ap.add_argument("--outcsv", required=True, help="output CSV with predicted absolute masses (GeV)")
    args = ap.parse_args()

    dfL = load_latent(args.latent)
    alpha, df_slopes = load_sector_slopes(args.slopes)
    dfK, zcol = load_locked(args.locked)

    # effective torus area from grid (or the full (2π)^2)
    A_eff, Ainfo = estimate_torus_area(dfL, force_full=args.force_full_area)

    # Prepare outputs
    meta = {"mode": args.mode, "alpha": alpha, "A_eff": A_eff, "Ainfo": Ainfo, "zcol_locked": zcol}

    if args.mode == "planck":
        m0 = planck_locked_m0(A_eff)
        t_star = None
        meta.update({"m0_GeV": m0})

    elif args.mode == "heattrace":
        # Build Laplacian eigenvalues on the latent grid dimensions
        Nx, Ny = Ainfo["Nx"], Ainfo["Ny"]
        dax, day = Ainfo["dax"], Ainfo["day"]
        lam = fft_eigs_2d(Nx, Ny, dax, day)
        # We only need z-values on the same grid footprint; use full z sample
        zvals = dfL["z"].values
        t_star, tmeta = fit_t_from_z_and_lambda(zvals, lam)
        m0 = heattrace_locked_m0(t_star, A_eff)
        meta.update({"t_star": t_star, "tmeta": tmeta, "m0_GeV": m0})

    else:  # higgs
        rowH = dfK[dfK["species"].astype(str).str.strip().str.lower() == args.species_H.lower()]
        if len(rowH)==0:
            raise ValueError(f"Could not find species '{args.species_H}' in locked CSV for Higgs-locked mode.")
        zH = float(rowH["z_use"].iloc[0])
        m0 = higgs_locked_m0(zH, alpha, args.higgs_mass)
        t_star = None
        meta.update({"m0_GeV": m0, "Higgs_mass_input_GeV": args.higgs_mass, "z_H": zH})

    # Predict absolute masses for all locked species
    df_out = dfK[["species","ax","ay","z_use"]].copy()
    df_out["alpha_used"] = alpha
    df_out["m0_GeV"] = m0
    df_out["m_pred_GeV"] = m0 * np.exp(alpha * df_out["z_use"].astype(float))

    # Order nicely
    order = ["species","ax","ay","z_use","alpha_used","m0_GeV","m_pred_GeV"]
    df_out = df_out[order]

    # Write
    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    df_out.to_csv(args.outcsv, index=False)
    with open(args.outcsv.replace(".csv","_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[WROTE] {args.outcsv}")
    print(f"[WROTE] {args.outcsv.replace('.csv','_meta.json')}")
    if args.mode == "heattrace":
        print(f"  alpha={alpha:.6f}, t_star={t_star:.6e}, m0={m0:.6e} GeV; A_eff={A_eff:.6f}")
    else:
        print(f"  alpha={alpha:.6f}, m0={m0:.6e} GeV; A_eff={A_eff:.6f}")