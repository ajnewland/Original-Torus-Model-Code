#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as npl

# ----------------------------
# Grid + interpolation helpers
# ----------------------------
def build_grid(axv, ayv, n):
    ax_min, ax_max = float(np.min(axv)), float(np.max(axv))
    ay_min, ay_max = float(np.min(ayv)), float(np.max(ayv))
    pad_x = 0.02*(ax_max-ax_min if ax_max>ax_min else 1.0)
    pad_y = 0.02*(ay_max-ay_min if ay_max>ay_min else 1.0)
    x = np.linspace(ax_min - pad_x, ax_max + pad_x, n)
    y = np.linspace(ay_min - pad_y, ay_max + pad_y, n)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y

def idw_interp(xp, yp, zp, X, Y, power=2, eps=1e-12):
    Zi = np.zeros_like(X, dtype=float)
    xp = xp.astype(float); yp = yp.astype(float); zp = zp.astype(float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dx = xp - X[i, j]; dy = yp - Y[i, j]
            dist2 = dx*dx + dy*dy
            if np.any(dist2 < 1e-14):
                Zi[i, j] = zp[np.argmin(dist2)]
            else:
                w = 1.0 / np.power(dist2 + eps, power/2.0)
                Zi[i, j] = np.sum(w * zp) / np.sum(w)
    return Zi

def rbf_gaussian_fit(xp, yp, zp, eps=0.12, lam=1e-6):
    """Fit Gaussian RBF: K_ij = exp(-eps^2 ||x_i - x_j||^2), (K + lam I) w = z"""
    P = np.stack([xp, yp], axis=1).astype(float)
    d2 = np.sum((P[:,None,:]-P[None,:,:])**2, axis=2)
    K = np.exp(-(eps**2) * d2) + lam*np.eye(len(xp))
    w = npl.solve(K, zp.astype(float))
    return w, P, eps

def rbf_gaussian_eval(w, P, eps, X, Y):
    pts = np.stack([X.ravel(), Y.ravel()], axis=1).astype(float)
    d2 = np.sum((pts[:,None,:]-P[None,:,:])**2, axis=2)
    Phi = np.exp(-(eps**2) * d2)     # (M x N)
    Z = Phi @ w
    return Z.reshape(X.shape)

def laplacian_log_omega(logOm, x, y):
    # 2nd-order central differences; periodic wrap not assumed.
    dx = float(np.mean(np.diff(x))); dy = float(np.mean(np.diff(y)))
    d2x = (np.roll(logOm, -1, axis=1) - 2*logOm + np.roll(logOm, +1, axis=1)) / (dx*dx)
    d2y = (np.roll(logOm, -1, axis=0) - 2*logOm + np.roll(logOm, +1, axis=0)) / (dy*dy)
    return d2x + d2y

def gaussian_matter(X, Y, centers, masses, sigma_m):
    rho = np.zeros_like(X, dtype=float)
    s2 = sigma_m*sigma_m
    for (cx, cy), m in zip(centers, masses):
        rho += m * np.exp(-((X-cx)**2 + (Y-cy)**2)/(2*s2))
    rho /= (2*np.pi*s2)  # keep total weight ~sum(masses)
    return rho

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Einstein residual check on torus slice from T_eff CSVs")
    ap.add_argument("--torsion_csv", required=True, help="CSV with columns: species, ax, ay, t_eff (case-insensitive)")
    ap.add_argument("--locked_csv", required=True, help="all_particles_locked.csv (species, m_GeV). Positions taken from torsion_csv.")
    ap.add_argument("--grid_n", type=int, default=320)
    ap.add_argument("--alpha", type=float, default=1.0, help="Omega = exp(alpha * T_eff)")
    ap.add_argument("--sigma_m", type=float, default=0.030, help="matter Gaussian width on (a_x,a_y)")

    # NEW: interpolation controls
    ap.add_argument("--interp", choices=["rbf","idw"], default="rbf")
    ap.add_argument("--rbf_eps", type=float, default=0.12, help="Gaussian RBF width in (a_x,a_y)")
    ap.add_argument("--rbf_lambda", type=float, default=1e-6, help="Tikhonov regularization")

    # Optional parameter pins
    ap.add_argument("--fix_beta", type=float, default=None, help="Fix beta (phi=beta*T_eff)")
    ap.add_argument("--fix_kappa", type=float, default=None, help="Fix kappa in R2 ≈ kappa * rho")
    ap.add_argument("--fix_mscale", type=float, default=None, help="Fix matter normalisation")
    ap.add_argument("--outdir", default="einstein_check_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Load torsion map ---
    tor = pd.read_csv(args.torsion_csv)
    tor.columns = [c.lower() for c in tor.columns]
    req = ["species","ax","ay","t_eff"]
    for r in req:
        if r not in tor.columns:
            raise ValueError(f"torsion_csv must include '{r}'")
    xp = tor.ax.to_numpy(float); yp = tor.ay.to_numpy(float); zp = tor.t_eff.to_numpy(float)

    # --- Grid + T_eff field ---
    x, y, X, Y = build_grid(xp, yp, args.grid_n)
    if args.interp == "idw":
        Tgrid = idw_interp(xp, yp, zp, X, Y, power=2)
    else:
        w, P, eps = rbf_gaussian_fit(xp, yp, zp, eps=args.rbf_eps, lam=args.rbf_lambda)
        Tgrid = rbf_gaussian_eval(w, P, eps, X, Y)

    # --- Geometry: Omega and R2 for conformal metric g_ij = Omega^2 delta_ij ---
    logOm = args.alpha * Tgrid
    Omega = np.exp(logOm)
    lap = laplacian_log_omega(logOm, x, y)
    R2 = -2.0 * (lap / (Omega*Omega))

    # --- Scalar field energy from phi = beta * T_eff on the slice ---
    dx = float(np.mean(np.diff(x))); dy = float(np.mean(np.diff(y)))
    dTx = (np.roll(Tgrid, -1, axis=1) - np.roll(Tgrid, +1, axis=1)) / (2*dx)
    dTy = (np.roll(Tgrid, -1, axis=0) - np.roll(Tgrid, +1, axis=0)) / (2*dy)
    dT2 = dTx*dTx + dTy*dTy
    OmInv2 = 1.0/(Omega*Omega)

    # --- Matter from locked CSV (positions from torsion file) ---
    lock = pd.read_csv(args.locked_csv)
    lock.columns = [c.lower() for c in lock.columns]
    ax_map = dict(zip(tor["species"].astype(str), zip(tor["ax"], tor["ay"])))
    centers, masses = [], []
    if "m_gev" in lock.columns:
        for s, m in zip(lock["species"], lock["m_gev"]):
            s = str(s)
            if s in ax_map:
                centers.append(ax_map[s]); masses.append(float(m))
    else:
        for s in lock["species"]:
            s = str(s)
            if s in ax_map:
                centers.append(ax_map[s]); masses.append(1.0)
    if not centers:
        raise ValueError("No overlapping species between torsion_csv and locked_csv")

    rho_m0 = gaussian_matter(X, Y, centers, np.array(masses, dtype=float), sigma_m=args.sigma_m)

    # --- Least-squares fit for (beta^2, mscale), then kappa ---
    R = R2.ravel()
    A_phi = 0.5 * OmInv2.ravel() * dT2.ravel()   # multiplies beta^2
    A_m   = rho_m0.ravel()                       # multiplies mscale

    Ny, Nx = R2.shape
    edge = np.zeros_like(R2, dtype=bool); edge[0,:]=edge[-1,:]=edge[:,0]=edge[:,-1]=True
    mask = (~edge).ravel()

    Rm, A_phim, A_mm = R[mask], A_phi[mask], A_m[mask]
    Rn = Rm - Rm.mean(); sd = Rn.std() if Rn.std()>0 else 1.0; Rn /= sd
    Xmat = np.vstack([A_phim, A_mm]).T
    col_sd = np.maximum(np.std(Xmat, axis=0), 1e-12)
    Xn = Xmat / col_sd

    fix_beta, fix_kappa, fix_m = args.fix_beta, args.fix_kappa, args.fix_mscale
    if fix_beta is None or fix_m is None:
        coeff, *_ = npl.lstsq(Xn, Rn, rcond=None)
        c_phi, c_m = coeff
        c_phi *= col_sd[0]; c_m *= col_sd[1]
        beta_sq_fit = max(1e-12, c_phi)
        mscale_fit  = max(1e-12, c_m)
        if fix_beta is not None: beta_sq_fit = fix_beta*fix_beta
        if fix_m    is not None: mscale_fit  = fix_m
    else:
        beta_sq_fit = fix_beta*fix_beta
        mscale_fit  = fix_m

    rho_phi = 0.5 * OmInv2 * beta_sq_fit * dT2
    rho_tot = rho_phi + mscale_fit * rho_m0

    Rt = R2[~edge]; rhot = rho_tot[~edge]
    num = np.sum(Rt * rhot); den = np.sum(rhot * rhot)
    kappa_fit = num/den if den>0 else 0.0
    if fix_kappa is not None: kappa_fit = fix_kappa

    residual = R2 - kappa_fit * rho_tot
    rms = np.sqrt(np.mean(residual[~edge]**2))
    rel = rms / (np.sqrt(np.mean(R2[~edge]**2)) + 1e-12)

    # ----------------------------
    # Save results
    # ----------------------------
    summary = {
        "grid_n": args.grid_n,
        "alpha": args.alpha,
        "sigma_m": args.sigma_m,
        "interp": args.interp,
        "rbf_eps": args.rbf_eps,
        "rbf_lambda": args.rbf_lambda,
        "beta_fit": float(np.sqrt(beta_sq_fit)),
        "kappa_fit": float(kappa_fit),
        "mscale_fit": float(mscale_fit),
        "rms_residual": float(rms),
        "relative_rms": float(rel)
    }
    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame([summary]).to_csv(os.path.join(args.outdir, "einstein_residual_summary.csv"), index=False)

    np.savetxt(os.path.join(args.outdir, "R2_field.csv"), R2, delimiter=",")
    np.savetxt(os.path.join(args.outdir, "rho_tot_field.csv"), rho_tot, delimiter=",")
    np.savetxt(os.path.join(args.outdir, "residual_field.csv"), residual, delimiter=",")

    # Plots
    plt.figure(figsize=(6,5))
    plt.imshow(R2, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
    plt.colorbar(label="R2 (2D Ricci scalar)")
    plt.title(r"Geometric curvature R2 from $\Omega=\exp(\alpha T_{\rm eff})$")
    plt.xlabel("a_x"); plt.ylabel("a_y")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_R2.png"), dpi=300)

    plt.figure(figsize=(6,5))
    plt.imshow(rho_tot, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
    plt.colorbar(label=r"$\rho_{\rm tot}=\rho_\phi + m_{\rm scale}\,\rho_{\rm matter}$")
    plt.title("Effective energy density on the torus slice")
    plt.xlabel("a_x"); plt.ylabel("a_y")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_rho_tot.png"), dpi=300)

    plt.figure(figsize=(6,5))
    plt.imshow(residual, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
    plt.colorbar(label="Residual  (R2 - κ ρ)")
    plt.title(f"Einstein residual (rel. RMS = {rel:.3e})")
    plt.xlabel("a_x"); plt.ylabel("a_y")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_residual.png"), dpi=300)

    print("=== Einstein residual check ===")
    for k,v in summary.items():
        print(f"{k:>14s}: {v}")
    print("Saved to:", args.outdir)

if __name__ == "__main__":
    main()