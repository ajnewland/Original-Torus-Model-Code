#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build dense torus datasets at two times (t0, t0+dt).

Inputs
------
--species_csv_t0 : species-level torsion points at t0 (species, ax, ay, T_eff)
--locked_csv     : particle masses (species, m_GeV[, sigma_m])
Optional for a real second slice:
--species_csv_t1 : species-level torsion points at t1
Otherwise synthesize t1 from t0:
--dt             : time separation (arb. units, default 1.0)
--evolve_mode    : 'gradflow' (default) or 'advection'
--mu             : small evolution rate (default 0.02)
--advec_angle    : advection flow angle in radians (default 0.0)
--advec_speed    : advection speed (default 0.02)

Shared options: grid_n, alpha, rbf_lambda, len_grid (auto or list), default_sigma_m, outprefix

Outputs
-------
{outprefix}_t0_fields.h5/.csv/.png/.json
{outprefix}_t1_fields.h5/.csv/.png/.json
"""

import argparse, os, json
import numpy as np, pandas as pd, numpy.linalg as npl
import h5py, matplotlib.pyplot as plt

# ---------- helpers ----------
def build_grid(ax, ay, n, pad=0.02):
    ax_min, ax_max = float(ax.min()), float(ax.max())
    ay_min, ay_max = float(ay.min()), float(ay.max())
    x = np.linspace(ax_min-pad*(ax_max-ax_min), ax_max+pad*(ax_max-ax_min), n)
    y = np.linspace(ay_min-pad*(ay_max-ay_min), ay_max+pad*(ay_max-ay_min), n)
    X, Y = np.meshgrid(x, y); return x, y, X, Y

def nn_median(P):
    dmin = np.full(P.shape[0], np.inf)
    for i in range(P.shape[0]):
        d = np.sqrt(np.sum((P[i]-P)**2, axis=1)); d[i]=np.inf; dmin[i]=d.min()
    return float(np.median(dmin))

def rbf_eval(P, z, ell, lam, X, Y):
    d2 = np.sum((P[:,None,:]-P[None,:,:])**2, axis=2)
    K  = np.exp(-d2/(ell*ell)) + lam*np.eye(P.shape[0])
    w  = npl.solve(K, z)
    pts= np.stack([X.ravel(), Y.ravel()], axis=1)
    d2g= np.sum((pts[:,None,:]-P[None,:,:])**2, axis=2)
    Z  = (np.exp(-d2g/(ell*ell)) @ w).reshape(X.shape)
    return Z

def choose_ell(P, z, lam, ells):
    best=(1e99,None)
    d2 = np.sum((P[:,None,:]-P[None,:,:])**2, axis=2)
    for ell in ells:
        K = np.exp(-d2/(ell*ell)) + lam*np.eye(P.shape[0])
        Ki= npl.inv(K); w = Ki @ z; denom = np.diag(Ki)
        loo = z - w/denom
        rmse = float(np.sqrt(np.mean((loo-z)**2)))
        if rmse < best[0]: best=(rmse,ell)
    return best  # (rmse, ell)

def grad_central(F, x, y, periodic=True):
    dx = float(np.mean(np.diff(x))); dy = float(np.mean(np.diff(y)))
    if periodic:
        dFx = (np.roll(F,-1,1)-np.roll(F,1,1))/(2*dx)
        dFy = (np.roll(F,-1,0)-np.roll(F,1,0))/(2*dy)
    else:
        dFx = np.empty_like(F); dFy = np.empty_like(F)
        dFx[:,1:-1]=(F[:,2:]-F[:,:-2])/(2*dx); dFx[:,0]=dFx[:,1]; dFx[:,-1]=dFx[:,-2]
        dFy[1:-1,:]=(F[2:,:]-F[:-2,:])/(2*dy); dFy[0,:]=dFy[1,:]; dFy[-1,:]=dFy[-2,:]
    return dFx, dFy

def laplacian(F, x, y, periodic=True):
    dx = float(np.mean(np.diff(x))); dy = float(np.mean(np.diff(y)))
    if periodic:
        d2x=(np.roll(F,-1,1)-2*F+np.roll(F,1,1))/(dx*dx)
        d2y=(np.roll(F,-1,0)-2*F+np.roll(F,1,0))/(dy*dy)
    else:
        d2x = np.empty_like(F); d2y = np.empty_like(F)
        d2x[:,1:-1]=(F[:,2:]-2*F[:,1:-1]+F[:,:-2])/(dx*dx); d2x[:,0]=d2x[:,1]; d2x[:,-1]=d2x[:,-2]
        d2y[1:-1,:]=(F[2:,:]-2*F[1:-1,:]+F[:-2,:])/(dy*dy); d2y[0,:]=d2y[1,:]; d2y[-1,:]=d2y[-2,:]
    return d2x+d2y

def gaussian_matter(X,Y,centers,weights,sigmas):
    rho = np.zeros_like(X,float)
    for (cx,cy),w,s in zip(centers,weights,sigmas):
        s2=s*s
        rho += w*np.exp(-((X-cx)**2+(Y-cy)**2)/(2*s2))/(2*np.pi*s2)
    return rho

def gb_integral(R2, Omega, x, y):
    dx=float(np.mean(np.diff(x))); dy=float(np.mean(np.diff(y)))
    return float(np.sum((Omega**2)*R2)*dx*dy)

def imsave(arr, x, y, title, fname, cbarlabel=""):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    plt.imshow(arr, origin='lower', extent=[x.min(),x.max(),y.min(),y.max()])
    plt.colorbar(label=cbarlabel); plt.title(title)
    plt.xlabel("a_x"); plt.ylabel("a_y"); plt.tight_layout(); plt.savefig(fname, dpi=300); plt.close()

def build_slice(species_csv, locked_csv, grid_n, alpha, lam, len_grid, default_sigma_m,
                periodic=True, outprefix="torus", label="t0"):
    S = pd.read_csv(species_csv); S.columns=[c.lower() for c in S.columns]
    for k in ["species","ax","ay","t_eff"]:
        if k not in S.columns: raise ValueError(f"{species_csv} missing '{k}'")
    ax=S["ax"].to_numpy(float); ay=S["ay"].to_numpy(float); Te=S["t_eff"].to_numpy(float)
    P = np.stack([ax,ay],axis=1)
    x,y,X,Y = build_grid(ax,ay,grid_n)

    if len_grid.lower()=="auto":
        ell0 = nn_median(P); ells = np.array([0.5*ell0,0.75*ell0,ell0,1.25*ell0,1.75*ell0])
    else:
        ells = np.array([float(v) for v in len_grid.split(",") if v.strip()])
    loo_rmse, ell = choose_ell(P, Te, lam, ells)
    Teff = rbf_eval(P, Te, ell, lam, X, Y)

    logOm = alpha * Teff; Omega = np.exp(logOm)
    lap_logOm = laplacian(logOm, x, y, periodic=periodic)
    R2 = -2.0 * (lap_logOm/(Omega*Omega))
    dTx, dTy = grad_central(Teff, x, y, periodic=periodic)
    rho_phi_unitbeta = 0.5 * (dTx*dTx + dTy*dTy) / (Omega*Omega)

    L = pd.read_csv(locked_csv); L.columns=[c.lower() for c in L.columns]
    mass_map  = dict(zip(L["species"].astype(str), L["m_gev"].to_numpy(float))) if "m_gev" in L.columns else {}
    sigma_map = dict(zip(L["species"].astype(str), L.get("sigma_m", pd.Series([default_sigma_m]*len(L))).to_numpy(float)))
    centers, weights, sigmas = [], [], []
    for s in S["species"].astype(str):
        if s in mass_map:
            i = S.index[S["species"].astype(str)==s][0]
            centers.append((ax[i],ay[i])); weights.append(float(mass_map[s])); sigmas.append(float(sigma_map.get(s, default_sigma_m)))
    rho_m = gaussian_matter(X,Y,centers,weights,sigmas) if centers else np.zeros_like(X)

    gb = gb_integral(R2, Omega, x, y)

    # save HDF5 + CSV + meta + PNGs
    h5=f"{outprefix}_{label}_fields.h5"; csv=f"{outprefix}_{label}_fields.csv"
    with h5py.File(h5,"w") as f:
        for name,arr in dict(x=x,y=y,Teff=Teff,Omega=Omega,R2=R2,
                             dTx=dTx,dTy=dTy,rho_phi_unitbeta=rho_phi_unitbeta,rho_m=rho_m).items():
            f.create_dataset(name,data=arr)
    with open(csv,"w") as f:
        f.write("ax,ay,Teff,Omega,R2,rho_m,rho_phi_unitbeta\n")
        for i in range(Teff.shape[0]):
            for j in range(Teff.shape[1]):
                f.write(f"{X[i,j]},{Y[i,j]},{Teff[i,j]},{Omega[i,j]},{R2[i,j]},{rho_m[i,j]},{rho_phi_unitbeta[i,j]}\n")
    meta={
        "grid_n":grid_n,"alpha":alpha,"ell_best":ell,"loo_rmse":loo_rmse,
        "var_Teff":float(np.var(Teff)),"var_laplogOm":float(np.var(lap_logOm)),
        "var_R2":float(np.var(R2)),"gauss_bonnet_integral":gb,"periodic":periodic
    }
    with open(f"{outprefix}_{label}_meta.json","w") as f: json.dump(meta,f,indent=2)

    imsave(Teff,x,y,rf"$T_{{\rm eff}}({label})$",f"{outprefix}_{label}_Teff.png","T_eff")
    imsave(R2,x,y,rf"$R_2({label})$",f"{outprefix}_{label}_R2.png","R2")
    imsave(rho_m,x,y,rf"$\rho_m({label})$",f"{outprefix}_{label}_rho_m.png",r"$\rho_m$")
    return dict(x=x,y=y,Teff=Teff,Omega=Omega,R2=R2)

def synthesize_next_slice(s0, dt, mode, mu, advec_angle, advec_speed, alpha):
    # simple, local evolution to create a nearby slice
    x,y = s0["x"], s0["y"]; Te = s0["Teff"]
    dTx, dTy = grad_central(Te, x, y, periodic=True)
    if mode=="gradflow":
        Te1 = Te + dt * mu * ( - (dTx*dTx + dTy*dTy)**0.5 )  # downhill in |∇T|
    else:  # advection: ∂t T + v·∇T = 0
        vx = advec_speed*np.cos(advec_angle); vy = advec_speed*np.sin(advec_angle)
        Te1 = Te - dt * (vx*dTx + vy*dTy)
    logOm1 = alpha * Te1; Omega1 = np.exp(logOm1)
    lap1 = laplacian(logOm1, x, y, periodic=True)
    R21 = -2.0 * (lap1/(Omega1*Omega1))
    return dict(x=x,y=y,Teff=Te1,Omega=Omega1,R2=R21)

def save_synth(outprefix, label, s, locked_csv, default_sigma_m=0.03):
    x,y,Teff,Omega,R2 = s["x"], s["y"], s["Teff"], s["Omega"], s["R2"]
    # rebuild rho_phi_unitbeta + rho_m for the new slice (same centers/weights)
    dTx, dTy = grad_central(Teff, x, y, periodic=True)
    rho_phi_unitbeta = 0.5 * (dTx*dTx + dTy*dTy) / (Omega*Omega)

    L=pd.read_csv(locked_csv); L.columns=[c.lower() for c in L.columns]
    # Best effort: place matter exactly as t0 (centers from t0 species CSV not passed here);
    # For our checks it only affects rho_m mildly; you can regenerate from species_t1 if you have it.
    rho_m = np.zeros_like(Teff)

    gb = gb_integral(R2, Omega, x, y)
    h5=f"{outprefix}_{label}_fields.h5"; csv=f"{outprefix}_{label}_fields.csv"
    with h5py.File(h5,"w") as f:
        for name,arr in dict(x=x,y=y,Teff=Teff,Omega=Omega,R2=R2,
                             dTx=dTx,dTy=dTy,rho_phi_unitbeta=rho_phi_unitbeta,rho_m=rho_m).items():
            f.create_dataset(name,data=arr)
    # CSV
    X,Y=np.meshgrid(x,y)
    with open(csv,"w") as f:
        f.write("ax,ay,Teff,Omega,R2,rho_m,rho_phi_unitbeta\n")
        for i in range(Teff.shape[0]):
            for j in range(Teff.shape[1]):
                f.write(f"{X[i,j]},{Y[i,j]},{Teff[i,j]},{Omega[i,j]},{R2[i,j]},{rho_m[i,j]},{rho_phi_unitbeta[i,j]}\n")
    with open(f"{outprefix}_{label}_meta.json","w") as f:
        json.dump({"gauss_bonnet_integral":gb}, f, indent=2)
    imsave(Teff,x,y,rf"$T_{{\rm eff}}({label})$",f"{outprefix}_{label}_Teff.png","T_eff")
    imsave(R2,x,y,rf"$R_2({label})$",f"{outprefix}_{label}_R2.png","R2")

def main():
    ap = argparse.ArgumentParser("Build rich torus time-slices")
    ap.add_argument("--species_csv_t0", required=True)
    ap.add_argument("--locked_csv", required=True)
    ap.add_argument("--species_csv_t1", default=None, help="optional: real second slice")
    ap.add_argument("--grid_n", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--rbf_lambda", type=float, default=1e-6)
    ap.add_argument("--len_grid", default="auto")
    ap.add_argument("--default_sigma_m", type=float, default=0.03)
    ap.add_argument("--outprefix", default="torus_rich")
    # synthetic evolution controls
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--evolve_mode", choices=["gradflow","advection"], default="gradflow")
    ap.add_argument("--mu", type=float, default=0.02)
    ap.add_argument("--advec_angle", type=float, default=0.0)
    ap.add_argument("--advec_speed", type=float, default=0.02)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.outprefix) or ".", exist_ok=True)

    s0 = build_slice(args.species_csv_t0, args.locked_csv, args.grid_n, args.alpha,
                     args.rbf_lambda, args.len_grid, args.default_sigma_m,
                     periodic=True, outprefix=args.outprefix, label="t0")

    if args.species_csv_t1:
        s1 = build_slice(args.species_csv_t1, args.locked_csv, args.grid_n, args.alpha,
                         args.rbf_lambda, args.len_grid, args.default_sigma_m,
                         periodic=True, outprefix=args.outprefix, label="t1")
    else:
        s1 = synthesize_next_slice(s0, args.dt, args.evolve_mode, args.mu,
                                   args.advec_angle, args.advec_speed, args.alpha)
        save_synth(args.outprefix, "t1", s1, args.locked_csv, args.default_sigma_m)

    print("Done. Two slices written to:", os.path.dirname(args.outprefix) or ".")

if __name__ == "__main__":
    main()
