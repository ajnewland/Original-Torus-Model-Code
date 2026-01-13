#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute ADM momentum constraint from two torus slices.
"""

import argparse, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

def central_grad(F, x, y, periodic=True):
    dx=float(np.mean(np.diff(x))); dy=float(np.mean(np.diff(y)))
    if periodic:
        dFx=(np.roll(F,-1,1)-np.roll(F,1,1))/(2*dx)
        dFy=(np.roll(F,-1,0)-np.roll(F,1,0))/(2*dy)
    else:
        dFx=np.empty_like(F); dFy=np.empty_like(F)
        dFx[:,1:-1]=(F[:,2:]-F[:,:-2])/(2*dx); dFx[:,0]=dFx[:,1]; dFx[:,-1]=dFx[:,-2]
        dFy[1:-1,:]=(F[2:,:]-F[:-2,:])/(2*dy); dFy[0,:]=dFy[1,:]; dFy[-1,:]=dFy[-2,:]
    return dFx, dFy

def div_tensor_2d(Txx, Txy, Tyx, Tyy, x, y, Gamma, periodic=True):
    dx=float(np.mean(np.diff(x))); dy=float(np.mean(np.diff(y)))
    dTxx_dx = (np.roll(Txx,-1,1)-np.roll(Txx,1,1))/(2*dx)
    dTxx_dy = (np.roll(Txx,-1,0)-np.roll(Txx,1,0))/(2*dy)
    dTxy_dx = (np.roll(Txy,-1,1)-np.roll(Txy,1,1))/(2*dx)
    dTxy_dy = (np.roll(Txy,-1,0)-np.roll(Txy,1,0))/(2*dy)
    dTyx_dx = (np.roll(Tyx,-1,1)-np.roll(Tyx,1,1))/(2*dx)
    dTyx_dy = (np.roll(Tyx,-1,0)-np.roll(Tyx,1,0))/(2*dy)
    dTyy_dx = (np.roll(Tyy,-1,1)-np.roll(Tyy,1,1))/(2*dx)
    dTyy_dy = (np.roll(Tyy,-1,0)-np.roll(Tyy,1,0))/(2*dy)

    Gxxx,Gxxy,Gxyx,Gxyy, Gyxx,Gyxy,Gyyx,Gyyy = Gamma

    Cx = (dTxx_dx + dTxy_dy
          + Gxxx*Txx + Gxxy*Tyx + Gxyx*Txy + Gxyy*Tyy
          - (Gxxx+Gyxx)*Txx - (Gxxy+Gyxy)*Txy)
    Cy = (dTyx_dx + dTyy_dy
          + Gyxx*Txx + Gyxy*Tyx + Gyyx*Txy + Gyyy*Tyy
          - (Gxxx+Gyxx)*Tyx - (Gxxy+Gyxy)*Tyy)
    return Cx, Cy

def main():
    ap=argparse.ArgumentParser("Momentum constraint from two slices")
    ap.add_argument("--h5_t0", required=True)
    ap.add_argument("--h5_t1", required=True)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--lapse", type=float, default=1.0)
    ap.add_argument("--outdir", default="momentum_check_out")
    args=ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with h5py.File(args.h5_t0,"r") as f0, h5py.File(args.h5_t1,"r") as f1:
        x=f0["x"][:]; y=f0["y"][:]
        Om0=f0["Omega"][:]; Om1=f1["Omega"][:]

    Om_dot = (Om1-Om0)/args.dt
    N = args.lapse
    Kxx = -(Om0*Om_dot)/N
    Kyy = Kxx.copy()
    Kxy = np.zeros_like(Kxx)
    Kyx = np.zeros_like(Kxx)

    ginv = 1.0/(Om0*Om0)
    K = ginv*(Kxx+Kyy)

    Kxx_c = ginv*ginv*Kxx
    Kyy_c = ginv*ginv*Kyy
    Kxy_c = np.zeros_like(Kxx)
    Kyx_c = np.zeros_like(Kxx)

    Sxx = Kxx_c - ginv*K
    Syy = Kyy_c - ginv*K
    Sxy = Kxy_c
    Syx = Kyx_c

    lnOm = np.log(Om0 + 1e-30)
    dln_dx, dln_dy = central_grad(lnOm, x, y, periodic=True)

    Gxxx = dln_dx; Gxxy = 0.0*dln_dx; Gxyx = dln_dy; Gxyy = -dln_dx
    Gyxx = -dln_dy; Gyxy = dln_dx; Gyyx = 0.0*dln_dx; Gyyy = dln_dy
    Gamma = (Gxxx,Gxxy,Gxyx,Gxyy, Gyxx,Gyxy,Gyyx,Gyyy)

    Cx, Cy = div_tensor_2d(Sxx, Sxy, Syx, Syy, x, y, Gamma, periodic=True)

    Cnorm = (Om0**2) * (Cx*Cx + Cy*Cy)

    mask = np.ones_like(Cnorm,bool); mask[0,:]=mask[-1,:]=mask[:,0]=mask[:,-1]=False
    rms = float(np.sqrt(np.mean(Cnorm[mask])))

    print("=== Momentum constraint check ===")
    print(f" dt={args.dt}, lapse N={N}")
    print(f" RMS(|C|) = {rms:.3e}")

    def imsave(arr, title, fname):
        plt.figure(figsize=(6,5))
        plt.imshow(arr, origin='lower', extent=[x.min(),x.max(),y.min(),y.max()])
        plt.colorbar()
        plt.title(title); plt.xlabel("a_x"); plt.ylabel("a_y")
        plt.tight_layout(); plt.savefig(fname, dpi=300); plt.close()

    pd.DataFrame([dict(dt=args.dt,lapse=N,rms_C=rms)]).to_csv(
        os.path.join(args.outdir,"momentum_summary.csv"), index=False)

    imsave(Cnorm, r"$|\mathcal{C}| = \sqrt{g_{ij} \mathcal{C}^i \mathcal{C}^j}$",
           os.path.join(args.outdir,"fig_Cnorm.png"))

    print("Saved ->", args.outdir)

if __name__=="__main__":
    main()