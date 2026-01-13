#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter search for torus geometry
UTF-8 safe, includes:
- Intercept (c0) in Einstein fit
- Laplacian (five/nine)
- Gaussian smoothing
- Mask margin
- Ω normalization to unit mean
"""

import argparse, os, sys
import numpy as np, h5py, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------- Helpers ----------
def central_grad(F, x, y, periodic=True):
    dx=float(np.mean(np.diff(x))); dy=float(np.mean(np.diff(y)))
    if periodic:
        dFx=(np.roll(F,-1,1)-np.roll(F,1,1))/(2*dx)
        dFy=(np.roll(F,-1,0)-np.roll(F,1,0))/(2*dy)
    else:
        dFx=np.empty_like(F); dFy=np.empty_like(F)
        dFx[:,1:-1]=(F[:,2:]-F[:,:-2])/(2*dx); dFx[:,0]=dFx[:,1]; dFx[:,-1]=dFx[:,-2]
        dFy[1:-1,:]=(F[2:,:]-F[:-2,:])/(2*dy); dFy[0,:]=dFy[1,:]; dFy[-1,:]=dFy[-2,:]
    return dFx,dFy

def laplacian_5pt(F,x,y):
    dx=float(np.mean(np.diff(x))); dy=float(np.mean(np.diff(y)))
    d2x=(np.roll(F,-1,1)-2*F+np.roll(F,1,1))/(dx*dx)
    d2y=(np.roll(F,-1,0)-2*F+np.roll(F,1,0))/(dy*dy)
    return d2x+d2y

def laplacian_9pt(F,x,y):
    h = float(0.5*(np.mean(np.diff(x)) + np.mean(np.diff(y))))
    fC=F
    fN=np.roll(F,-1,0); fS=np.roll(F,1,0)
    fE=np.roll(F,-1,1); fW=np.roll(F,1,1)
    fNE=np.roll(fN,-1,1); fNW=np.roll(fN,1,1)
    fSE=np.roll(fS,-1,1); fSW=np.roll(fS,1,1)
    return (-20*fC + 4*(fN+fS+fE+fW) + (fNE+fNW+fSE+fSW))/(6*h*h)

def gaussian1d_kernel(sigma_px,radius=None):
    if sigma_px<=0: return None
    if radius is None: radius=int(max(3,round(3*sigma_px)))
    x=np.arange(-radius,radius+1)
    k=np.exp(-0.5*(x/sigma_px)**2); k/=k.sum()
    return k

def gaussian_smooth2d(F,sigma_px):
    k=gaussian1d_kernel(sigma_px)
    if k is None: return F
    pad=len(k)//2
    G=F.copy()
    for j in range(F.shape[0]):
        row=np.r_[F[j,-pad:],F[j],F[j,:pad]]
        conv=np.convolve(row,k,mode="same")
        G[j,:]=conv[pad:pad+F.shape[1]]
    H=G.copy()
    for i in range(F.shape[1]):
        col=np.r_[G[-pad:,i],G[:,i],G[:pad,i]]
        conv=np.convolve(col,k,mode="same")
        H[:,i]=conv[pad:pad+F.shape[0]]
    return H

# ---------- Einstein fit ----------
def hamiltonian_fit_with_intercept(R2, rho_phi_unit, rho_m, mask):
    A1=rho_phi_unit[mask].ravel(); A2=rho_m[mask].ravel(); r=R2[mask].ravel()
    X=np.vstack([A1,A2,np.ones_like(A1)]).T
    Xs=X.copy(); cols=[0,1]
    scale=np.maximum(Xs[:,cols].std(0),1e-12)
    Xs[:,cols]/=scale
    rc=r-r.mean(); rs=rc/(rc.std() if rc.std()>0 else 1.0)
    coeff,*_=np.linalg.lstsq(Xs,rs,rcond=None)
    c1s,c2s,c0s=coeff
    r_mean=r.mean(); r_std=r.std() if r.std()>0 else 1.0
    c1=(r_std*c1s)/(scale[0] if scale[0]>0 else 1.0)
    c2=(r_std*c2s)/(scale[1] if scale[1]>0 else 1.0)
    c0=r_mean+r_std*c0s
    beta2=max(0.0,c1); mscale=max(0.0,c2)
    rho_tot=beta2*rho_phi_unit+mscale*rho_m
    rt,rhot=R2[mask],rho_tot[mask]
    kappa=float(((rt-c0)*rhot).sum()/(rhot*rhot).sum())
    resid=R2-(kappa*rho_tot+c0)
    ein_rms=float(np.sqrt(np.mean(resid[mask]**2)))
    ein_rel=float(ein_rms/(np.sqrt(np.mean(R2[mask]**2))+1e-12))
    return beta2,mscale,kappa,c0,resid,ein_rms,ein_rel

# ---------- Core ----------
def compute_metrics_for(alpha,dt,mu,x,y,T0,rho_m,lapse,
                        laplace_kind="five",smooth_px=0.0,mask_margin=1):
    mask=np.ones_like(T0,bool)
    mm=int(mask_margin)
    mask[:mm,:]=mask[-mm:,:]=mask[:,:mm]=mask[:,-mm:]=False

    lnOm0=alpha*T0
    if smooth_px>0.0: lnOm0=gaussian_smooth2d(lnOm0,smooth_px)
    Om0=np.exp(lnOm0)
    Om0=Om0/(Om0.mean()+1e-30)   # normalization

    lap0=laplacian_9pt(lnOm0,x,y) if laplace_kind=="nine" else laplacian_5pt(lnOm0,x,y)
    R2=-2.0*lap0/(Om0*Om0)
    dTdx0,dTdy0=central_grad(T0,x,y)
    rho_phi_unit=0.5*(dTdx0*dTdx0+dTdy0*dTdy0)/(Om0*Om0)
    beta2,mscale,kappa,c0,ein_resid,ein_rms,ein_rel=hamiltonian_fit_with_intercept(R2,rho_phi_unit,rho_m,mask)

    grad_norm=np.sqrt(dTdx0*dTdx0+dTdy0*dTdy0)
    T1=T0+dt*mu*(-grad_norm)
    lnOm1=alpha*T1
    if smooth_px>0.0: lnOm1=gaussian_smooth2d(lnOm1,smooth_px)
    Om1=np.exp(lnOm1)
    Om1=Om1/(Om1.mean()+1e-30)   # normalization

    Om_dot=(Om1-Om0)/dt; N=lapse
    Kxx=-(Om0*Om_dot)/N; Kyy=Kxx.copy()
    ginv=1.0/(Om0*Om0); K=ginv*(Kxx+Kyy)
    dKdx,dKdy=central_grad(K,x,y)
    gradK_norm=(Om0**2)*(dKdx**2+dKdy**2)
    mom_rms_abs=float(np.sqrt(np.mean(gradK_norm[mask])))
    mom_rms_ref=float(np.sqrt(np.mean(gradK_norm[mask]))+1e-12)
    mom_rel=float(mom_rms_abs/mom_rms_ref)
    mom_rms_abs_src=mom_rms_abs; mom_rel_src=mom_rel
    score=float(np.sqrt(ein_rel**2+mom_rel_src**2))
    return dict(alpha=alpha,dt=dt,mu=mu,beta=np.sqrt(beta2),kappa=kappa,
                mscale=mscale,c0=c0,ein_rms=ein_rms,ein_rel=ein_rel,
                mom_rms_abs=mom_rms_abs,mom_rel=mom_rel,
                mom_rms_abs_src=mom_rms_abs_src,mom_rel_src=mom_rel_src,score=score)

# ---------- Main ----------
def main():
    ap=argparse.ArgumentParser("Parameter search (UTF-8 safe, normalized Ω)")
    ap.add_argument("--h5_t0",required=True)
    ap.add_argument("--alpha_list",required=True)
    ap.add_argument("--dt_list",required=True)
    ap.add_argument("--mu_list",required=True)
    ap.add_argument("--lapse",type=float,default=1.0)
    ap.add_argument("--laplace",choices=["five","nine"],default="nine")
    ap.add_argument("--smooth_px",type=float,default=1.6)
    ap.add_argument("--mask_margin",type=int,default=8)
    ap.add_argument("--outdir",default="torus_param_search_out")
    args=ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)
    with h5py.File(args.h5_t0,"r") as f0:
        x=f0["x"][:]; y=f0["y"][:]; T0=f0["Teff"][:]; rho_m=f0["rho_m"][:]
    alphas=[float(v) for v in args.alpha_list.split(",") if v.strip()]
    dts=[float(v) for v in args.dt_list.split(",") if v.strip()]
    mus=[float(v) for v in args.mu_list.split(",") if v.strip()]
    results=[]; best=None
    for a in alphas:
        for dt in dts:
            for mu in mus:
                out=compute_metrics_for(a,dt,mu,x,y,T0,rho_m,args.lapse,
                                        laplace_kind=args.laplace,
                                        smooth_px=args.smooth_px,
                                        mask_margin=args.mask_margin)
                results.append(out)
                if (best is None) or (out["score"]<best["score"]): best=out
    df=pd.DataFrame(results); df.sort_values("score",inplace=True)
    csv=os.path.join(args.outdir,"search_results.csv"); df.to_csv(csv,index=False)
    with open(os.path.join(args.outdir,"best_summary.txt"),"w",encoding="utf-8") as f:
        f.write(f"=== Best (alpha, dt, mu) ===\nalpha={best['alpha']}\n"
                f"dt={best['dt']}\nmu={best['mu']}\n\n"
                f"β≈{best['beta']:.4g}, κ≈{best['kappa']:.4g}, "
                f"mscale≈{best['mscale']:.4g}, c0≈{best['c0']:.4g}\n"
                f"ein_rel={best['ein_rel']:.4g}, mom_rel_src={best['mom_rel_src']:.4g}, "
                f"SCORE={best['score']:.4g}\n")
    print(open(os.path.join(args.outdir,"best_summary.txt"),encoding="utf-8").read())

if __name__=="__main__":
    main()
