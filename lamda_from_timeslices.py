#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometric Λ proxy from a single torus time-slice pair (DIRECT Hamiltonian fit)

Outputs (in --outdir):
  summary.csv                  # A, B, C (Λ proxy), fit quality, correlations, Λ proxies
  fit_field.csv                # A ρφ + B ρm + C
  residual_field.csv           # R2 - fit
  R2_field.csv, rho_phi.csv, rho_m.csv
  report.pdf                   # one-page metrics + field panels
"""
import argparse, os, sys, json
import numpy as np, h5py, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

def central_grad(F,x,y,periodic=True):
    dx=float(np.mean(np.diff(x))); dy=float(np.mean(np.diff(y)))
    if periodic:
        dFx=(np.roll(F,-1,1)-np.roll(F,1,1))/(2*dx)
        dFy=(np.roll(F,-1,0)-np.roll(F,1,0))/(2*dy)
    else:
        dFx=np.empty_like(F); dFy=np.empty_like(F)
        dFx[:,1:-1]=(F[:,2:]-F[:,:-2])/(2*dx); dFx[:,0]=dFx[:,1]; dFx[:,-1]=dFx[:,-2]
        dFy[1:-1,:]=(F[2:,:]-F[:-2,:])/(2*dy); dFy[0,:]=dFy[1,:]; dFy[-1,:]=dFy[-2,:]
    return dFx,dFy

def laplacian_spectral(F, x, y):
    Ny,Nx=F.shape
    Lx=float(x.max()-x.min()) if x.max()>x.min() else Nx
    Ly=float(y.max()-y.min()) if y.max()>y.min() else Ny
    kx=2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
    ky=2*np.pi*np.fft.fftfreq(Ny, d=Ly/Ny)
    KX,KY=np.meshgrid(kx,ky); k2=KX*KX+KY*KY
    Fhat=np.fft.fft2(F); lap_hat=-k2*Fhat; lap_hat[0,0]=0.0
    return np.fft.ifft2(lap_hat).real

def gaussian1d_kernel(sigma_px,radius=None):
    if sigma_px<=0: return None
    if radius is None: radius=int(max(3, round(3*sigma_px)))
    x=np.arange(-radius, radius+1, dtype=float)
    k=np.exp(-0.5*(x/sigma_px)**2); k/=k.sum()
    return k

def gaussian_smooth2d(F, sigma_px):
    k=gaussian1d_kernel(sigma_px)
    if k is None: return F
    pad=len(k)//2
    G=F.copy()
    for j in range(F.shape[0]):
        row=np.r_[F[j,-pad:],F[j],F[j,:pad]]
        G[j,:]=np.convolve(row,k,mode="same")[pad:pad+F.shape[1]]
    H=G.copy()
    for i in range(F.shape[1]):
        col=np.r_[G[-pad:,i],G[:,i],G[:pad,i]]
        H[:,i]=np.convolve(col,k,mode="same")[pad:pad+F.shape[0]]
    return H

def direct_fit(R2, rho_phi, rho_m, mask):
    r=R2[mask].ravel(); a=rho_phi[mask].ravel(); b=rho_m[mask].ravel()
    X=np.vstack([a,b,np.ones_like(a)]).T
    coef, *_ = np.linalg.lstsq(X, r, rcond=None)
    A,B,C = coef.tolist()
    fit = A*rho_phi + B*rho_m + C
    resid = R2 - fit
    r_mask=R2[mask].ravel(); f_mask=fit[mask].ravel()
    ss_res=float(np.sum((r_mask-f_mask)**2)); ss_tot=float(np.sum((r_mask-r_mask.mean())**2))+1e-30
    R2_score = 1.0 - ss_res/ss_tot
    ein_rms=float(np.sqrt(np.mean(resid[mask]**2)))
    ein_rel=float(ein_rms/(np.sqrt(np.mean(R2[mask]**2))+1e-12))
    # correlations
    def pearson(u,v):
        u=u-u.mean(); v=v-v.mean()
        d=np.sqrt((u*u).sum()*(v*v).sum())+1e-30
        return float((u*v).sum()/d)
    r_rhophi=pearson(r,a); r_rhom=pearson(r,b); rhophi_rhom=pearson(a,b)
    return A,B,C,fit,resid,ein_rms,ein_rel,R2_score,(r_rhophi,r_rhom,rhophi_rhom)

def imshow_ax(fig,ax,arr,x,y,title,clabel=""):
    im=ax.imshow(arr,origin='lower',extent=[x.min(),x.max(),y.min(),y.max()])
    cb=fig.colorbar(im,ax=ax);
    if clabel: cb.set_label(clabel)
    ax.set_title(title); ax.set_xlabel("a_x"); ax.set_ylabel("a_y")

def main():
    ap=argparse.ArgumentParser("Λ proxy from torus time-slices (DIRECT fit)")
    ap.add_argument("--h5_t0", required=True)
    ap.add_argument("--h5_t1", required=True)
    ap.add_argument("--alpha", type=float, default=4.9)
    ap.add_argument("--smooth_px", type=float, default=1.6)
    ap.add_argument("--mask_margin", type=int, default=8)
    ap.add_argument("--outdir", default="lambda_from_slices_out")
    args=ap.parse_args(); os.makedirs(args.outdir, exist_ok=True)

    with h5py.File(args.h5_t0,"r") as f0, h5py.File(args.h5_t1,"r") as f1:
        x=f0["x"][:]; y=f0["y"][:]
        T0=f0["Teff"][:]; T1=f1["Teff"][:]
        rho_m=f0["rho_m"][:]

    mm=int(args.mask_margin)
    mask=np.ones_like(T0,bool)
    mask[:mm,:]=mask[-mm:,:]=mask[:,:mm]=mask[:,-mm:]=False

    lnOm=args.alpha*T0
    if args.smooth_px>0: lnOm=gaussian_smooth2d(lnOm,args.smooth_px)
    Om=np.exp(lnOm); Om/=Om.mean()+1e-30

    lap_lnOm=laplacian_spectral(lnOm,x,y)
    R2 = -2.0*lap_lnOm/(Om*Om)

    dTdx,dTdy = central_grad(T0,x,y,periodic=True)
    rho_phi = 0.5*(dTdx*dTdx + dTdy*dTdy)/(Om*Om)

    A,B,C,fit,resid,ein_rms,ein_rel,R2_score,(r_rhophi,r_rhom,rhophi_rhom) = \
        direct_fit(R2,rho_phi,rho_m,mask)

    # Λ proxies
    Lambda_proxy_C = C
    Lambda_proxy_mean_resid = float(resid[mask].mean())
    Lambda_proxy_rel = float(C / (np.sqrt(np.mean(R2[mask]**2))+1e-12))

    # save fields
    np.savetxt(os.path.join(args.outdir,"R2_field.csv"), R2, delimiter=",")
    np.savetxt(os.path.join(args.outdir,"rho_phi.csv"), rho_phi, delimiter=",")
    np.savetxt(os.path.join(args.outdir,"rho_m.csv"), rho_m, delimiter=",")
    np.savetxt(os.path.join(args.outdir,"fit_field.csv"), fit, delimiter=",")
    np.savetxt(os.path.join(args.outdir,"residual_field.csv"), resid, delimiter=",")

    # summary
    summary = pd.DataFrame([dict(
        nx=R2.shape[1], ny=R2.shape[0], alpha=args.alpha, smooth_px=args.smooth_px, mask_margin=args.mask_margin,
        A=A, B=B, C=C, R2_score=R2_score, ein_rms=ein_rms, ein_rel=ein_rel,
        corr_R2_rho_phi=r_rhophi, corr_R2_rho_m=r_rhom, corr_rho_phi_rho_m=rhophi_rhom,
        Lambda_proxy_C=Lambda_proxy_C,
        Lambda_proxy_mean_resid=Lambda_proxy_mean_resid,
        Lambda_proxy_rel=Lambda_proxy_rel
    )])
    summary.to_csv(os.path.join(args.outdir,"summary.csv"), index=False)

    # PDF
    pdf=os.path.join(args.outdir,"report.pdf")
    with PdfPages(pdf) as out:
        fig=plt.figure(figsize=(8.6,6.2)); ax=fig.add_subplot(111); ax.axis('off')
        ax.text(0.02,0.98,(
          "Λ proxy from time-slice (DIRECT fit)\n\n"
          f"Grid: {R2.shape[0]}x{R2.shape[1]}   alpha={args.alpha}  smooth_px={args.smooth_px}\n"
          f"A(=κβ²)≈{A:.6g}  B(=κ mscale)≈{B:.6g}  C(Λ-proxy)≈{C:.6g}\n"
          f"R²≈{R2_score:.4f}  RMS≈{ein_rms:.3e}  rel≈{ein_rel:.3e}\n"
          f"corr(R2,ρφ)≈{r_rhophi:.3f}  corr(R2,ρm)≈{r_rhom:.3f}  corr(ρφ,ρm)≈{rhophi_rhom:.3f}\n\n"
          f"Λ proxies:  C={Lambda_proxy_C:.6g}   mean(resid)={Lambda_proxy_mean_resid:.6g}   C/rms(R2)={Lambda_proxy_rel:.3g}\n"
        ), va='top', family='monospace')
        out.savefig(fig); plt.close(fig)

        def page(arr,title,clabel=""):
            fig,ax=plt.subplots(figsize=(6.8,5.0))
            imshow_ax(fig,ax,arr,x,y,title,clabel); out.savefig(fig); plt.close(fig)
        page(R2, "R2 field", "R2")
        page(rho_phi, "rho_phi (unit β²)", "rho_phi")
        page(rho_m, "rho_m", "rho_m")
        page(fit, "Fit: A ρφ + B ρm + C", "fit")
        page(resid, "Residual: R2 - fit", "residual")

    print("=== Λ proxy (DIRECT fit) ===")
    print(summary.to_string(index=False))
    print("Saved:", pdf, "and CSVs in", args.outdir)

if __name__ == "__main__":
    main()