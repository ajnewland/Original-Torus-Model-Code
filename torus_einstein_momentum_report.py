#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Torus Einstein + Momentum report (normalized Ω, GB, α sweep)
- UTF-8 safe
- Laplacian: five / nine / spectral (FFT)
- Gaussian smoothing for lnΩ
- Interior mask control
- Hamiltonian fit with intercept c0
- Optional κ modulation across the torus: κ(ax,ay) = κ0 + κx*ax + κy*ay

Outputs in <outdir>/ :
  summary.csv
  alpha_sweep_results.csv
  einstein_residual_field.csv
  momentum_abs_field.csv
  momentum_with_source_field.csv
  (kappa_map.csv)        # only when --kappa_mode affine
  report.pdf
"""

import argparse, os, json, sys
import numpy as np, h5py, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------- UTF-8 console ----------
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------- FD helpers ----------
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

def laplacian_5pt(F,x,y):
    dx=float(np.mean(np.diff(x))); dy=float(np.mean(np.diff(y)))
    d2x=(np.roll(F,-1,1)-2*F+np.roll(F,1,1))/(dx*dx)
    d2y=(np.roll(F,-1,0)-2*F+np.roll(F,1,0))/(dy*dy)
    return d2x+d2y

def laplacian_9pt(F,x,y):
    # (-20 f + 4*(N,S,E,W) + 1*(NE,NW,SE,SW)) / (6 h^2), for dx≈dy=h
    h=float(0.5*(np.mean(np.diff(x))+np.mean(np.diff(y))))
    fC=F
    fN=np.roll(F,-1,0); fS=np.roll(F, 1,0)
    fE=np.roll(F,-1,1); fW=np.roll(F, 1,1)
    fNE=np.roll(fN,-1,1); fNW=np.roll(fN, 1,1)
    fSE=np.roll(fS,-1,1); fSW=np.roll(fS, 1,1)
    return (-20*fC + 4*(fN+fS+fE+fW) + (fNE+fNW+fSE+fSW)) / (6*h*h)

# ---------- Spectral Laplacian (periodic, exact on grid) ----------
def laplacian_spectral(F, x, y):
    """
    Periodic spectral Laplacian: invFFT[-(kx^2+ky^2)*FFT(F)].
    """
    Ny, Nx = F.shape
    Lx = float(x.max() - x.min())
    Ly = float(y.max() - y.min())
    if Lx <= 0 or Ly <= 0:
        # assume unit-period if coordinates are not increasing
        Lx = Nx; Ly = Ny
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)  # (Nx,)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=Ly/Ny)  # (Ny,)
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX*KX + KY*KY
    Fhat = np.fft.fft2(F)
    lap_hat = -k2 * Fhat
    lap_hat[0,0] = 0.0  # drop mean mode to avoid NaN/Inf
    lap = np.real(np.fft.ifft2(lap_hat))
    return lap

# ---------- Smoothing ----------
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
    # rows (periodic)
    for j in range(F.shape[0]):
        row=np.r_[F[j,-pad:], F[j], F[j,:pad]]
        conv=np.convolve(row, k, mode="same")
        G[j,:]=conv[pad:pad+F.shape[1]]
    # cols (periodic)
    H=G.copy()
    for i in range(F.shape[1]):
        col=np.r_[G[-pad:, i], G[:, i], G[:pad, i]]
        conv=np.convolve(col, k, mode="same")
        H[:, i]=conv[pad:pad+F.shape[0]]
    return H

# ---------- Hamiltonian fits ----------
def fit_hamiltonian_scalar_kappa(R2, rho_tot, mask):
    r = R2[mask].ravel()
    a = rho_tot[mask].ravel()
    Xs = np.vstack([(a-a.mean())/(a.std() if a.std()>0 else 1.0), np.ones_like(a)]).T
    rs = (r-r.mean())/(r.std() if r.std()>0 else 1.0)
    coeff, *_ = np.linalg.lstsq(Xs, rs, rcond=None)
    c1s, c0s = coeff
    r_mean=r.mean(); r_std=r.std() if r.std()>0 else 1.0
    a_std = a.std() if a.std()>0 else 1.0
    kappa = (r_std*c1s)/a_std
    c0    = r_mean + r_std*c0s
    resid = R2 - (kappa*rho_tot + c0)
    ein_rms = float(np.sqrt(np.mean(resid[mask]**2)))
    ein_rel = float(ein_rms/(np.sqrt(np.mean(R2[mask]**2))+1e-12))
    return float(kappa), float(c0), resid, ein_rms, ein_rel

def fit_hamiltonian_affine_kappa(R2, rho_tot, ax_grid, ay_grid, mask):
    # r ≈ [rho_tot, ax*rho_tot, ay*rho_tot, 1] · [κ0, κx, κy, c0]
    r   = R2[mask].ravel()
    a   = rho_tot[mask].ravel()
    ax  = ax_grid[mask].ravel()
    ay  = ay_grid[mask].ravel()
    X = np.vstack([a, ax*a, ay*a, np.ones_like(a)]).T
    Xs=X.copy(); scales=np.ones(4)
    for j in range(3):
        s=X[:,j].std()
        if s>0: Xs[:,j]/=s; scales[j]=s
    rs=(r-r.mean())/(r.std() if r.std()>0 else 1.0)
    coeff, *_ = np.linalg.lstsq(Xs, rs, rcond=None)
    cvec_s = coeff[:3]; c0s=coeff[3]
    r_mean=r.mean(); r_std=r.std() if r.std()>0 else 1.0
    scales3=scales[:3]
    cvec = (r_std * cvec_s) / np.where(scales3>0, scales3, 1.0)
    c0   = r_mean + r_std * c0s
    k0, kx, ky = cvec.tolist()
    kappa_map = k0 + kx*ax_grid + ky*ay_grid
    resid = R2 - (kappa_map * rho_tot + c0)
    ein_rms = float(np.sqrt(np.mean(resid[mask]**2)))
    ein_rel = float(ein_rms/(np.sqrt(np.mean(R2[mask]**2))+1e-12))
    return (float(k0), float(kx), float(ky)), float(c0), kappa_map, resid, ein_rms, ein_rel

# ---------- Momentum helper ----------
def div_tensor_2d(Txx, Txy, Tyx, Tyy, x, y, Gamma):
    dx=float(np.mean(np.diff(x))); dy=float(np.mean(np.diff(y)))
    dTxx_dx=(np.roll(Txx,-1,1)-np.roll(Txx,1,1))/(2*dx)
    dTxy_dy=(np.roll(Txy,-1,0)-np.roll(Txy,1,0))/(2*dy)
    dTyx_dx=(np.roll(Tyx,-1,1)-np.roll(Tyx,1,1))/(2*dx)
    dTyy_dy=(np.roll(Tyy,-1,0)-np.roll(Tyy,1,0))/(2*dy)
    Gxxx,Gxxy,Gxyx,Gxyy, Gyxx,Gyxy,Gyyx,Gyyy = Gamma
    Cx = (dTxx_dx + dTxy_dy
          + Gxxx*Txx + Gxxy*Tyx + Gxyx*Txy + Gxyy*Tyy
          - (Gxxx+Gyxx)*Txx - (Gxxy+Gyxy)*Txy)
    Cy = (dTyx_dx + dTyy_dy
          + Gyxx*Txx + Gyxy*Tyx + Gyyx*Txy + Gyyy*Tyy
          - (Gxxx+Gyxx)*Tyx - (Gxxy+Gyxy)*Tyy)
    return Cx, Cy

def imshow_ax(fig,ax,arr,x,y,title,clabel=""):
    im=ax.imshow(arr,origin='lower',extent=[x.min(),x.max(),y.min(),y.max()])
    cb=fig.colorbar(im,ax=ax)
    if clabel: cb.set_label(clabel)
    ax.set_title(title); ax.set_xlabel("a_x"); ax.set_ylabel("a_y")
    return im

def infer_meta_path(h5_path):
    base=os.path.basename(h5_path)
    if base.endswith("_fields.h5"): return h5_path.replace("_fields.h5","_meta.json")
    return None

# ---------- Main ----------
def main():
    ap=argparse.ArgumentParser("Torus Einstein + Momentum report (GB + α sweep + intercept, normalized Ω, κ-mod)")
    ap.add_argument("--h5_t0", required=True)
    ap.add_argument("--h5_t1", required=True)
    ap.add_argument("--meta_t0", default=None)
    ap.add_argument("--meta_t1", default=None)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--lapse", type=float, default=1.0)
    ap.add_argument("--alpha_sweep", default=None, help="Comma list, e.g. '4.6,4.75,4.9'.")
    ap.add_argument("--laplace", choices=["five","nine","spectral"], default="nine")
    ap.add_argument("--smooth_px", type=float, default=1.6)
    ap.add_argument("--mask_margin", type=int, default=8)
    ap.add_argument("--kappa_mode", choices=["none","affine"], default="none",
                    help="Allow κ to vary across the torus: 'affine' fits κ0, κx, κy.")
    ap.add_argument("--outdir", default="torus_report_out")
    args=ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load slices
    with h5py.File(args.h5_t0,"r") as f0, h5py.File(args.h5_t1,"r") as f1:
        x=f0["x"][:]; y=f0["y"][:]
        T0=f0["Teff"][:]; T1=f1["Teff"][:]
        rho_m=f0["rho_m"][:]

    Ny, Nx = T0.shape
    ax_grid, ay_grid = np.meshgrid(x, y)

    mm = int(args.mask_margin)
    mask = np.ones_like(T0,bool)
    mask[:mm,:]=mask[-mm:,:]=mask[:,:mm]=mask[:,-mm:]=False

    # Gauss–Bonnet (optional)
    def load_gb(path):
        try:
            with open(path,"r",encoding="utf-8") as f: j=json.load(f)
            return float(j.get("gauss_bonnet_integral", np.nan))
        except Exception:
            return np.nan
    meta_t0 = args.meta_t0 or infer_meta_path(args.h5_t0)
    meta_t1 = args.meta_t1 or infer_meta_path(args.h5_t1)
    GB_t0 = load_gb(meta_t0) if meta_t0 else np.nan
    GB_t1 = load_gb(meta_t1) if meta_t1 else np.nan

    # ∇T on t0
    dTdx0,dTdy0 = central_grad(T0,x,y,periodic=True)

    # Alpha list
    alphas = [float(v) for v in args.alpha_sweep.split(",")] if args.alpha_sweep else [4.6,4.75,4.9]

    results=[]; best=None

    for alpha in alphas:
        # lnΩ (smooth, then exponentiate)
        lnOm0 = alpha*T0
        lnOm1 = alpha*T1
        if args.smooth_px>0.0:
            lnOm0 = gaussian_smooth2d(lnOm0, args.smooth_px)
            lnOm1 = gaussian_smooth2d(lnOm1, args.smooth_px)
        Om0 = np.exp(lnOm0); Om1 = np.exp(lnOm1)
        # Normalize Ω to unit mean (remove global amplitude bias)
        Om0 = Om0/(Om0.mean()+1e-30)
        Om1 = Om1/(Om1.mean()+1e-30)

        # R2 with chosen Laplacian
        if args.laplace == "nine":
            lap0 = laplacian_9pt(lnOm0, x, y)
        elif args.laplace == "five":
            lap0 = laplacian_5pt(lnOm0, x, y)
        else:
            lap0 = laplacian_spectral(lnOm0, x, y)
        R2 = -2.0 * lap0 / (Om0*Om0)

        # rho_phi (unit β^2) and ρ_tot (for any β^2, mscale)
        rho_phi_unit = 0.5*(dTdx0*dTdx0 + dTdy0*dTdy0)/(Om0*Om0)

        # First fit β^2 and mscale (independent of κ-mode), with intercept c0 (linear regression)
        A1 = rho_phi_unit[mask].ravel()
        A2 = rho_m[mask].ravel()
        r  = R2[mask].ravel()
        X  = np.vstack([A1,A2,np.ones_like(A1)]).T
        Xs = X.copy()
        scales = np.maximum(Xs[:,:2].std(0), 1e-12)
        Xs[:,:2] /= scales
        rc = r - r.mean(); rs = rc/(rc.std() if rc.std()>0 else 1.0)
        coeff, *_ = np.linalg.lstsq(Xs, rs, rcond=None)
        c1s,c2s,c0s = coeff
        r_mean=r.mean(); r_std=r.std() if r.std()>0 else 1.0
        beta2  = max(0.0, (r_std*c1s)/scales[0])
        mscale = max(0.0, (r_std*c2s)/scales[1])
        c0_lin = r_mean + r_std*c0s

        rho_tot = beta2 * rho_phi_unit + mscale * rho_m

        # κ fit (scalar or affine)
        if args.kappa_mode == "affine":
            (k0,kx,ky), c0, kappa_map, ein_resid, ein_rms, ein_rel = \
                fit_hamiltonian_affine_kappa(R2, rho_tot, ax_grid, ay_grid, mask)
            kappa_used = "affine"
        else:
            kappa, c0, ein_resid, ein_rms, ein_rel = \
                fit_hamiltonian_scalar_kappa(R2, rho_tot, mask)
            k0,kx,ky = kappa, 0.0, 0.0
            kappa_map = np.full_like(R2, k0)
            kappa_used = "scalar"

        # Momentum constraint
        dt=args.dt; N=args.lapse
        Om_dot=(Om1-Om0)/dt
        Kxx=-(Om0*Om_dot)/N; Kyy=Kxx.copy()
        ginv=1.0/(Om0*Om0); K=ginv*(Kxx+Kyy)

        # Traceless part S_ij in conformal metric
        Kxx_c=ginv*ginv*Kxx; Kyy_c=ginv*ginv*Kyy
        Kxy_c=np.zeros_like(Kxx); Kyx_c=np.zeros_like(Kxx)
        Sxx=Kxx_c - ginv*K
        Syy=Kyy_c - ginv*K
        Sxy=Kxy_c; Syx=Kyx_c

        lnOm=np.log(Om0+1e-30)
        dln_dx,dln_dy=central_grad(lnOm,x,y,periodic=True)
        Gxxx=dln_dx; Gxxy=0.0*dln_dx; Gxyx=dln_dy; Gxyy=-dln_dx
        Gyxx=-dln_dy; Gyxy=dln_dx; Gyyx=0.0*dln_dx; Gyyy=dln_dy
        Gamma=(Gxxx,Gxxy,Gxyx,Gxyy, Gyxx,Gyxy,Gyyx,Gyyy)

        Cx,Cy=div_tensor_2d(Sxx,Sxy,Syx,Syy,x,y,Gamma)
        Cnorm=(Om0**2)*(Cx*Cx+Cy*Cy)

        dKdx,dKdy=central_grad(K,x,y,periodic=True)
        gradK_norm=(Om0**2)*(dKdx**2+dKdy**2)
        mom_rms_abs=float(np.sqrt(np.mean(Cnorm[mask])))
        mom_rms_ref=float(np.sqrt(np.mean(gradK_norm[mask]))+1e-12)
        mom_rel=float(mom_rms_abs/mom_rms_ref)

        # Scalar current j^i ∝ Ω^{-2} Ṫ ∇^iT
        Tdot=(T1-T0)/dt
        jx=ginv*Tdot*dTdx0; jy=ginv*Tdot*dTdy0
        num=float(np.sum(Cx[mask]*jx[mask]+Cy[mask]*jy[mask]))
        den=float(np.sum(jx[mask]*jx[mask]+jy[mask]*jy[mask])+1e-30)
        c_phi=num/den
        Cx_src=Cx-c_phi*jx; Cy_src=Cy-c_phi*jy
        Cnorm_src=(Om0**2)*(Cx_src*Cx_src+Cy_src*Cy_src)
        mom_rms_abs_src=float(np.sqrt(np.mean(Cnorm_src[mask])))
        mom_rel_src=float(mom_rms_abs_src/mom_rms_ref)

        score=float(np.sqrt(ein_rel**2+mom_rel_src**2))
        beta=float(np.sqrt(beta2))

        rec = dict(
            alpha=alpha, beta=beta,
            kappa=k0, kappa_x=kx, kappa_y=ky, kappa_mode=kappa_used,
            mscale=mscale, c0=c0,
            ein_rms=ein_rms, ein_rel=ein_rel,
            mom_rms_abs=mom_rms_abs, mom_rel=mom_rel,
            c_phi=c_phi, mom_rms_abs_src=mom_rms_abs_src, mom_rel_src=mom_rel_src,
            score=score,
            fields=dict(R2=R2, rho_tot=rho_tot, ein_resid=ein_resid,
                        Cnorm=Cnorm, Cnorm_src=Cnorm_src, kappa_map=kappa_map),
            GB_t0=GB_t0, GB_t1=GB_t1
        )
        results.append(rec)
        if (best is None) or (score<best["score"]): best=rec

    # Save sweep table
    sweep_df=pd.DataFrame([{k:v for k,v in r.items() if k!="fields"} for r in results])
    sweep_df.to_csv(os.path.join(args.outdir,"alpha_sweep_results.csv"), index=False)

    # Best fields/metrics
    R2        = best["fields"]["R2"]
    rho_tot   = best["fields"]["rho_tot"]
    ein_resid = best["fields"]["ein_resid"]
    Cnorm     = best["fields"]["Cnorm"]
    Cnorm_src = best["fields"]["Cnorm_src"]
    kmap      = best["fields"]["kappa_map"]

    # Save fields
    np.savetxt(os.path.join(args.outdir,"einstein_residual_field.csv"), ein_resid, delimiter=",")
    np.savetxt(os.path.join(args.outdir,"momentum_abs_field.csv"), Cnorm, delimiter=",")
    np.savetxt(os.path.join(args.outdir,"momentum_with_source_field.csv"), Cnorm_src, delimiter=",")
    if args.kappa_mode=="affine":
        np.savetxt(os.path.join(args.outdir,"kappa_map.csv"), kmap, delimiter=",")

    # One-line summary
    summary = pd.DataFrame([dict(
        nx=R2.shape[1], ny=R2.shape[0], dt=args.dt, lapse=args.lapse,
        alpha=best["alpha"], beta=best["beta"],
        kappa=best["kappa"], kappa_x=best.get("kappa_x",0.0), kappa_y=best.get("kappa_y",0.0),
        kappa_mode=best["kappa_mode"],
        mscale=best["mscale"], c0=best["c0"],
        ein_rms=best["ein_rms"], ein_rel=best["ein_rel"],
        mom_rms_abs=best["mom_rms_abs"], mom_rel=best["mom_rel"],
        c_phi=best["c_phi"], mom_rms_abs_src=best["mom_rms_abs_src"], mom_rel_src=best["mom_rel_src"],
        GB_t0=best["GB_t0"], GB_t1=best["GB_t1"], constraint_score=best["score"]
    )])
    summary.to_csv(os.path.join(args.outdir,"summary.csv"), index=False)

    # PDF
    pdf_path=os.path.join(args.outdir,"report.pdf")
    with PdfPages(pdf_path) as pdf:
        fig=plt.figure(figsize=(8.7,6.4)); ax=fig.add_subplot(111); ax.axis('off')
        txt = (
            "Torus constraint report\n\n"
            f"Grid: {R2.shape[0]} x {R2.shape[1]}\n"
            f"dt = {args.dt},  lapse N = {args.lapse}\n\n"
            f"Gauss–Bonnet (torus → 0): GB(t0) ≈ {best['GB_t0']:.3e},  GB(t1) ≈ {best['GB_t1']:.3e}\n\n"
            f"Best alpha = {best['alpha']:.3g}\n"
            "Hamiltonian (with intercept):\n"
            f"  beta ≈ {best['beta']:.4g}   κ-mode: {best['kappa_mode']}\n"
            f"  κ0 ≈ {best['kappa']:.4g}\n"
            f"  mscale ≈ {best['mscale']:.4g}   c0 ≈ {best['c0']:.4g}\n"
            f"  residual RMS = {best['ein_rms']:.3e}   rel = {best['ein_rel']:.3e}\n\n"
            "Momentum constraint:\n"
            f"  ||C||_rms = {best['mom_rms_abs']:.3e}   rel = {best['mom_rel']:.3e}\n"
            "  With scalar current  j^i ∝ Ω^{-2} Ṫ ∇^iT:\n"
            f"    c_phi ≈ {best['c_phi']:.3e}\n"
            f"    ||C - c_phi j||_rms = {best['mom_rms_abs_src']:.3e}   rel = {best['mom_rel_src']:.3e}\n\n"
            f"S = sqrt(ein_rel^2 + mom_rel_src^2) = {best['score']:.3e}\n"
            f"(laplace={args.laplace}, smooth_px={args.smooth_px}, mask_margin={args.mask_margin})\n"
        )
        ax.text(0.02,0.98,txt,va='top',family='monospace')
        pdf.savefig(fig); plt.close(fig)

        def page(arr,title,clabel=""):
            fig,ax=plt.subplots(figsize=(7,5))
            imshow_ax(fig,ax,arr,x,y,title,clabel)
            pdf.savefig(fig); plt.close(fig)

        page(R2, r"Geometric curvature $R_2$", "R2")
        page(rho_tot, r"$\rho_{\rm tot}=\beta^2\rho_\phi + m_{\rm scale}\rho_m$")
        page(ein_resid, "Einstein residual", "R2 - (κρ + c0)")
        page(Cnorm, r"$|\mathcal{C}|$")
        page(Cnorm_src, r"$|\mathcal{C}-c_\phi\, j|$")

    print("=== Torus report (GB + alpha sweep + intercept) ===")
    print(summary.to_string(index=False))
    print("Saved:", pdf_path, "and CSVs in", args.outdir)

if __name__ == "__main__":
    main()

