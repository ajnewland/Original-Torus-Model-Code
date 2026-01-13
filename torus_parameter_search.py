#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid search over (alpha, dt, mu) to minimise constraint score
S = sqrt(ein_rel^2 + mom_rel_src^2).

Inputs
------
--h5_t0         : path to *_t0_fields.h5 (must contain x,y,Teff,rho_m)
--alpha_list    : comma list, e.g. "1,1.5,2,3,4"
--dt_list       : comma list, e.g. "0.1,0.2,0.5,1.0"
--mu_list       : comma list for gradflow evolution, e.g. "0.005,0.01,0.02"
--lapse         : ADM lapse N (default 1.0)
--mode          : 'gradflow' (default) or 'advection'
--advec_angle   : radians (used if mode=advection)
--advec_speed   : speed (used if mode=advection)
--outdir        : output folder

Outputs
-------
<outdir>/
search_results.csv              (one row per (alpha, dt, mu))
best_summary.txt                (human-readable best pick)
best_preview.pdf                (maps for the best combo)
"""

import argparse, os
import numpy as np, h5py, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------- finite-difference helpers ----------------
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

def laplacian(F, x, y):
  dx=float(np.mean(np.diff(x))); dy=float(np.mean(np.diff(y)))
  d2x=(np.roll(F,-1,1)-2*F+np.roll(F,1,1))/(dx*dx)
  d2y=(np.roll(F,-1,0)-2*F+np.roll(F,1,0))/(dy*dy)
  return d2x+d2y

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

def imshow_ax(fig, ax, arr, x, y, title, cbarlabel=""):
  im = ax.imshow(arr, origin='lower', extent=[x.min(),x.max(),y.min(),y.max()])
  cb = fig.colorbar(im, ax=ax);
  if cbarlabel: cb.set_label(cbarlabel)
  ax.set_title(title); ax.set_xlabel("a_x"); ax.set_ylabel("a_y")
  return im

# --------------- core metric computation -------------------
def compute_metrics_for(alpha, dt, mu, mode, angle, speed, x, y, T0, rho_m, lapse):
  mask = np.ones_like(T0,bool); mask[0,:]=mask[-1,:]=mask[:,0]=mask[:,-1]=False

  # Geometry on t0 for this alpha
  lnOm0 = alpha*T0
  Om0 = np.exp(lnOm0)
  R2 = -2.0 * laplacian(lnOm0, x, y) / (Om0*Om0)

  # rho_phi (unit beta^2)
  dTdx0, dTdy0 = central_grad(T0, x, y, periodic=True)
  rho_phi_unit = 0.5 * (dTdx0*dTdx0 + dTdy0*dTdy0) / (Om0*Om0)

  # Hamiltonian fit: R2 ≈ kappa (beta^2 rho_phi + mscale rho_m)
  A1 = rho_phi_unit[mask].ravel()
  A2 = rho_m[mask].ravel()
  r  = R2[mask].ravel()
  X = np.vstack([A1, A2]).T
  Xn = X/np.maximum(X.std(0),1e-12)
  rn = r - r.mean()
  if rn.std()>0: rn /= rn.std()
  coeff, *_ = np.linalg.lstsq(Xn, rn, rcond=None)
  c1, c2 = coeff * np.maximum(X.std(0),1e-12)
  beta2  = float(max(1e-12, c1))
  mscale = float(max(1e-12, c2))

  rho_tot = beta2 * rho_phi_unit + mscale * rho_m
  rt, rhot = R2[mask], rho_tot[mask]
  kappa = float((rt*rhot).sum() / (rhot*rhot).sum())

  ein_resid = R2 - kappa * rho_tot
  ein_rms = float(np.sqrt(np.mean(ein_resid[mask]**2)))
  ein_rel = float(ein_rms / (np.sqrt(np.mean(R2[mask]**2)) + 1e-12))

  # --- synthesize T1 ---
  if mode=="gradflow":
      grad_norm = np.sqrt(dTdx0*dTdx0 + dTdy0*dTdy0)
      T1 = T0 + dt * mu * ( - grad_norm )
  else:  # advection: ∂t T + v·∇T = 0
      vx = speed*np.cos(angle); vy = speed*np.sin(angle)
      T1 = T0 - dt * (vx*dTdx0 + vy*dTdy0)

  # momentum constraint
  lnOm1 = alpha*T1; Om1 = np.exp(lnOm1)
  Om_dot = (Om1-Om0)/dt
  N = lapse
  Kxx = -(Om0*Om_dot)/N; Kyy = Kxx.copy()
  ginv = 1.0/(Om0*Om0)
  K = ginv*(Kxx+Kyy)

  Kxx_c = ginv*ginv*Kxx; Kyy_c = ginv*ginv*Kyy
  Kxy_c = np.zeros_like(Kxx); Kyx_c = np.zeros_like(Kxx)
  Sxx = Kxx_c - ginv*K
  Syy = Kyy_c - ginv*K
  Sxy = Kxy_c; Syx = Kyx_c

  lnOm = np.log(Om0 + 1e-30)
  dln_dx, dln_dy = central_grad(lnOm, x, y, periodic=True)
  Gxxx = dln_dx; Gxxy = 0.0*dln_dx; Gxyx = dln_dy; Gxyy = -dln_dx
  Gyxx = -dln_dy; Gyxy = dln_dx; Gyyx = 0.0*dln_dx; Gyyy = dln_dy
  Gamma = (Gxxx,Gxxy,Gxyx,Gxyy, Gyxx,Gyxy,Gyyx,Gyyy)

  Cx, Cy = div_tensor_2d(Sxx, Sxy, Syx, Syy, x, y, Gamma)
  Cnorm = (Om0**2) * (Cx*Cx + Cy*Cy)
  mom_rms_abs = float(np.sqrt(np.mean(Cnorm[mask])))

  dKdx, dKdy = central_grad(K, x, y, periodic=True)
  gradK_norm = (Om0**2) * (dKdx**2 + dKdy**2)
  mom_rms_ref = float(np.sqrt(np.mean(gradK_norm[mask])) + 1e-12)
  mom_rel = float(mom_rms_abs / mom_rms_ref)

  # source current j^i ∝ Ω^{-2} Ṫ ∇^i T
  Tdot = (T1 - T0)/dt
  jx = ginv * Tdot * dTdx0
  jy = ginv * Tdot * dTdy0
  num = float(np.sum( Cx[mask]*jx[mask] + Cy[mask]*jy[mask] ))
  den = float(np.sum( jx[mask]*jx[mask] + jy[mask]*jy[mask] ) + 1e-30)
  c_phi = num/den
  Cx_src = Cx - c_phi*jx
  Cy_src = Cy - c_phi*jy
  Cnorm_src = (Om0**2) * (Cx_src*Cx_src + Cy_src*Cy_src)
  mom_rms_abs_src = float(np.sqrt(np.mean(Cnorm_src[mask])))
  mom_rel_src = float(mom_rms_abs_src / mom_rms_ref)

  score = float(np.sqrt(ein_rel**2 + mom_rel_src**2))

  return dict(alpha=alpha, dt=dt, mu=mu,
              beta=np.sqrt(beta2), kappa=kappa, mscale=mscale,
              ein_rms=ein_rms, ein_rel=ein_rel,
              mom_rms_abs=mom_rms_abs, mom_rel=mom_rel,
              c_phi=c_phi, mom_rms_abs_src=mom_rms_abs_src, mom_rel_src=mom_rel_src,
              score=score,
              fields=dict(R2=R2, rho_tot=rho_tot, ein_resid=ein_resid,
                          Cnorm=Cnorm, Cnorm_src=Cnorm_src, Om0=Om0))

def main():
  ap = argparse.ArgumentParser("Parameter search: alpha, dt, mu")
  ap.add_argument("--h5_t0", required=True)
  ap.add_argument("--alpha_list", required=True, help="e.g. '1,1.5,2,3'")
  ap.add_argument("--dt_list", required=True, help="e.g. '0.1,0.2,0.5,1.0'")
  ap.add_argument("--mu_list", required=True, help="e.g. '0.005,0.01,0.02'")
  ap.add_argument("--lapse", type=float, default=1.0)
  ap.add_argument("--mode", choices=["gradflow","advection"], default="gradflow")
  ap.add_argument("--advec_angle", type=float, default=0.0)
  ap.add_argument("--advec_speed", type=float, default=0.02)
  ap.add_argument("--outdir", default="torus_param_search_out")
  args = ap.parse_args()
  os.makedirs(args.outdir, exist_ok=True)

  # Load t0 slice
  with h5py.File(args.h5_t0,"r") as f0:
      x=f0["x"][:]; y=f0["y"][:]; T0=f0["Teff"][:]; rho_m=f0["rho_m"][:]

  alphas=[float(v) for v in args.alpha_list.split(",") if v.strip()]
  dts   =[float(v) for v in args.dt_list.split(",") if v.strip()]
  mus   =[float(v) for v in args.mu_list.split(",") if v.strip()]

  results=[]
  best=None

  for a in alphas:
      for dt in dts:
          for mu in mus:
              out = compute_metrics_for(a, dt, mu, args.mode, args.advec_angle, args.advec_speed,
                                        x, y, T0, rho_m, args.lapse)
              results.append({k:v for k,v in out.items() if k!="fields"})
              if (best is None) or (out["score"] < best["score"]):
                  best = out

  # Save table
  df = pd.DataFrame(results)
  df.sort_values("score", inplace=True)
  csv_path = os.path.join(args.outdir,"search_results.csv")
  df.to_csv(csv_path, index=False)

  # Write best summary
  summary_txt = os.path.join(args.outdir,"best_summary.txt")
  with open(summary_txt,"w") as f:
      f.write("=== Best (alpha, dt, mu) ===\n")
      f.write(f"alpha = {best['alpha']}\n")
      f.write(f"dt    = {best['dt']}\n")
      f.write(f"mu    = {best['mu']}\n\n")
      f.write("Fitted (t0 Einstein):\n")
      f.write(f"  beta  ≈ {best['beta']:.5g}\n")
      f.write(f"  kappa ≈ {best['kappa']:.5g}\n")
      f.write(f"  mscale≈ {best['mscale']:.5g}\n")
      f.write(f"ein_rel = {best['ein_rel']:.4g}\n")
      f.write(f"mom_rel_src = {best['mom_rel_src']:.4g}\n")
      f.write(f"SCORE = {best['score']:.4g}\n")
  print(open(summary_txt).read())

  # Quick PDF preview for the best combo
  pdf = os.path.join(args.outdir,"best_preview.pdf")
  with PdfPages(pdf) as doc:
      R2 = best["fields"]["R2"]
      rho_tot = best["fields"]["rho_tot"]
      ein_resid = best["fields"]["ein_resid"]
      Cnorm = best["fields"]["Cnorm"]
      Cnorm_src = best["fields"]["Cnorm_src"]
      Om0 = best["fields"]["Om0"]

      fig = plt.figure(figsize=(8.6,6.2)); ax=fig.add_subplot(111); ax.axis('off')
      ax.text(0.02,0.98,
              (f"Best combo\n\nalpha={best['alpha']}, dt={best['dt']}, mu={best['mu']}\n"
               f"beta≈{best['beta']:.4g}, kappa≈{best['kappa']:.4g}, mscale≈{best['mscale']:.4g}\n"
               f"ein_rel={best['ein_rel']:.3e}, mom_rel_src={best['mom_rel_src']:.3e}\n"
               f"SCORE={best['score']:.3e}"),
              va='top', family='monospace')
      doc.savefig(fig); plt.close(fig)

      def page(arr, title, cbar=""):
          fig, ax = plt.subplots(figsize=(7,5))
          imshow_ax(fig, ax, arr, x, y, title, cbar)
          doc.savefig(fig); plt.close(fig)

      page(R2, r"$R_2$", "R2")
      page(rho_tot, r"$\rho_{\rm tot}$", "")
      page(ein_resid, "Einstein residual", "R2 - κ ρ")
      page(Cnorm, r"$|\mathcal{C}|$", "")
      page(Cnorm_src, r"$|\mathcal{C}-c_\phi j|$", "")

  print("Saved:", csv_path, summary_txt, pdf)

if __name__=="__main__":
  main()