#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Torsion handedness asymmetry from Teff time-slices
- Handedness proxy: vorticity of the normalized gradient field n = ∇T / |∇T|
- Reports signed/absolute means at t0, t1, and their change Δω
- Saves maps and a compact PDF report
"""
import argparse, os, sys
import numpy as np, h5py, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

def central_grad(F, x, y):
    dx=float(np.mean(np.diff(x))); dy=float(np.mean(np.diff(y)))
    dFx=(np.roll(F,-1,1)-np.roll(F,1,1))/(2*dx)
    dFy=(np.roll(F,-1,0)-np.roll(F,1,0))/(2*dy)
    return dFx,dFy

def vorticity_of_unit_grad(T, x, y, eps=1e-12):
    dTx,dTy = central_grad(T,x,y)
    g = np.sqrt(dTx*dTx + dTy*dTy) + eps
    nx, ny = dTx/g, dTy/g
    dny_dx = (np.roll(ny,-1,1)-np.roll(ny,1,1))/(2*float(np.mean(np.diff(x))))
    dnx_dy = (np.roll(nx,-1,0)-np.roll(nx,1,0))/(2*float(np.mean(np.diff(y))))
    omega = dny_dx - dnx_dy
    return omega, (nx,ny), g

def imshow_ax(fig,ax,arr,x,y,title,clabel=""):
    im=ax.imshow(arr,origin='lower',extent=[x.min(),x.max(),y.min(),y.max()])
    cb=fig.colorbar(im,ax=ax);
    if clabel: cb.set_label(clabel)
    ax.set_title(title); ax.set_xlabel("a_x"); ax.set_ylabel("a_y")

def main():
    ap=argparse.ArgumentParser("Torsion handedness asymmetry from Teff slices")
    ap.add_argument("--h5_t0", required=True)
    ap.add_argument("--h5_t1", required=True)
    ap.add_argument("--mask_margin", type=int, default=8)
    ap.add_argument("--outdir", default="torsion_handedness_out")
    args=ap.parse_args(); os.makedirs(args.outdir, exist_ok=True)

    with h5py.File(args.h5_t0,"r") as f0, h5py.File(args.h5_t1,"r") as f1:
        x=f0["x"][:]; y=f0["y"][:]
        T0=f0["Teff"][:]; T1=f1["Teff"][:]

    mm=int(args.mask_margin)
    mask=np.ones_like(T0,bool)
    mask[:mm,:]=mask[-mm:,:]=mask[:,:mm]=mask[:,-mm:]=False

    w0, (nx0,ny0), g0 = vorticity_of_unit_grad(T0,x,y)
    w1, (nx1,ny1), g1 = vorticity_of_unit_grad(T1,x,y)
    dw = w1 - w0

    def stats(w):
        wm=float(w[mask].mean())
        wa=float(np.abs(w[mask]).mean())
        std=float(w[mask].std()+1e-30)
        skew=float((( (w[mask]-wm)/std )**3).mean())
        return dict(mean=wm, abs_mean=wa, std=std, skew=skew)

    s0=stats(w0); s1=stats(w1); sd=stats(dw)

    # save maps
    np.savetxt(os.path.join(args.outdir,"omega_t0.csv"), w0, delimiter=",")
    np.savetxt(os.path.join(args.outdir,"omega_t1.csv"), w1, delimiter=",")
    np.savetxt(os.path.join(args.outdir,"delta_omega.csv"), dw, delimiter=",")

    # summary CSV
    summary = pd.DataFrame([{
        "nx":T0.shape[1],"ny":T0.shape[0],"mask_margin":args.mask_margin,
        "omega_t0_mean":s0["mean"], "omega_t0_absmean":s0["abs_mean"], "omega_t0_std":s0["std"], "omega_t0_skew":s0["skew"],
        "omega_t1_mean":s1["mean"], "omega_t1_absmean":s1["abs_mean"], "omega_t1_std":s1["std"], "omega_t1_skew":s1["skew"],
        "delta_mean":sd["mean"], "delta_absmean":sd["abs_mean"], "delta_std":sd["std"], "delta_skew":sd["skew"]
    }])
    summary.to_csv(os.path.join(args.outdir,"summary.csv"), index=False)

    # PDF
    pdf=os.path.join(args.outdir,"report.pdf")
    with PdfPages(pdf) as out:
        # field maps
        def page(arr,title):
            fig,ax=plt.subplots(figsize=(6.8,5.0))
            imshow_ax(fig,ax,arr,x,y,title,"")
            out.savefig(fig); plt.close(fig)
        page(w0, "Handedness proxy ω(t0)")
        page(w1, "Handedness proxy ω(t1)")
        page(dw, "Δω = ω(t1) - ω(t0)")

        # histograms
        def hist(arr,ttl):
            fig,ax=plt.subplots(figsize=(6.4,4.2))
            ax.hist(arr[mask].ravel(), bins=80, alpha=0.8)
            ax.set_title(ttl); ax.set_xlabel("ω"); ax.set_ylabel("count")
            out.savefig(fig); plt.close(fig)
        hist(w0, "Histogram ω(t0)")
        hist(w1, "Histogram ω(t1)")
        hist(dw, "Histogram Δω")

        # cover page with stats
        fig=plt.figure(figsize=(8.6,6.0)); ax=fig.add_subplot(111); ax.axis('off')
        ax.text(0.02,0.98,(
            "Torsion handedness asymmetry (vorticity of unit ∇T)\n\n"
            f"Grid: {T0.shape[0]}x{T0.shape[1]}   mask_margin={args.mask_margin}\n\n"
            f"t0: mean={s0['mean']:.4e}  |mean|={s0['abs_mean']:.4e}  std={s0['std']:.4e}  skew={s0['skew']:.3f}\n"
            f"t1: mean={s1['mean']:.4e}  |mean|={s1['abs_mean']:.4e}  std={s1['std']:.4e}  skew={s1['skew']:.3f}\n"
            f"Δ : mean={sd['mean']:.4e}  |mean|={sd['abs_mean']:.4e}  std={sd['std']:.4e}  skew={sd['skew']:.3f}\n"
            "\nInterpretation: non-zero signed means/skew indicate net handedness; "
            "growth from t0→t1 (Δω) is a parity-odd channel that can fuel CP-odd processes (baryo/leptogenesis).\n"
        ), va='top', family='monospace')
        out.savefig(fig); plt.close(fig)

    print("=== Torsion handedness ===")
    print(summary.to_string(index=False))
    print("Saved:", pdf, "and CSVs in", args.outdir)

if __name__ == "__main__":
    main()