#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a perturbed torus ensemble with t0, t1, t2 per run (for stronger K_ij signal).

Inputs:
 --template_h5 : stable t0 fields (e.g., torus_rich_t0_fields.h5 with datasets: Teff, rho_m, ax, ay)
Outputs (per run_i):
 run_XXX/series_t0_fields.h5
 run_XXX/series_t1_fields.h5
 run_XXX/series_t2_fields.h5

We perturb only Teff slightly (keeps the internal lattice/particle anchors intact),
then do two grad-flow steps (dt each) to get t1 and t2.

Author: A. J. Newland, 2025
"""
import os, sys, argparse, numpy as np, h5py, pandas as pd

try:
   sys.stdout.reconfigure(encoding="utf-8")
except Exception:
   pass

def laplacian_spectral_unit(F):
   Ny, Nx = F.shape
   kx = 2*np.pi*np.fft.fftfreq(Nx, d=1.0)
   ky = 2*np.pi*np.fft.fftfreq(Ny, d=1.0)
   KX, KY = np.meshgrid(kx, ky)
   k2 = KX*KX + KY*KY
   Fhat = np.fft.fft2(F)
   out = np.fft.ifft2(-k2 * Fhat).real
   out[0,0] = 0.0
   return out

def evolve_step_gradflow(T, mu, dt):
   return T + dt * mu * laplacian_spectral_unit(T)

def sanitize(A, fill=0.0):
   B = A.copy()
   bad = ~np.isfinite(B)
   if bad.any(): B[bad] = fill
   return B

def main():
   ap = argparse.ArgumentParser("Build torus ensemble with t0,t1,t2 slices per run")
   ap.add_argument("--template_h5", required=True)
   ap.add_argument("--N", type=int, default=12)
   ap.add_argument("--alpha0", type=float, default=4.9)      # provenance only
   ap.add_argument("--mu0", type=float, default=0.002)
   ap.add_argument("--dalpha", type=float, default=0.02)     # provenance only
   ap.add_argument("--dmu", type=float, default=0.001)
   ap.add_argument("--noise_Teff_std", type=float, default=0.02)
   ap.add_argument("--dt", type=float, default=0.08)
   ap.add_argument("--seed", type=int, default=1234)
   ap.add_argument("--outdir", default="torus_ensemble_t2")
   args = ap.parse_args()
   os.makedirs(args.outdir, exist_ok=True)

   rng = np.random.default_rng(args.seed)

   with h5py.File(args.template_h5, "r") as h:
       T0_base = h["Teff"][:]
       rho_m = h["rho_m"][:]
       ax = h["ax"][:] if "ax" in h else None
       ay = h["ay"][:] if "ay" in h else None

   Tstd = float(np.std(T0_base))
   rows = []

   for i in range(1, args.N+1):
       run_dir = os.path.join(args.outdir, f"run_{i:03d}")
       os.makedirs(run_dir, exist_ok=True)

       mu_i = args.mu0 + rng.uniform(-args.dmu, args.dmu)
       noise = args.noise_Teff_std * Tstd * rng.normal(size=T0_base.shape)

       T0 = sanitize(T0_base + noise)
       T1 = sanitize(evolve_step_gradflow(T0, mu_i, args.dt))
       T2 = sanitize(evolve_step_gradflow(T1, mu_i, args.dt))

       for tag, T in (("t0",T0), ("t1",T1), ("t2",T2)):
           out_h5 = os.path.join(run_dir, f"series_{tag}_fields.h5")
           with h5py.File(out_h5, "w") as w:
               w.create_dataset("Teff", data=T)
               w.create_dataset("rho_m", data=rho_m)
               if ax is not None: w.create_dataset("ax", data=ax)
               if ay is not None: w.create_dataset("ay", data=ay)

       rows.append(dict(run=i, mu=mu_i))
       print(f"[run {i:02d}] wrote t0,t1,t2 to {run_dir}")

   pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "ensemble_manifest.csv"), index=False)
   print("Saved ensemble to:", args.outdir)

if __name__ == "__main__":
   main()