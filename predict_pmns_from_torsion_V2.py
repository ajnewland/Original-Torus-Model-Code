#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, numpy as np, pandas as pd

def idw_interp(xp, yp, zp, X, Y, power=2, eps=1e-12):
    Zi = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dx = xp - X[i, j]; dy = yp - Y[i, j]
            dist2 = dx*dx + dy*dy
            if np.any(dist2 < 1e-14): Zi[i, j] = zp[np.argmin(dist2)]
            else:
                w = 1.0 / np.power(dist2 + eps, power/2.0)
                Zi[i, j] = np.sum(w * zp) / np.sum(w)
    return Zi

def gaussian_2d(x, y, x0, y0, s): return np.exp(-((x-x0)**2+(y-y0)**2)/(2*s*s))
def nearest_unitary(A): U,s,Vh=np.linalg.svd(A,full_matrices=False); return U@Vh
def all_perms(): return [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]

def build_grid(tor, n):
    ax_min, ax_max = float(tor.ax.min()), float(tor.ax.max())
    ay_min, ay_max = float(tor.ay.min()), float(tor.ay.max())
    px, py = 0.02*(ax_max-ax_min or 1), 0.02*(ay_max-ay_min or 1)
    x = np.linspace(ax_min-px, ax_max+px, n)
    y = np.linspace(ay_min-py, ay_max+py, n)
    X, Y = np.meshgrid(x, y); return x, y, X, Y

def path_avg(phi, xs, ys, x0,y0,x1,y1, n=200):
    t = np.linspace(0,1,n); xv = x0+(x1-x0)*t; yv = y0+(y1-y0)*t
    xi = np.clip(np.searchsorted(xs, xv, side='left'), 1, len(xs)-1)
    yi = np.clip(np.searchsorted(ys, yv, side='left'), 1, len(ys)-1)
    vals = np.unwrap(phi[yi, xi]); return float(np.mean(vals))

def compute_pmns(tor, sigma=0.025, grid_n=220):
    x,y,X,Y = build_grid(tor, grid_n)
    T = idw_interp(tor.ax.to_numpy(float), tor.ay.to_numpy(float),
                   tor.t_eff.to_numpy(float), X, Y, power=2)
    dTx,dTy = np.gradient(T, x, y, edge_order=2)
    phi = np.arctan2(dTy, dTx)

    def pos(sp):
        r = tor.loc[tor.species==sp]
        if r.empty: raise ValueError(f"Species '{sp}' not found")
        r = r.iloc[0]; return float(r.ax), float(r.ay)

    # rows: charged leptons; cols: neutrinos
    L = ["e","mu","tau"]
    N = ["nu1","nu2","nu3"]

    Lpos = [pos(s) for s in L]
    Npos = [pos(s) for s in N]

    G_L = [gaussian_2d(X,Y,a,b,sigma) for (a,b) in Lpos]
    G_N = [gaussian_2d(X,Y,a,b,sigma) for (a,b) in Npos]

    # spatial overlaps (no phase) to choose best neutrino column order
    S = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            W = G_L[i]*G_N[j]
            S[i,j] = np.sum(W)/np.sqrt(np.sum(G_L[i]**2)*np.sum(G_N[j]**2))

    best,score = None,-1
    for p in all_perms():
        s = S[0,p[0]] + S[1,p[1]] + S[2,p[2]]
        if s>score: best,score = p,s
    P = np.eye(3)[:,list(best)]
    S_perm = S @ P
    N_perm = [N[k] for k in best]

    # pair-wise path-averaged phases
    Phi = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            j0 = best[j]
            (lx,ly) = Lpos[i]; (nx,ny) = Npos[j0]
            Phi[i,j] = path_avg(phi, x, y, lx, ly, nx, ny, n=200)

    A = S_perm * np.exp(1j*Phi)
    U = nearest_unitary(A)
    return np.abs(U), S, best, ["nu1","nu2","nu3"], N_perm

def rmse_to_pdg(Uabs):
    # PDG-ish |U_PMNS| central (magnitudes): rough reference
    PDG = np.array([
        [0.821, 0.550, 0.150],
        [0.432, 0.582, 0.692],
        [0.378, 0.600, 0.706],
    ])
    return float(np.sqrt(np.mean((Uabs-PDG)**2)))

def main():
    ap = argparse.ArgumentParser(description="Predict PMNS from torsion-phase geometry")
    ap.add_argument("--torsion_csv", required=True)
    ap.add_argument("--locked_csv", required=True)
    ap.add_argument("--sigma", type=float, default=0.025)
    ap.add_argument("--grid_n", type=int, default=220)
    ap.add_argument("--outdir", default="pmns_out")
    ap.add_argument("--sweep_sigma", nargs=3, type=float, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tor = pd.read_csv(args.torsion_csv)
    tor.columns = [c.lower() for c in tor.columns]
    for req in ["species","ax","ay","t_eff"]:
        if req not in tor.columns: raise ValueError(f"Missing '{req}'")

    Uabs, S, perm, N_orig, N_perm = compute_pmns(tor, sigma=args.sigma, grid_n=args.grid_n)
    df = pd.DataFrame(Uabs, index=["e","mu","tau"], columns=N_perm)
    df.to_csv(os.path.join(args.outdir,"predicted_pmns.csv"), float_format="%.6f")

    pd.DataFrame(S, index=["e","mu","tau"], columns=N_orig)\
      .to_csv(os.path.join(args.outdir,"spatial_overlap_S.csv"), float_format="%.6f")
    with open(os.path.join(args.outdir,"chosen_permutation.txt"),"w") as f:
        f.write(f"Best perm of neutrino columns: {perm} -> {N_perm}\n")

    rmse = rmse_to_pdg(Uabs)
    pd.DataFrame([{"RMSE":rmse,"sigma":args.sigma}]).to_csv(
        os.path.join(args.outdir,"pmns_comparison.csv"), index=False, float_format="%.6f"
    )

    print("=== PMNS (geometric) ===")
    print("Neutrino column order:", N_perm)
    print("\n|U_PMNS| (rows e,mu,tau; cols", N_perm, ")")
    print(df.to_string(float_format=lambda v: f"{v:0.6f}"))
    print("\nRMSE vs PDG:", f"{rmse:.6f}")

    if args.sweep_sigma is not None:
        s0,s1,n = args.sweep_sigma; n = int(n)
        sigmas = np.linspace(s0,s1,n); rows=[]
        for s in sigmas:
            Uabs_s,_,_,_,_ = compute_pmns(tor, sigma=s, grid_n=args.grid_n)
            rows.append({"sigma":s, "rmse":rmse_to_pdg(Uabs_s)})
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir,"rmse_vs_sigma.csv"),
                                  index=False, float_format="%.6f")
        print("Saved sweep: rmse_vs_sigma.csv")

if __name__ == "__main__":
    main()
