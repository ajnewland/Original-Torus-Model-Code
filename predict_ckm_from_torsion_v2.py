#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved geometric CKM predictor:
 - Aligns quark generations by maximizing diagonal spatial overlap (S),
 - Uses path-averaged torsion-phase between centers for complex phases,
 - Projects to nearest unitary and compares to PDG.

Usage:
  python predict_ckm_from_torsion_v2.py \
    --torsion_csv torsion_asymmetry.csv \
    --locked_csv  all_particles_locked.csv \
    --sigma 0.025 --grid_n 220 --outdir ckm_out

Optional:
  --sweep_sigma 0.015 0.050 8   # start end npoints; saves rmse_vs_sigma.csv
"""
import argparse, os, numpy as np, pandas as pd

def idw_interp(xp, yp, zp, X, Y, power=2, eps=1e-12):
    Zi = np.zeros_like(X, dtype=float)
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

def gaussian_2d(x, y, x0, y0, sigma):
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2.0*sigma**2))

def nearest_unitary(A):
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    return U @ Vh

def jarlskog(V):
    return np.imag(V[0,0]*V[1,1]*np.conj(V[0,1])*np.conj(V[1,0]))

def all_permutations_3():
    return [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]

def build_grid(tor, grid_n):
    ax_min, ax_max = float(tor["ax"].min()), float(tor["ax"].max())
    ay_min, ay_max = float(tor["ay"].min()), float(tor["ay"].max())
    pad_x = 0.02*(ax_max-ax_min if ax_max>ax_min else 1.0)
    pad_y = 0.02*(ay_max-ay_min if ay_max>ay_min else 1.0)
    x = np.linspace(ax_min - pad_x, ax_max + pad_x, grid_n)
    y = np.linspace(ay_min - pad_y, ay_max + pad_y, grid_n)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y

def path_average(phi_grid, x_coords, y_coords, x0, y0, x1, y1, n=200):
    t = np.linspace(0.0, 1.0, n)
    xs = x0 + (x1 - x0)*t; ys = y0 + (y1 - y0)*t
    xi = np.searchsorted(x_coords, xs, side='left')
    yi = np.searchsorted(y_coords, ys, side='left')
    xi = np.clip(xi, 1, len(x_coords)-1); yi = np.clip(yi, 1, len(y_coords)-1)
    vals = phi_grid[yi, xi]
    vals_unwrapped = np.unwrap(vals)
    return float(np.mean(vals_unwrapped))

def compute_ckm(tor, sigma=0.025, grid_n=220):
    x, y, X, Y = build_grid(tor, grid_n)
    Tv = tor["t_eff"].to_numpy(float)
    Xp = tor["ax"].to_numpy(float); Yp = tor["ay"].to_numpy(float)
    Tgrid = idw_interp(Xp, Yp, Tv, X, Y, power=2)
    dTx, dTy = np.gradient(Tgrid, x, y, edge_order=2)
    phi = np.arctan2(dTy, dTx)

    def pos(sp):
        r = tor.loc[tor["species"]==sp]
        if r.empty: raise ValueError(f"Species '{sp}' not in torsion CSV.")
        r = r.iloc[0]; return float(r["ax"]), float(r["ay"])
    up_species   = ["u","c","t"]; down_species = ["d","s","b"]
    up_pos   = [pos(s) for s in up_species]
    down_pos = [pos(s) for s in down_species]

    G_up   = [gaussian_2d(X, Y, ax0, ay0, sigma) for (ax0,ay0) in up_pos]
    G_down = [gaussian_2d(X, Y, ax0, ay0, sigma) for (ax0,ay0) in down_pos]
    S = np.zeros((3,3), dtype=float)
    for i in range(3):
        for j in range(3):
            Wij = G_up[i]*G_down[j]
            S[i,j] = np.sum(Wij) / np.sqrt(np.sum(G_up[i]**2)*np.sum(G_down[j]**2))

    best_perm, best_score = None, -1.0
    for perm in all_permutations_3():
        score = S[0,perm[0]] + S[1,perm[1]] + S[2,perm[2]]
        if score > best_score: best_score, best_perm = score, perm
    P = np.eye(3)[:, list(best_perm)]
    S_perm = S @ P
    down_species_perm = [down_species[k] for k in best_perm]

    Phi = np.zeros((3,3), dtype=float)
    for i in range(3):
        for j in range(3):
            j_orig = best_perm[j]
            (ux,uy) = up_pos[i]; (dx,dy) = down_pos[j_orig]
            Phi[i,j] = path_average(phi, x, y, ux, uy, dx, dy, n=200)

    A = S_perm * np.exp(1j * Phi)
    V = nearest_unitary(A)
    Vabs = np.abs(V)
    return Vabs, S, best_perm, ["d","s","b"], down_species_perm

def rmse_to_pdg(Vabs):
    V_pdg = np.array([
        [0.97401, 0.22650, 0.00361],
        [0.22636, 0.97320, 0.04053],
        [0.00854, 0.03978, 0.999172],
    ])
    return float(np.sqrt(np.mean((Vabs - V_pdg)**2)))

def main():
    ap = argparse.ArgumentParser(description="Improved CKM prediction from torsion-phase geometry")
    ap.add_argument("--torsion_csv", required=True)
    ap.add_argument("--locked_csv", required=True)
    ap.add_argument("--sigma", type=float, default=0.025)
    ap.add_argument("--grid_n", type=int, default=220)
    ap.add_argument("--outdir", default="ckm_out_v2")
    ap.add_argument("--sweep_sigma", nargs=3, type=float, default=None, help="start end npoints")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tor = pd.read_csv(args.torsion_csv)
    tor.columns = [c.lower() for c in tor.columns]
    for req in ["species","sector","ax","ay","t_eff"]:
        if req not in tor.columns: raise ValueError(f"torsion_csv must include '{req}'")

    Vabs, S, perm, down_orig, down_perm = compute_ckm(tor, sigma=args.sigma, grid_n=args.grid_n)
    pred_df = pd.DataFrame(Vabs, index=["u","c","t"], columns=down_perm)
    pred_path = os.path.join(args.outdir, "predicted_ckm_v2.csv")
    pred_df.to_csv(pred_path, float_format="%.6f")

    align_df = pd.DataFrame(S, index=["u","c","t"], columns=down_orig)
    align_df.to_csv(os.path.join(args.outdir, "spatial_overlap_S.csv"), float_format="%.6f")
    with open(os.path.join(args.outdir, "chosen_permutation.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best permutation of down columns: {perm}  -> {down_perm}\n")

    V_pdg = np.array([
        [0.97401, 0.22650, 0.00361],
        [0.22636, 0.97320, 0.04053],
        [0.00854, 0.03978, 0.999172],
    ])
    comp = pd.DataFrame({
        "pred_Vud":[Vabs[0,0]],"PDG_Vud":[V_pdg[0,0]],"delta_Vud":[Vabs[0,0]-V_pdg[0,0]],
        "pred_Vus":[Vabs[0,1]],"PDG_Vus":[V_pdg[0,1]],"delta_Vus":[Vabs[0,1]-V_pdg[0,1]],
        "pred_Vub":[Vabs[0,2]],"PDG_Vub":[V_pdg[0,2]],"delta_Vub":[Vabs[0,2]-V_pdg[0,2]],
        "pred_Vcd":[Vabs[1,0]],"PDG_Vcd":[V_pdg[1,0]],"delta_Vcd":[Vabs[1,0]-V_pdg[1,0]],
        "pred_Vcs":[Vabs[1,1]],"PDG_Vcs":[V_pdg[1,1]],"delta_Vcs":[Vabs[1,1]-V_pdg[1,1]],
        "pred_Vcb":[Vabs[1,2]],"PDG_Vcb":[V_pdg[1,2]],"delta_Vcb":[Vabs[1,2]-V_pdg[1,2]],
        "pred_Vtd":[Vabs[2,0]],"PDG_Vtd":[0.00854],"delta_Vtd":[Vabs[2,0]-0.00854],
        "pred_Vts":[Vabs[2,1]],"PDG_Vts":[0.03978],"delta_Vts":[Vabs[2,1]-0.03978],
        "pred_Vtb":[Vabs[2,2]],"PDG_Vtb":[0.999172],"delta_Vtb":[Vabs[2,2]-0.999172],
        "RMSE":[rmse_to_pdg(Vabs)]
    })
    comp_path = os.path.join(args.outdir, "ckm_comparison_v2.csv")
    comp.to_csv(comp_path, index=False, float_format="%.6f")

    # Optional sigma sweep
    if args.sweep_sigma is not None:
        s0, s1, n = args.sweep_sigma
        n = int(n)
        sigmas = np.linspace(s0, s1, n)
        rows = []
        for s in sigmas:
            Vabs_s, _, _, _, _ = compute_ckm(tor, sigma=s, grid_n=args.grid_n)
            rows.append({"sigma": s, "rmse": rmse_to_pdg(Vabs_s)})
        sweep_df = pd.DataFrame(rows)
        sweep_df.to_csv(os.path.join(args.outdir, "rmse_vs_sigma.csv"), index=False, float_format="%.6f")

    print("=== CKM v2 (aligned) ===")
    print("Down permutation chosen:", down_perm)
    print("\n|V_CKM| magnitudes (rows u,c,t; cols", down_perm, "):")
    print(pred_df.to_string(float_format=lambda v: f"{v:0.6f}"))
    print("\nRMSE vs PDG:", f"{rmse_to_pdg(Vabs):.6f}")
    print("\nSaved:")
    print(" -", pred_path)
    print(" -", comp_path)
    print(" -", os.path.join(args.outdir, "spatial_overlap_S.csv"))
    print(" -", os.path.join(args.outdir, "chosen_permutation.txt"))
    if args.sweep_sigma is not None:
        print(" -", os.path.join(args.outdir, "rmse_vs_sigma.csv"))

if __name__ == "__main__":
    main()
