#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electroweak couplings from DEC on a torus with exact period constraints.
We scan the rectangular metric anisotropy (a_x, a_y) and compute g, g', sin^2θ_W, mW, mZ.

Key ideas:
- Build periodic triangular mesh on an Lx × Ly grid with small random jitter.
- Circumcentric dual lengths → diagonal Hodge star ⋆1 on edges.
- Exact period constraints along fundamental cycles γ_x, γ_y:
    minimize (1/2) ω^T ⋆1 ω  subject to  A ω = b,
  solved analytically: ω* = ⋆1^{-1} A^T (A ⋆1^{-1} A^T)^{-1} b
  Energy at optimum = b^T (A ⋆1^{-1} A^T)^{-1} b  ⇒ K = energy.
- Map stiffness to couplings: g = 2π / sqrt(K_xx), g' = 2π / sqrt(K_yy).
- Then sin²θ_W = g'^2 / (g^2 + g'^2),  m_W = g v / 2,  m_Z = v/2 sqrt(g^2+g'^2),  ρ=1.

Outputs:
- out_ax_scan/ax_scan.csv  (grid of ax, statistics over seeds)
- out_ax_scan/summary.txt
- out_ax_scan/sin2_vs_ax.png
"""

import sys
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, diags
from numpy.linalg import solve

# -------------- geometry / mesh ---------------------------------------------

def build_points_torus(Lx, Ly, ax=1.0, ay=1.0, jitter=0.05, seed=0):
    """
    Periodic points on a rectangular torus fundamental domain [0, Lx] × [0, Ly]
    with metric scale factors (ax, ay). We generate coordinates in R^2 that
    already include anisotropy (i.e. x*ax, y*ay), so geometric lengths/areas
    pick up the intended metric. Small jitter keeps the mesh irregular.
    Returns:
      P: (V,2) float32 points
      vid: function (ix,iy) -> vertex index
    """
    rng = np.random.default_rng(seed)
    V = Lx*Ly
    P = np.zeros((V,2), dtype=np.float64)
    def vid(ix, iy):
        return (iy % Ly)*Lx + (ix % Lx)

    for iy in range(Ly):
        for ix in range(Lx):
            x = ix + jitter*(rng.random()-0.5)
            y = iy + jitter*(rng.random()-0.5)
            P[vid(ix,iy), 0] = ax * x
            P[vid(ix,iy), 1] = ay * y
    return P, vid

def triangulate_grid_torus(Lx, Ly, vid):
    """
    Make a periodic triangularization by splitting each quad into 2 triangles.
    Faces: two triangles per cell: (A,B,D) and (A,D,C), with periodic wrap.
    Returns:
      E: list of undirected edges (a,b) with a<b
      F: list of faces as triples of directed edges indices [e_ab, e_bd, e_da], with signs
      d0: incidence V×E (vertex→edge), oriented
      d1: incidence E×F (edge→face), oriented
      face_edges: (F,3) oriented edge indices, face_signs: (F,3) ±1
    """
    # First, enumerate undirected edges with consistent (min,max) key
    edge_dict = {}
    edges = []
    faces = []
    face_edges = []
    face_signs = []

    def add_edge(a,b):
        if a==b: return None
        key = (a,b) if a<b else (b,a)
        if key not in edge_dict:
            edge_dict[key] = len(edges)
            edges.append(key)
        return edge_dict[key], 1 if a<b else -1

    def v(ix,iy): return vid(ix,iy)
    # cells
    for iy in range(Ly):
        for ix in range(Lx):
            A = v(ix,iy)
            B = v(ix+1,iy)
            C = v(ix,iy+1)
            D = v(ix+1,iy+1)
            # face 1: (A,B,D)
            tri1 = [(A,B),(B,D),(D,A)]
            fe_idx = []
            fe_sgn = []
            for (s,t) in tri1:
                ei, sgn = add_edge(s,t)
                fe_idx.append(ei); fe_sgn.append(sgn)
            faces.append((A,B,D))
            face_edges.append(fe_idx); face_signs.append(fe_sgn)
            # face 2: (A,D,C)
            tri2 = [(A,D),(D,C),(C,A)]
            fe_idx = []
            fe_sgn = []
            for (s,t) in tri2:
                ei, sgn = add_edge(s,t)
                fe_idx.append(ei); fe_sgn.append(sgn)
            faces.append((A,D,C))
            face_edges.append(fe_idx); face_signs.append(fe_sgn)

    E = np.array(edges, dtype=np.int64)
    F = np.array(faces, dtype=np.int64)
    face_edges = np.array(face_edges, dtype=np.int64)
    face_signs = np.array(face_signs, dtype=np.int8)

    V = Lx*Ly
    NE = len(E); NF = len(F)

    # d0: V×E (tail→head: +1 at head, -1 at tail), choose global orientation = stored order (a<b)
    rows=[]; cols=[]; vals=[]
    for ei,(a,b) in enumerate(E):
        rows.extend([a,b]); cols.extend([ei,ei]); vals.extend([-1,+1])
    d0 = coo_matrix((vals,(rows,cols)), shape=(V,NE)).tocsr()

    # d1: E×F with oriented agreement per face_edges & face_signs
    rows=[]; cols=[]; vals=[]
    for fi in range(NF):
        for k in range(3):
            ei = face_edges[fi,k]
            sgn = face_signs[fi,k]
            rows.append(ei); cols.append(fi); vals.append(sgn)
    d1 = coo_matrix((vals,(rows,cols)), shape=(NE,NF)).tocsr()

    return E, F, d0, d1, face_edges, face_signs

# -------------- geometry primitives ----------------------------------------

def tri_area(p,q,r):
    # 2D area (with anisotropy already in coordinates P) — robust and NumPy-safe
    return 0.5*abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))

def circumcenter(p,q,r):
    """Circumcenter of triangle (p,q,r) in 2D."""
    a = q - p
    b = r - p
    adot = np.dot(a,a)
    bdot = np.dot(b,b)
    cross = a[0]*b[1] - a[1]*b[0]
    if abs(cross) < 1e-14:
        return (p+q+r)/3.0  # fallback to centroid if nearly collinear
    cx = p[0] + (bdot*(a[1]) - adot*(b[1]))/(2*cross)
    cy = p[1] + (-bdot*(a[0]) + adot*(b[0]))/(2*cross)
    return np.array([cx,cy], dtype=np.float64)

def edge_length(P,a,b):
    return float(np.linalg.norm(P[b]-P[a]))

# -------------- circumcentric ⋆1 -------------------------------------------

def star1_circum(P, E, F, face_edges, face_signs):
    """
    Diagonal Hodge star on edges for circumcentric dual:
    ⋆1(e) = |dual_e| / |e|, with |dual_e| the length of the dual segment between
    circumcenters of the two adjacent faces sharing e (periodic mesh ⇒ every edge has two faces).
    """
    NE = len(E); NF = len(F)
    # For speed, precompute circumcenters of faces
    C = np.zeros((NF,2), dtype=np.float64)
    for fi in range(NF):
        a,b,c = F[fi]
        C[fi,:] = circumcenter(P[a], P[b], P[c])

    # Find for each edge the two incident faces
    faces_of_edge = [[] for _ in range(NE)]
    for fi in range(NF):
        for k in range(3):
            ei = face_edges[fi,k]
            faces_of_edge[ei].append(fi)
    # Build ⋆1 diagonal
    diag = np.zeros(NE, dtype=np.float64)
    for ei,(a,b) in enumerate(E):
        fids = faces_of_edge[ei]
        if len(fids) != 2:
            # on a proper torus we should always have 2
            # fallback: set small positive number to avoid zero
            diag[ei] = 1e-12
            continue
        f0,f1 = fids
        dual_len = float(np.linalg.norm(C[f1]-C[f0]))
        elen = edge_length(P,a,b)
        diag[ei] = max(dual_len,1e-15)/max(elen,1e-15)
    return diags(diag, 0, shape=(NE,NE)).tocsr()

# -------------- exact period constraints -----------------------------------

def fundamental_cycles(Lx, Ly, E, vid):
    """
    Construct two fundamental 1-cycles γ_x, γ_y as lists of directed edges
    that wrap once around the torus. We choose:
      γ_x: a path along a fixed row iy0 from (0,iy0)→(1,iy0)→...→(Lx,iy0≡0)
      γ_y: a path along a fixed column ix0 from (ix0,0)→...→(ix0,Ly≡0)
    We then convert each step into a (edge_index, +1/-1) using the undirected E list.
    """
    # maps a directed pair (u->v) to (ei, sign) where E[ei] = (min, max) and sign = +1 if matches, -1 otherwise
    edict = {}
    for ei,(a,b) in enumerate(E):
        edict[(a,b)] = (ei, +1)
        edict[(b,a)] = (ei, -1)

    # x-cycle on row iy0
    iy0 = Ly//2
    gamma_x = []
    for ix in range(Lx):
        a = vid(ix,iy0)
        b = vid(ix+1,iy0)
        ei, sgn = edict[(a,b)]
        gamma_x.append((ei, sgn))

    # y-cycle on column ix0
    ix0 = Lx//2
    gamma_y = []
    for iy in range(Ly):
        a = vid(ix0,iy)
        b = vid(ix0,iy+1)
        ei, sgn = edict[(a,b)]
        gamma_y.append((ei, sgn))

    return gamma_x, gamma_y

def period_constraint_matrix(NE, gamma):
    """
    Build a 1×NE row A such that (A ω) = sum_{e∈γ} s_e ω_e ℓ_e  = period(ω along γ),
    where we weight by the **primal edge length** ℓ_e so the constraint is geometric length-normalized.
    We will fill edge lengths later; here we put ±1 in the positions and multiply by ℓ outside.
    """
    row = np.zeros((1,NE), dtype=np.float64)
    idxs = [ei for (ei,sgn) in gamma]
    # we return the indices and signs so we can inject the actual ℓ_e before solving
    signs = np.array([sgn for (ei,sgn) in gamma], dtype=np.float64)
    return row, idxs, signs

def constrained_min_energy(star1, A, b):
    """
    Solve min (1/2) ω^T ⋆1 ω  subject to A ω = b.
    Analytic solution:  ω* = ⋆1^{-1} A^T (A ⋆1^{-1} A^T)^{-1} b
    Returns ω*,  energy = b^T (A ⋆1^{-1} A^T)^{-1} b
    """
    # ⋆1 is diagonal → use reciprocal diagonal
    diag = star1.diagonal()
    inv_diag = 1.0/np.maximum(diag, 1e-15)
    star1_inv = diags(inv_diag, 0, shape=star1.shape).tocsr()

    # M = A ⋆1^{-1} A^T
    M = A @ (star1_inv @ A.T)
    # robust solve: use dense solve for small constraint matrices
    Md = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    bd = np.asarray(b, dtype=np.float64)
    y = solve(Md, bd)  # y = (A ⋆1^{-1} A^T)^{-1} b
    omega = star1_inv @ (A.T @ y)
    energy = float(bd.T @ y)
    return omega, energy

# -------------- EW from geometry -------------------------------------------

def ew_from_geometry(P, Lx, Ly, ax, ay, v=246.0, jitter=0.05, seed=0):
    """
    One geometry:
      - triangulate
      - ⋆1 (circumcentric)
      - exact periods → K_xx, K_yy
      - g,g', sin²θW, mW, mZ (ρ=1)
    """
    # points P given; rebuild indexing & mesh
    def vid(ix,iy): return (iy%Ly)*Lx + (ix%Lx)
    E, F, d0, d1, face_edges, face_signs = triangulate_grid_torus(Lx, Ly, vid)

    # Hodge ⋆1
    star1 = star1_circum(P, E, F, face_edges, face_signs)

    # cycles and constraint rows (with edge-length weights)
    gamma_x, gamma_y = fundamental_cycles(Lx, Ly, E, vid)

    NE = len(E)
    # edge lengths vector
    L_e = np.array([edge_length(P,a,b) for (a,b) in E], dtype=np.float64)

    # Build A_x, A_y as sparse 1×NE rows with entries sgn * L_e on the cycle edges
    row_x = np.zeros((1,NE), dtype=np.float64)
    for (ei, sgn) in gamma_x:
        row_x[0, ei] += sgn * L_e[ei]
    A_x = csr_matrix(row_x)

    row_y = np.zeros((1,NE), dtype=np.float64)
    for (ei, sgn) in gamma_y:
        row_y[0, ei] += sgn * L_e[ei]
    A_y = csr_matrix(row_y)

    # period targets b_x=1, b_y=1 (unit line-integral around fundamental cycles)
    b_x = np.array([1.0], dtype=np.float64)
    b_y = np.array([1.0], dtype=np.float64)

    # Solve constrained minimization to get energies → stiffnesses
    _, K_xx = constrained_min_energy(star1, A_x, b_x)
    _, K_yy = constrained_min_energy(star1, A_y, b_y)

    # Map to couplings (2π normalization for a U(1)-like convention)
    g  = 2.0*math.pi / math.sqrt(max(K_xx, 1e-30))
    gp = 2.0*math.pi / math.sqrt(max(K_yy, 1e-30))

    s2 = (gp*gp) / (g*g + gp*gp)  # sin^2 θ_W
    mW = 0.5 * g  * v
    mZ = 0.5 * v * math.sqrt(g*g + gp*gp)
    rho = 1.0

    return dict(
        Lx=Lx, Ly=Ly, ax=ax, ay=ay, jitter=jitter,
        g=g, gp=gp, sin2=s2, mW=mW, mZ=mZ, rho=rho,
        Kxx=K_xx, Kyy=K_yy, NE=len(E), NF=len(F)
    )

# -------------- batch scan --------------------------------------------------

def make_points(Lx, Ly, ax, ay, jitter, seed):
    P, _ = build_points_torus(Lx, Ly, ax=ax, ay=ay, jitter=jitter, seed=seed)
    return P

def scan_ax(Lx=16, Ly=16, n_meshes=12, ax_min=1.3, ax_max=2.6, ax_steps=14,
            ay=1.0, jitter=0.08, v=246.0, seed=2025):
    """
    1D scan over ax; average EW outputs over n_meshes seeds for each ax.
    """
    rng = np.random.default_rng(seed)
    ax_vals = np.linspace(ax_min, ax_max, ax_steps)
    rows = []
    for ax in ax_vals:
        sins = []; gvals=[]; gpvals=[]; mWs=[]; mZs=[]; Kxs=[]; Kys=[]
        for k in range(n_meshes):
            s = int(rng.integers(0, 10**9))
            P = make_points(Lx, Ly, ax, ay, jitter, s)
            res = ew_from_geometry(P, Lx, Ly, ax, ay, v=v, jitter=jitter, seed=s)
            sins.append(res['sin2']); gvals.append(res['g']); gpvals.append(res['gp'])
            mWs.append(res['mW']); mZs.append(res['mZ'])
            Kxs.append(res['Kxx']); Kys.append(res['Kyy'])
        rows.append(dict(
            ax=ax, ay=ay,
            sin2_mean=np.mean(sins), sin2_std=np.std(sins),
            g_mean=np.mean(gvals), g_std=np.std(gvals),
            gp_mean=np.mean(gpvals), gp_std=np.std(gpvals),
            mW_mean=np.mean(mWs), mW_std=np.std(mWs),
            mZ_mean=np.mean(mZs), mZ_std=np.std(mZs),
            Kxx_mean=np.mean(Kxs), Kxx_std=np.std(Kxs),
            Kyy_mean=np.mean(Kys), Kyy_std=np.std(Kys),
        ))
    df = pd.DataFrame(rows)
    return df

# -------------- CLI / plotting ---------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Scan ax anisotropy with exact period constraints and circumcentric ⋆1.")
    ap.add_argument("--Lx", type=int, default=16, help="grid in x")
    ap.add_argument("--Ly", type=int, default=16, help="grid in y")
    ap.add_argument("--n", type=int, default=12, help="meshes per ax")
    ap.add_argument("--ax_min", type=float, default=1.3)
    ap.add_argument("--ax_max", type=float, default=2.6)
    ap.add_argument("--ax_steps", type=int, default=14)
    ap.add_argument("--ay", type=float, default=1.0)
    ap.add_argument("--jitter", type=float, default=0.08)
    ap.add_argument("--v", type=float, default=246.0)
    ap.add_argument("--target_sin2", type=float, default=0.231)
    ap.add_argument("--out", type=str, default="out_ax_scan")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== ax scan with exact period constraints (circum ⋆1) ===")
    print(f"Lx={args.Lx} Ly={args.Ly}  n/ax={args.n}  jitter={args.jitter}")
    print(f"ax∈[{args.ax_min},{args.ax_max}] steps={args.ax_steps}   ay={args.ay}")
    print(f"target sin^2θ_W ≈ {args.target_sin2}")

    df = scan_ax(Lx=args.Lx, Ly=args.Ly, n_meshes=args.n,
                 ax_min=args.ax_min, ax_max=args.ax_max, ax_steps=args.ax_steps,
                 ay=args.ay, jitter=args.jitter, v=args.v, seed=args.seed)

    csv_path = outdir/"ax_scan.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    # find nearest to target
    idx_best = int(np.argmin(np.abs(df["sin2_mean"].values - args.target_sin2)))
    best = df.iloc[idx_best].to_dict()

    # summary
    lines = []
    lines.append("=== ax scan summary ===\n")
    lines.append(f"Lx={args.Lx} Ly={args.Ly}  n/ax={args.n}  jitter={args.jitter}\n")
    lines.append(f"ax range: [{args.ax_min},{args.ax_max}] steps={args.ax_steps}, ay={args.ay}\n")
    lines.append(f"target sin^2θ_W = {args.target_sin2:.6f}\n\n")
    lines.append("Best (by |sin2- target|):\n")
    lines.append(f"  ax = {best['ax']:.6f}\n")
    lines.append(f"  sin^2θ_W mean = {best['sin2_mean']:.6f}  ± {best['sin2_std']:.6f}\n")
    # reconstruct typical g,g',masses from means:
    lines.append(f"  g ≈ {best['g_mean']:.6f},  g' ≈ {best['gp_mean']:.6f}\n")
    lines.append(f"  mW ≈ {best['mW_mean']:.6f} GeV,  mZ ≈ {best['mZ_mean']:.6f} GeV\n")

    (outdir/"summary.txt").write_text("".join(lines), encoding="utf-8")

    # quick plot
    fig = plt.figure(figsize=(7,4))
    x = df["ax"].values
    y = df["sin2_mean"].values
    yerr = df["sin2_std"].values
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, label=r"$\sin^2\theta_W$")
    plt.axhline(args.target_sin2, linestyle="--", label="target")
    plt.xlabel(r"$a_x$ (with $a_y=1$)")
    plt.ylabel(r"$\sin^2\theta_W$")
    plt.title("EW mixing vs metric anisotropy (exact periods)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(outdir/"sin2_vs_ax.png", dpi=180)
    plt.close(fig)

    print("\n=== Scan complete ===")
    print(f"Nearest to target at ax={best['ax']:.3f} → sin^2θ_W={best['sin2_mean']:.3f} ± {best['sin2_std']:.3f}")
    print(f"Wrote:\n - {csv_path}\n - {outdir/'summary.txt'}\n - {outdir/'sin2_vs_ax.png'}")


if __name__ == "__main__":
    # Friendly defaults if run bare:
    if len(sys.argv) == 1:
        sys.argv += ["--Lx","16","--Ly","16","--n","12",
                     "--ax_min","1.3","--ax_max","2.6","--ax_steps","14",
                     "--ay","1.0","--jitter","0.08",
                     "--v","246","--out","out_ax_scan"]
    main()