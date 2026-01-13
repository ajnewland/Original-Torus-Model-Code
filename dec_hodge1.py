#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEC bosons with Hodge decomposition on a triangulated 2-torus (T^2).

Key conventions (shapes):
  - d0 : (NE x NV)  : 0-forms -> 1-forms
  - d1 : (NF x NE)  : 1-forms -> 2-forms
  - *0 : (NV x NV)  diagonal vertex Hodge
  - *1 : (NE x NE)  diagonal edge Hodge (circumcentric dual)
  - *2 : (NF x NF)  diagonal face Hodge (area)

Operators:
  - Scalar Laplacian on 0-forms: L0 = *0^{-1} d0^T *1 d0
  - 1-form Hodge Laplacian:     Δ1 = d0 *0^{-1} d0^T *1 + *1^{-1} d1^T *2 d1

We extract the 2D nullspace (harmonic 1-forms) of Δ1, measure periods along the
two fundamental cycles, and map them to electroweak couplings via:
  g  ~ 2π / |∮_x ω_x|,   g' ~ 2π / |∮_y ω_y|
Then mW = g v / 2, mZ = sqrt(g^2 + g'^2) v / 2, sin^2θW = g'^2/(g^2+g'^2), ρ=1.

Usage:
  Single run:
    python dec_hodge_fixed.py --mode single --Lx 12 --Ly 12 --jitter 0.25 --v 246 --out out_single
  Batch (universality check):
    python dec_hodge_fixed.py --mode batch --n 20 --Lx 12 --Ly 12 --jitter 0.25 --v 246 --out out_batch
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, diags, csc_matrix
from scipy.sparse.linalg import eigsh

# --------------------- small helpers ---------------------

def tri_area(p, q, r):
    """Triangle area in 2D (NumPy 2.0+ safe via 3D cross)."""
    P = np.array([p[0], p[1], 0.0])
    Q = np.array([q[0], q[1], 0.0])
    R = np.array([r[0], r[1], 0.0])
    return 0.5 * np.linalg.norm(np.cross(Q - P, R - P))

def normalize_star1(vec, star1, eps=1e-14):
    """Normalize a 1-form vector with *1 inner product."""
    n2 = float(vec.T @ (star1 @ vec))
    if n2 <= eps:
        return vec, 0.0
    return vec / np.sqrt(n2), np.sqrt(n2)

def gs_star1(vectors, star1, eps=1e-12):
    """Gram–Schmidt orthonormalization in *1 inner product."""
    basis = []
    for v in vectors:
        w = v.copy()
        for b in basis:
            proj = float(b.T @ (star1 @ w))
            w -= proj * b
        w, nrm = normalize_star1(w, star1)
        if nrm > eps:
            basis.append(w)
    return basis

# --------------------- mesh & DEC assembly ---------------------

def build_torus(Lx, Ly, jitter=0.0, seed=None):
    """
    Periodic rectangular grid (Lx x Ly) with two triangles per cell.
    Returns:
      V: (NV,2) vertices
      E: list of edges (u,v) with u<v (canonical orientation)
      F: list of faces (v0,v1,v2) oriented CCW
      edge2idx: dict (u,v)->ei
      cycles: cycle_x, cycle_y as lists of (ei, sgn) along fundamental cycles
    """
    rng = np.random.default_rng(seed)
    xs = np.arange(Lx)
    ys = np.arange(Ly)

    def vid(x, y):
        return (y % Ly) * Lx + (x % Lx)

    # base vertex coords with optional jitter inside cells
    V = np.array([(x, y) for y in ys for x in xs], dtype=float)
    if jitter > 0.0:
        V += rng.uniform(-jitter/2, jitter/2, size=V.shape)

    # faces: (x,y) cell split into two triangles along (x,y) -> (x+1,y+1)
    F = []
    for y in ys:
        for x in xs:
            v00 = vid(x, y)
            v10 = vid(x+1, y)
            v01 = vid(x, y+1)
            v11 = vid(x+1, y+1)
            # CCW orientation in the plane:
            F.append((v00, v10, v11))
            F.append((v00, v11, v01))

    # edges: unique undirected, canonical (u<v)
    edge_set = set()
    for (a,b,c) in F:
        for u,v in [(a,b),(b,c),(c,a)]:
            edge_set.add((min(u,v), max(u,v)))
    E = sorted(list(edge_set))
    edge2idx = {e:i for i,e in enumerate(E)}

    # fundamental cycles: along +x at fixed y=0, and along +y at fixed x=0
    cycle_x = []
    y0 = 0
    for x in xs:
        u = vid(x, y0); v = vid(x+1, y0)
        e = (min(u,v), max(u,v))
        ei = edge2idx[e]
        # sign is +1 if we traverse in the canonical edge direction (u<v), else -1
        sgn = +1 if u < v else -1
        cycle_x.append((ei, sgn))

    cycle_y = []
    x0 = 0
    for y in ys:
        u = vid(x0, y); v = vid(x0, y+1)
        e = (min(u,v), max(u,v))
        ei = edge2idx[e]
        sgn = +1 if u < v else -1
        cycle_y.append((ei, sgn))

    return V, E, F, edge2idx, cycle_x, cycle_y

def assemble_d0_d1(V, E, F):
    """
    Assemble d0 (NE x NV) and d1 (NF x NE).
    d0: for edge e=(u,v) with u<v (canonical), (d0 φ)_e = φ[v] - φ[u]
         => row e has -1 at u, +1 at v.
    d1: for face f=(a,b,c) CCW, (d1 ω)_f = ω_ab + ω_bc + ω_ca with signs matching face boundary.
         => row f has +1/-1 at edges depending on whether edge aligns with boundary orientation.
    """
    NV = V.shape[0]
    NE = len(E)
    NF = len(F)

    # d0
    r, c, d = [], [], []
    for ei, (u,v) in enumerate(E):
        r += [ei, ei]
        c += [u, v]
        d += [-1.0, +1.0]
    d0 = coo_matrix((d, (r, c)), shape=(NE, NV)).tocsr()

    # helper: map edge to ei
    edge2idx = {e:i for i,e in enumerate(E)}

    # d1
    r, c, d = [], [], []
    for fi, (a,b,c0) in enumerate(F):
        boundary = [(a,b), (b,c0), (c0,a)]
        for (x,y) in boundary:
            e = (min(x,y), max(x,y))
            ei = edge2idx[e]
            # edge sign: +1 if traversing along canonical (min->max) equals (x->y)
            sgn = +1 if x < y else -1
            r.append(fi); c.append(ei); d.append(sgn)
    d1 = coo_matrix((d, (r, c)), shape=(NF, NE)).tocsr()
    return d0, d1

def star0_uniform(V, F):
    """*0: vertex dual area = 1/3 sum of incident face areas."""
    NV = V.shape[0]
    areas = np.zeros(NV)
    for (a,b,c) in F:
        A = tri_area(V[a], V[b], V[c])
        areas[a] += A/3; areas[b] += A/3; areas[c] += A/3
    areas = np.clip(areas, 1e-15, None)
    return diags(areas)

def star2_face_areas(V, F):
    """*2: face areas on diagonal."""
    NF = len(F)
    diag = np.zeros(NF)
    for i,(a,b,c) in enumerate(F):
        diag[i] = tri_area(V[a], V[b], V[c])
    diag = np.clip(diag, 1e-15, None)
    return diags(diag)

def star1_circumcentric(V, E, F):
    """
    *1 for edges: |dual edge| / |edge| using centroids of adjacent faces as dual nodes.
    (On near-rectangular meshes this is stable; swap to true circumcenters if desired.)
    """
    NE = len(E)
    # face centroids
    face_center = np.array([(V[a]+V[b]+V[c])/3.0 for (a,b,c) in F])
    # adjacency: for each edge, find the two incident faces
    edge2idx = {e:i for i,e in enumerate(E)}
    # build edge->faces
    edge_faces = [[] for _ in range(NE)]
    for fi,(a,b,c) in enumerate(F):
        for (x,y) in [(a,b),(b,c),(c,a)]:
            e = (min(x,y), max(x,y))
            edge_faces[edge2idx[e]].append(fi)
    # diagonal entries
    diag = np.zeros(NE)
    for ei,(u,v) in enumerate(E):
        le = np.linalg.norm(V[v]-V[u])
        fns = edge_faces[ei]
        if len(fns)==2:
            de = np.linalg.norm(face_center[fns[1]] - face_center[fns[0]])
        else:
            de = le
        diag[ei] = de / max(le, 1e-15)
    return diags(np.clip(diag, 1e-15, None))

# --------------------- DEC Laplacians ---------------------

def laplacian_0form(d0, star0, star1):
    """L0 = *0^{-1} d0^T *1 d0  (NV x NV)."""
    star0_inv = diags(1.0 / star0.diagonal())
    return star0_inv @ (d0.T @ (star1 @ d0))

def laplacian_1form(d0, d1, star0, star1, star2):
    """Δ1 = d0 *0^{-1} d0^T *1 + *1^{-1} d1^T *2 d1  (NE x NE)."""
    star0_inv = diags(1.0 / star0.diagonal())
    star1_inv = diags(1.0 / star1.diagonal())
    termA = d0 @ (star0_inv @ d0.T)    # NE x NE
    termA = termA @ star1              # NE x NE
    termB = star1_inv @ (d1.T @ (star2 @ d1))  # NE x NE
    return (termA + termB).tocsc()

# --------------------- harmonic extraction & periods ---------------------

def harmonic_basis(Delta1, star1, k=6, tol=1e-10):
    """Return ~2 harmonic 1-forms (nullspace) of Δ1, *1-orthonormalized."""
    NE = star1.shape[0]
    vals, vecs = eigsh(Delta1, k=min(k, NE-2), which="SM")
    idx = np.where(vals < tol)[0]
    if len(idx) < 2:
        idx = np.argsort(vals)[:2]
    raw = [vecs[:,i] for i in idx[:2]]
    return gs_star1(raw, star1), vals[idx[:2]]

def period(omega, cycle_edges):
    """Discrete line integral along a cycle: sum sgn * ω_e."""
    s = 0.0
    for ei, sgn in cycle_edges:
        s += sgn * omega[ei]
    return float(s)

def periods_to_couplings(Px, Py, eps=1e-12):
    two_pi = 2.0*np.pi
    g  = two_pi / max(abs(Px), eps)
    gp = two_pi / max(abs(Py), eps)
    return g, gp

def ew_obs(g, gp, v):
    mW = 0.5 * g * v
    mZ = 0.5 * np.sqrt(g*g + gp*gp) * v
    sin2 = (gp*gp) / (g*g + gp*gp) if (g>0 or gp>0) else np.nan
    rho = 1.0
    return mW, mZ, sin2, rho

# --------------------- pipeline ---------------------

def run_single(Lx, Ly, jitter, v, outdir, seed=None):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)

    V, E, F, edge2idx, cycle_x, cycle_y = build_torus(Lx, Ly, jitter=jitter, seed=seed)
    d0, d1 = assemble_d0_d1(V, E, F)
    star0 = star0_uniform(V, F)
    star1 = star1_circumcentric(V, E, F)
    star2 = star2_face_areas(V, F)

    # scalar spectrum
    L0 = laplacian_0form(d0, star0, star1)      # NV x NV
    NV = V.shape[0]; NE = len(E); NF = len(F)
    k0 = min(20, NV-1)
    vals0, _ = eigsh(L0, k=k0, which="SM")
    vals0 = np.sort(vals0)
    lam1 = float(vals0[1]) if len(vals0)>1 else float(vals0[0])
    Atot = float(np.sum(star2.diagonal()))
    lam1_hat = lam1 * Atot

    # 1-form Laplacian
    Delta1 = laplacian_1form(d0, d1, star0, star1, star2)  # NE x NE
    basis, _ = harmonic_basis(Delta1, star1, k=6, tol=1e-10)
    if len(basis) < 2:
        print("Warning: <2 harmonic modes found; consider larger k or check mesh.")

    # align basis roughly with cycles: pick vector with larger |Px| as ω_x
    W = np.stack(basis, axis=1)  # NE x 2
    Pxs = np.array([period(W[:,i], cycle_x) for i in range(W.shape[1])])
    Pys = np.array([period(W[:,i], cycle_y) for i in range(W.shape[1])])
    if abs(Pxs[0]) >= abs(Pxs[1]):
        ix, iy = 0,1
    else:
        ix, iy = 1,0
    omega_x = W[:, ix]; omega_y = W[:, iy]
    Px = period(omega_x, cycle_x)
    Py = period(omega_y, cycle_y)

    g, gp = periods_to_couplings(Px, Py)
    mW, mZ, s2, rho = ew_obs(g, gp, v)

    # save CSV
    summary = dict(Lx=Lx, Ly=Ly, NV=NV, NE=NE, NF=NF, jitter=jitter, seed=(seed if seed is not None else -1),
                   A_tot=Atot, lambda1=lam1, lambda1_hat=lam1_hat,
                   Px=Px, Py=Py, g=g, gp=gp, v=v, mW=mW, mZ=mZ, sin2thetaW=s2, rho=rho)
    pd.DataFrame([summary]).to_csv(out/"summary.csv", index=False)

    # plot scalar spectrum
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(len(vals0)), vals0, 'o')
    plt.xlabel("mode index")
    plt.ylabel("λ (0-form)")
    plt.title("Scalar Laplacian spectrum (first modes)")
    plt.tight_layout()
    plt.savefig(out/"scalar_spectrum.png", dpi=140)
    plt.close()

    # console
    print("=== DEC bosons (Hodge, single) ===")
    print(f"Lx={Lx} Ly={Ly}  V={NV} E={NE} F={NF}")
    print(f"A_tot={Atot:.6f}  lambda1={lam1:.6e}  lambda1_hat={lam1_hat:.6f}")
    print(f"Periods: Px={Px:.6f}  Py={Py:.6f}")
    print(f"Couplings: g={g:.6f}  g'={gp:.6f}")
    print(f"Masses: mW={mW:.6f}  mZ={mZ:.6f}  sin^2θW={s2:.6f}  ρ={rho:.6f}")
    print(f"Wrote: {out.resolve()}")
    return summary

def run_batch(n, Lx, Ly, jitter, v, outdir, seed=12345):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sd = int(rng.integers(0, 2**31-1))
        s = run_single(Lx, Ly, jitter, v, out/f"mesh_{i:03d}", seed=sd)
        rows.append(dict(idx=i, **s))
    df = pd.DataFrame(rows)
    df.to_csv(out/"meshes.csv", index=False)

    mu = df[["lambda1","lambda1_hat","g","gp","mW","mZ","sin2thetaW"]].mean()
    sg = df[["lambda1","lambda1_hat","g","gp","mW","mZ","sin2thetaW"]].std()

    lines = []
    lines.append("=== Hodge-derived couplings: batch summary ===\n")
    for k in ["lambda1","lambda1_hat","g","gp","mW","mZ","sin2thetaW"]:
        lines.append(f"{k:>12s}: mean={mu[k]:.6f}  std={sg[k]:.6f}\n")
    Path(out/"summary.txt").write_text("".join(lines), encoding="utf-8")

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.scatter(df["g"], df["gp"], s=18)
    plt.xlabel("g"); plt.ylabel("g'")
    plt.title("Couplings from periods")
    plt.subplot(1,2,2)
    plt.hist(df["sin2thetaW"], bins=12, alpha=0.85)
    plt.xlabel("sin^2(theta_W)")
    plt.tight_layout()
    plt.savefig(out/"quick_plots.png", dpi=140)
    plt.close()

    print("=== DEC bosons (Hodge, batch) ===")
    print((out/"summary.txt").read_text(encoding="utf-8"))

# --------------------- CLI ---------------------

def parse_args():
    ap = argparse.ArgumentParser(description="DEC bosons via Hodge decomposition on T^2")
    ap.add_argument("--mode", choices=["single","batch"], default="single")
    ap.add_argument("--Lx", type=int, default=12)
    ap.add_argument("--Ly", type=int, default=12)
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--v", type=float, default=246.0)
    ap.add_argument("--n", type=int, default=20, help="batch size")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=str, default="out_hodge_fixed")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.mode == "single":
        run_single(args.Lx, args.Ly, args.jitter, args.v, args.out, seed=args.seed)
    else:
        run_batch(args.n, args.Lx, args.Ly, args.jitter, args.v, args.out, seed=args.seed)

if __name__ == "__main__":
    main()