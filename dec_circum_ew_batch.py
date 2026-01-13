#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEC photon/EW on irregular triangulated T^2 with circumcentric Hodge stars.

Pipeline (per mesh):
  1) Build irregular torus triangulation (Lx x Ly fundamental grid with jitter; each cell split into 2 triangles).
  2) Oriented incidence d0 (V×E), d1 (E×F); lengths & areas.
  3) Circumcentric duals → Hodge stars: *0 (Voronoi area), *1 (dual length / edge length), *2 (1/face area).
  4) Constrained harmonic solver for 1-forms:
       minimize  E(a) = ||d a||^2_{*2} + ||δ a||^2_{*0}  subject to  cᵀ a = 2π,
     with δ = *0^{-1} d0 *1, co-closedness encouraged by the energy term.
     Solve KKT system to get a_x (x-cycle), a_y (y-cycle).
  5) Kinetic metric on harmonic subspace: K_ij = a_iᵀ *1 a_j.
     Couplings: g  = 2π / sqrt(K_xx),  g' = 2π / sqrt(K_yy).
  6) Masses: m_W = ½ g v;  m_Z = ½ v sqrt(g² + g'²);  sin²θ_W = g'²/(g²+g'²);  ρ = m_W² / (m_Z²(1 - sin²θ_W)).

Outputs:
  - CSV with one row per mesh
  - Aggregate stats printed
  - Quick plots saved (histograms of sin²θ_W and ρ)

Author: (your project)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from scipy.sparse import coo_matrix, csr_matrix, diags, identity
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 130

# ---------------------------
# Utilities
# ---------------------------

def tri_area(p, q, r):
    # 2D area: 1/2 | (q-p) x (r-p) |
    return 0.5 * abs(np.cross(q - p, r - p))

def circumcenter(p, q, r):
    """Circumcenter of triangle (2D) in R^2."""
    a = q - p
    b = r - p
    adot = a.dot(a)
    bdot = b.dot(b)
    cross = a[0]*b[1] - a[1]*b[0]
    if abs(cross) < 1e-14:
        # near-colinear fallback: use centroid
        return (p + q + r) / 3.0
    ux = p[0] + ( (bdot*(a[1]) - adot*(b[1])) / (2*cross) )
    uy = p[1] + ( (-bdot*(a[0]) + adot*(b[0])) / (2*cross) )
    return np.array([ux, uy])

def torus_wrap(coords):
    """Wrap points to [0,1)^2 fundamental domain (for visualization metrics only)."""
    return coords - np.floor(coords)

# ---------------------------
# Mesh generation (irregular T^2)
# ---------------------------

@dataclass
class Mesh:
    V: int
    E: int
    F: int
    coords: np.ndarray     # (V,2)
    faces: np.ndarray      # (F,3) vertex indices
    d0: csr_matrix         # (V,E)
    d1: csr_matrix         # (E,F)
    e2v: np.ndarray        # (E,2) oriented (tail, head)
    f2e: np.ndarray        # (F,3) oriented edge indices for face
    elen: np.ndarray       # (E,) primal edge lengths
    farea: np.ndarray      # (F,) face areas
    ccent: np.ndarray      # (F,2) circumcenters

def build_irregular_torus(Lx=12, Ly=12, jitter=0.25, seed=0):
    rng = np.random.default_rng(seed)
    # base grid
    xs = (np.arange(Lx) + rng.uniform(-jitter, jitter, size=Lx)) / Lx
    ys = (np.arange(Ly) + rng.uniform(-jitter, jitter, size=Ly)) / Ly
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)

    def vid(i,j): return (i % Lx)*Ly + (j % Ly)

    faces = []
    # split each cell into two triangles [i,j] square: lower-left index (i,j)
    for i in range(Lx):
        for j in range(Ly):
            v00 = vid(i, j)
            v10 = vid(i+1, j)
            v01 = vid(i, j+1)
            v11 = vid(i+1, j+1)
            # choose diagonal based on parity to avoid bias
            if (i + j) % 2 == 0:
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])
            else:
                faces.append([v00, v10, v01])
                faces.append([v10, v11, v01])
    faces = np.array(faces, dtype=int)
    F = faces.shape[0]
    V = coords.shape[0]

    # build edges (undirected set), then orient by face
    edge_dict = {}
    e2v_list = []
    f2e = np.empty((F,3), dtype=int)
    ecount = 0
    for fi, (a,b,c) in enumerate(faces):
        tri = [(a,b), (b,c), (c,a)]
        for k,(u,v) in enumerate(tri):
            key = (min(u,v), max(u,v))
            if key not in edge_dict:
                edge_dict[key] = ecount
                e2v_list.append([key[0], key[1]])
                ecount += 1
            eid = edge_dict[key]
            # orientation: +1 if edge direction matches (u->v), else -1 if (v->u)
            if (e2v_list[eid][0]==u and e2v_list[eid][1]==v):
                f2e[fi,k] = eid
            else:
                f2e[fi,k] = -eid-1  # encode flipped sign as negative index
    E = ecount
    e2v = np.array(e2v_list, dtype=int)

    # incidence d0 (V×E)
    rows, cols, data = [], [], []
    for e,(u,v) in enumerate(e2v):
        rows += [u, v];  cols += [e, e];  data += [-1.0, +1.0]
    d0 = coo_matrix((data, (rows, cols)), shape=(V,E)).tocsr()

    # incidence d1 (E×F)
    rows, cols, data = [], [], []
    for f in range(F):
        for k in range(3):
            raw = f2e[f,k]
            if raw >= 0:
                e = raw; sgn = +1.0
            else:
                e = -raw-1; sgn = -1.0
            rows.append(e); cols.append(f); data.append(sgn)
    d1 = coo_matrix((data, (rows, cols)), shape=(E,F)).tocsr()

    # lengths & areas
    elen = np.linalg.norm(coords[e2v[:,1]] - coords[e2v[:,0]], axis=1)
    farea = np.array([tri_area(coords[a], coords[b], coords[c]) for (a,b,c) in faces])
    # circumcenters
    ccent = np.array([circumcenter(coords[a], coords[b], coords[c]) for (a,b,c) in faces])

    return Mesh(V=V,E=E,F=F, coords=coords, faces=faces,
                d0=d0, d1=d1, e2v=e2v, f2e=f2e,
                elen=elen, farea=farea, ccent=ccent)

# ---------------------------
# Circumcentric Hodge stars
# ---------------------------

def hodge_stars_circum(mesh: Mesh):
    V,E,F = mesh.V, mesh.E, mesh.F
    # *2 on faces (map 2-form to scalar dual): weight = 1/area
    star2 = diags(1.0 / np.maximum(mesh.farea, 1e-16))

    # *1 on edges: dual length / primal length (circumcentric dual)
    # For each edge, find its two adjacent faces (periodic faces may be >2 but here triangulated torus → exactly 2).
    # Build face adjacency per edge:
    edge_faces = [[] for _ in range(E)]
    f2e = mesh.f2e
    for f in range(F):
        for k in range(3):
            raw = f2e[f,k]
            e = raw if raw>=0 else -raw-1
            edge_faces[e].append(f)

    # dual length: distance between face circumcenters across the edge
    dual_len = np.zeros(E)
    for e,(u,v) in enumerate(mesh.e2v):
        adj = edge_faces[e]
        if len(adj)==2:
            c1 = mesh.ccent[adj[0]]
            c2 = mesh.ccent[adj[1]]
            dual_len[e] = norm(c2 - c1)
        else:
            # boundary-like artifact shouldn't happen on torus; fallback to small value
            dual_len[e] = 1e-8
    star1 = diags(dual_len / np.maximum(mesh.elen, 1e-16))

    # *0 on vertices: dual cell area (Voronoi area around vertex using circumcenters):
    # Approximate by splitting each incident triangle into 3 and summing wedge sectors
    v_area = np.zeros(V)
    for f,(a,b,c) in enumerate(mesh.faces):
        cc = mesh.ccent[f]
        for v in (a,b,c):
            # area of tri (vertex v, cc, mid-edge), simple approximation: 1/3 of face area
            v_area[v] += mesh.farea[f] / 3.0
    star0 = diags(np.maximum(v_area, 1e-16))

    # Inverses (as diagonals)
    star0_inv = diags(1.0 / np.maximum(v_area, 1e-16))
    star1_inv = diags(np.divide(mesh.elen, np.maximum(dual_len, 1e-16)))
    star2_inv = diags(np.maximum(mesh.farea, 1e-16))

    return star0, star1, star2, star0_inv, star1_inv, star2_inv

# ---------------------------
# Harmonic 1-form solver (constrained)
# ---------------------------

def laplace_1_form(mesh: Mesh, star0_inv, star1, star2_inv):
    # Hodge Laplacian on 1-forms with *1 inner-product convention:
    # Δ1 ≈ d0ᵀ *0^{-1} d0  +  d1 *2^{-1} d1ᵀ
    # This operator acts in Euclidean coordinates; we add a small diagonal for conditioning.
    term0 = mesh.d0.T @ (star0_inv @ mesh.d0)
    term2 = mesh.d1 @ (star2_inv @ mesh.d1.T)
    L1 = term0 + term2
    # precondition with *1 to reflect the <.,.>_*1 metric in linear algebra space
    # (i.e., solve (*1 L1) a = rhs to penalize with the correct norm)
    A = star1 @ L1 + 1e-12 * identity(mesh.E, format="csr")
    return A.tocsr()

def period_selector(mesh: Mesh, axis="x"):
    """
    Build a simple period constraint vector c (length E) that sums A along a column (x-cycle)
    or a row (y-cycle) of the underlying grid. We approximate using edges whose midpoints
    fall in a thin slab.
    """
    # edge midpoints
    m = 0.5 * (mesh.coords[mesh.e2v[:,0]] + mesh.coords[mesh.e2v[:,1]])
    # choose a slab near x≈0 for y-cycle (wrap upwards), and near y≈0 for x-cycle (wrap sideways)
    if axis == "x":
        slab = (m[:,1] < 0.05) | (m[:,1] > 0.95)  # near bottom/top
        # prefer edges largely oriented along +x (|dx|>|dy|)
        dirv = mesh.coords[mesh.e2v[:,1]] - mesh.coords[mesh.e2v[:,0]]
        choose = slab & (np.abs(dirv[:,0]) >= np.abs(dirv[:,1]))
    else:
        slab = (m[:,0] < 0.05) | (m[:,0] > 0.95)  # near left/right
        dirv = mesh.coords[mesh.e2v[:,1]] - mesh.coords[mesh.e2v[:,0]]
        choose = slab & (np.abs(dirv[:,1]) > np.abs(dirv[:,0]))

    c = np.zeros(mesh.E)
    # give each chosen edge a sign consistent with its local axis orientation
    sgn = np.sign(dirv[:,0]) if axis=="x" else np.sign(dirv[:,1])
    sgn[sgn==0] = 1.0
    c[choose] = sgn[choose]
    # normalize to have ~unit period length
    if np.sum(np.abs(c)) > 0:
        c = c / np.sum(np.abs(c))
    return c

def solve_harmonic(mesh: Mesh, A, star1, c_vec, target=2*np.pi):
    """
    Solve KKT system:
        [ A  Cᵀ ] [ a ] = [ 0 ]
        [ C   0 ] [ λ ]   [ t ]
    where C is 1×E (period constraint), t=target (2π).
    Returns a (E,) 1-form.
    """
    E = mesh.E
    C = c_vec.reshape(1,-1)
    # build KKT in sparse blocks
    # For numerical stability, assemble as dense blocks for the 1×1 lower-right
    from scipy.sparse import bmat
    KKT = bmat([[A,        csr_matrix(C.T)],
                [csr_matrix(C),  None]], format="csr")
    rhs = np.zeros(E+1)
    rhs[-1] = target
    # fill the small (last,last) with 0 explicitly
    # spsolve handles this as a saddle-point system
    sol = spsolve(KKT, rhs)
    a = sol[:E]
    return a

# ---------------------------
# EW observables
# ---------------------------

def ew_from_K(K, v):
    # g = 2π / sqrt(K_xx), g' analogously
    Kxx = K[0,0]; Kyy = K[1,1]
    g  = 2*np.pi / np.sqrt(max(Kxx, 1e-18))
    gp = 2*np.pi / np.sqrt(max(Kyy, 1e-18))
    mW = 0.5 * g * v
    mZ = 0.5 * v * np.sqrt(g*g + gp*gp)
    s2 = (gp*gp) / (g*g + gp*gp + 1e-18)
    rho = (mW*mW) / (mZ*mZ * (1.0 - s2 + 1e-18))
    return g, gp, mW, mZ, s2, rho

# ---------------------------
# Batch runner
# ---------------------------

def run_batch(n=12, Lx=12, Ly=12, jitter=0.25, v=246.0, outdir="out_ew_circum", seed0=2025):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(n):
        seed = seed0 + k
        mesh = build_irregular_torus(Lx=Lx, Ly=Ly, jitter=jitter, seed=seed)
        star0, star1, star2, star0_inv, star1_inv, star2_inv = hodge_stars_circum(mesh)
        A = laplace_1_form(mesh, star0_inv, star1, star2_inv)

        # x-cycle harmonic with period 2π
        cx = period_selector(mesh, axis="x")
        ax = solve_harmonic(mesh, A, star1, cx, target=2*np.pi)
        # y-cycle harmonic
        cy = period_selector(mesh, axis="y")
        ay = solve_harmonic(mesh, A, star1, cy, target=2*np.pi)

        # K metric on the harmonic subspace
        # K_ij = a_iᵀ *1 a_j
        K11 = float(ax @ (star1 @ ax))
        K22 = float(ay @ (star1 @ ay))
        K12 = float(ax @ (star1 @ ay))
        K = np.array([[K11, K12],[K12, K22]])

        g, gp, mW, mZ, s2, rho = ew_from_K(K, v=v)

        rows.append({
            "seed": seed,
            "Lx": Lx, "Ly": Ly, "jitter": jitter,
            "V": mesh.V, "E": mesh.E, "F": mesh.F,
            "A_tot": float(np.sum(mesh.farea)),
            "K_xx": K11, "K_yy": K22, "K_xy": K12,
            "g": g, "gprime": gp,
            "mW": mW, "mZ": mZ, "sin2thetaW": s2, "rho": rho
        })

    # write CSV
    import csv
    csv_path = Path(outdir, "ew_circum_batch.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)

    # aggregate
    s2_vals = np.array([r["sin2thetaW"] for r in rows])
    rho_vals = np.array([r["rho"] for r in rows])
    mW_vals  = np.array([r["mW"] for r in rows])
    mZ_vals  = np.array([r["mZ"] for r in rows])

    def mean_std(x): return np.mean(x), np.std(x, ddof=1) if x.size>1 else (x[0], 0.0)

    m_s2, s_s2 = mean_std(s2_vals)
    m_rho, s_rho = mean_std(rho_vals)
    m_mW, s_mW = mean_std(mW_vals)
    m_mZ, s_mZ = mean_std(mZ_vals)

    print(f"=== Circumcentric DEC EW over {n} meshes ===")
    print(f"sin^2(theta_W): mean={m_s2:.6f} ± {s_s2:.6f}")
    print(f"rho:             mean={m_rho:.6f} ± {s_rho:.6f}")
    print(f"mW (GeV):        mean={m_mW:.3f} ± {s_mW:.3f}")
    print(f"mZ (GeV):        mean={m_mZ:.3f} ± {s_mZ:.3f}")
    print(f"Wrote: {csv_path}")

    # quick plots
    fig,axs = plt.subplots(1,3, figsize=(10,3.2))
    axs[0].hist(s2_vals, bins=10)
    axs[0].axvline(0.231, ls="--")
    axs[0].set_title(r"$\sin^2\theta_W$")
    axs[1].hist(rho_vals, bins=10)
    axs[1].axvline(1.0, ls="--")
    axs[1].set_title(r"$\rho$")
    axs[2].scatter(mW_vals, mZ_vals, s=18)
    axs[2].set_xlabel("mW")
    axs[2].set_ylabel("mZ")
    fig.tight_layout()
    fig.savefig(Path(outdir,"quick_plots.png"), dpi=160)
    plt.close(fig)

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Circumcentric DEC EW batch")
    ap.add_argument("--n", type=int, default=12, help="number of meshes")
    ap.add_argument("--Lx", type=int, default=12)
    ap.add_argument("--Ly", type=int, default=12)
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--v", type=float, default=246.0)
    ap.add_argument("--out", type=str, default="out_ew_circum")
    ap.add_argument("--seed0", type=int, default=2025)
    args = ap.parse_args()
    run_batch(n=args.n, Lx=args.Lx, Ly=args.Ly, jitter=args.jitter, v=args.v, outdir=args.out, seed0=args.seed0)

if __name__ == "__main__":
    main()