#!/usr/bin/env python3
# dec_circum_ew_batch.py
#
# Circumcentric DEC electroweak couplings from geometry on a torus (T^2).
# - Irregular torus mesh generator (triangulated rectangle with periodic seams).
# - Circumcentric Hodge star on 1-forms: (*_1)_{e,e} = |e*| / |e|
# - Harmonic 1-forms (omega_x, omega_y) with unit periods on x/y cycles:
#   solve min_{alpha} ||d alpha||^2 with circulation constraints; then
#   project to co-closed space to get Hodge-harmonic reps.
# - K_ij = <omega_i, *_1 omega_j>; whiten by eigendecomposition; set
#   g = 2 pi / sqrt(lambda1), g' = 2 pi / sqrt(lambda2)
# - Compute mW, mZ, sin^2(theta_W), rho; batch across meshes; save CSV + plots.

import argparse
from dataclasses import dataclass
import numpy as np
from numpy.linalg import norm
from scipy.sparse import coo_matrix, csr_matrix, diags, vstack, hstack, identity
from scipy.sparse.linalg import spsolve, lsqr, eigsh
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import math
np.set_printoptions(precision=6, suppress=True)

# ---------- geometry helpers ----------

def tri_area(p, q, r):
    # 2D cross returns scalar in numpy >=2.0; wrap for safety
    return 0.5 * abs(np.cross(q - p, r - p))

def periodic_grid_torus(Lx, Ly, jitter=0.0, seed=None):
    """Build an irregular periodic rectangular grid (V,E,F) on T^2 with 2 triangles per cell."""
    rng = np.random.default_rng(seed)
    xs = np.arange(Lx)
    ys = np.arange(Ly)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    pts = np.stack([X.ravel(), Y.ravel()], axis=1).astype(float)

    if jitter > 0:
        pts += rng.uniform(-jitter, jitter, size=pts.shape)

    # map to unit torus square [0,1)^2 (only for geometry — topology stays periodic)
    pts[:, 0] /= Lx
    pts[:, 1] /= Ly

    def vid(x, y):
        return (y % Ly) * Lx + (x % Lx)

    V = pts
    faces = []
    # two triangles per cell: (x,y)-(x+1,y)-(x,y+1) and (x+1,y+1)-(x,y+1)-(x+1,y)
    for y in range(Ly):
        for x in range(Lx):
            v00 = vid(x, y)
            v10 = vid(x+1, y)
            v01 = vid(x, y+1)
            v11 = vid(x+1, y+1)
            faces.append([v00, v10, v01])
            faces.append([v11, v01, v10])
    F = np.array(faces, dtype=int)

    # build undirected edge list with (a<b) convention and incidence for d0 (V->E) and d1 (E->F)
    edges_set = {}
    for f_idx, (a,b,c) in enumerate(F):
        tri = [a,b,c]
        for i in range(3):
            u = tri[i]
            v = tri[(i+1)%3]
            a_, b_ = (u,v) if u < v else (v,u)
            if (a_, b_) not in edges_set:
                edges_set[(a_,b_)] = []
            edges_set[(a_,b_)].append((f_idx, u, v))  # store oriented use in face

    E_list = list(edges_set.keys())
    E_map = {e:i for i,e in enumerate(E_list)}
    E = np.array(E_list, dtype=int)
    Vn = V.shape[0]
    En = E.shape[0]
    Fn = F.shape[0]

    # d0: V->E  (oriented each edge from a->b)
    rows, cols, data = [], [], []
    for ei,(a,b) in enumerate(E):
        rows += [ei, ei]
        cols += [a, b]
        data += [-1.0, +1.0]
    d0 = coo_matrix((data,(rows,cols)), shape=(En,Vn)).tocsr()

    # d1: E->F  (for each face, oriented boundary; we need consistent signs)
    # choose face orientation as (a->b, b->c, c->a) positive
    rows, cols, data = [], [], []
    for f_idx,(a,b,c) in enumerate(F):
        for (u,v,sgn) in [(a,b,+1),(b,c,+1),(c,a,+1)]:
            a_, b_ = (u,v) if u < v else (v,u)
            ei = E_map[(a_,b_)]
            # if we flipped order to match stored edge (a_<b_), adjust sign
            if u < v:
                s = +1
            else:
                s = -1
            rows.append(f_idx)
            cols.append(ei)
            data.append(sgn * s)
    d1 = coo_matrix((data,(rows,cols)), shape=(Fn,En)).tocsr()

    return V, E, F, d0, d1

def circumcenter(p, q, r):
    """Circumcenter of triangle pqr in R^2."""
    a = q - p
    b = r - p
    adot = np.dot(a,a)
    bdot = np.dot(b,b)
    a_perp = np.array([a[1], -a[0]])
    b_perp = np.array([b[1], -b[0]])
    M = np.column_stack([a_perp, b_perp])
    rhs = np.array([adot/2.0, bdot/2.0])
    try:
        uv = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        # degenerate -> use centroid as fallback to avoid crashes
        return (p + q + r)/3.0
    return p + uv[0]*a_perp + uv[1]*b_perp

def circumcentric_star_1(V, E, F):
    """Return diag(*_1) for 1-forms: |dual_edge| / |edge| using circumcenters."""
    # build adjacency: for each undirected edge, find the two incident faces and their circumcenters
    # first map edges to face list with oriented use already in periodic_grid_torus
    # rebuild a face->edges map:
    edge_to_faces = {tuple(e): [] for e in E}
    # index edges for quick lookup
    E_map = {tuple(sorted((a,b))): i for i,(a,b) in enumerate(E)}
    # faces incidence
    for f_idx,(a,b,c) in enumerate(F):
        tri = [a,b,c]
        for i in range(3):
            u = tri[i]
            v = tri[(i+1)%3]
            key = tuple(sorted((u,v)))
            ei = E_map[key]
            edge_to_faces[tuple(E[ei])].append(f_idx)

    # precompute circumcenters
    C = np.zeros((F.shape[0], 2))
    for f_idx,(a,b,c) in enumerate(F):
        C[f_idx] = circumcenter(V[a], V[b], V[c])

    star_diag = np.zeros(E.shape[0])
    for ei,(a,b) in enumerate(E):
        # length of primal edge
        ell = norm(V[b]-V[a])
        # incident faces
        inc = edge_to_faces[(a,b)]
        if len(inc) == 2:
            f1, f2 = inc
            dual_len = norm(C[f2] - C[f1])
        elif len(inc) == 1:
            # boundary-like (shouldn't happen on torus), fallback small dual
            dual_len = 1e-12
        else:
            dual_len = 1e-12
        star_diag[ei] = dual_len / (ell + 1e-15)
    return star_diag  # 1D array for diag

# ---------- harmonic 1-forms on T^2 ----------

def cycle_incidence(d0, Lx, Ly, V):
    """Return two vertex indicator rows c_x, c_y ∈ R^{1×V} for +1 jump around x- and y-cycles.
       We enforce periods via edge-space constraints; here we create edge RHS later."""
    Vn = V.shape[0]
    # Not used directly; will build edge constraints by summing potential drops along a seam.
    cx = np.zeros(Vn); cy = np.zeros(Vn)
    return cx, cy

def solve_harmonic_1form(d0, d1, star0_diag, star1_diag, Lx, Ly, V, which='x'):
    """
    Construct a harmonic 1-form with unit period along the chosen cycle.
    We solve in edge space: find 1-form omega minimizing <omega, *_1 omega>
    subject to:
      - co-closed: d1 @ omega = 0
      - period constraint: sum_{edges in cycle} omega(e) = 1
    Implement via Lagrange multipliers / constrained least squares:
      minimize 0.5 * omega^T S omega
      s.t. A omega = b, where A stacks d1 and a single period row.
    """
    En = d0.shape[0]
    Fn = d1.shape[0]
    S = diags(star1_diag)  # *_1 on 1-forms

    # Build a single-cycle “period” row in edge space:
    # For T^2 embedded as periodic grid, we approximate the x-cycle as all edges whose
    # start x < end x across the seam (wrap); similarly for y. On irregular meshes,
    # this is a crude but workable proxy.
    # Here we derive a signed incidence by sampling edge direction in parameter space.
    # Recover edge vectors in param-coords:
    # We assume V stored in [0,1)^2 param coordinates (periodic).
    # Build average direction vector per edge and pick cycle by dot with ex or ey.
    e_dir = np.zeros((En, 2))
    for ei,(a,b) in enumerate(E):
        va, vb = V[a], V[b]
        dv = vb - va
        # unwrap small periodic jumps (choose shortest torus displacement)
        for j in (0,1):
            if dv[j] > 0.5:  dv[j] -= 1.0
            if dv[j] < -0.5: dv[j] += 1.0
        e_dir[ei] = dv
    ex = np.array([1.0,0.0])
    ey = np.array([0.0,1.0])
    if which == 'x':
        w = e_dir @ ex  # projection on x
    else:
        w = e_dir @ ey  # projection on y

    # Build a period row: weight edges by sign of projection; normalize so that
    # sum(period_row * omega) ≈ integral around that cycle.
    period_row = w
    # Normalize to give a reasonably scaled constraint; we set RHS=1
    if norm(period_row) < 1e-12:
        period_row = np.zeros_like(w)
        period_row[0] = 1.0

    # Constraint matrix A omega = b:
    # d1 omega = 0 (Fn equations), and period_row @ omega = 1
    A_top = d1
    A_bot = coo_matrix(period_row.reshape(1, -1)).tocsr()
    A = vstack([A_top, A_bot], format='csr')
    b = np.zeros(Fn + 1)
    b[-1] = 1.0

    # Solve the constrained quadratic program via normal equations:
    # Minimize 0.5 * omega^T S omega subject to A omega = b
    # KKT system: [ S  A^T ][omega] = [0]
    #             [ A   0  ][lambda]= [b]
    # We solve in least-squares stable form: omega = S^{-1/2} z; then solve
    #   minimize 0.5 ||z||^2 s.t. A S^{-1/2} z = b  => z = (A S^{-1/2})^+ b
    s_half_inv = diags(1.0/np.sqrt(star1_diag + 1e-18))
    ASm = A @ s_half_inv
    z = lsqr(ASm, b, atol=1e-12, btol=1e-12, iter_lim=2000)[0]
    omega = s_half_inv @ z

    # Project to co-closed explicitly to kill numerical residue: enforce d1 omega = 0 by
    # solving min ||omega - d1^T y|| with constraint d1(omega - d1^T y)=0 → set y = (d1 d1^T)^-1 d1 omega
    # but d1 d1^T = face Laplacian; better: subtract exact component from a potential phi: omega ← omega - d0 phi
    # and then enforce co-closed by least squares:
    # Compute exact component via solving (d0^T S d0) phi = d0^T S omega  (Hodge projection)
    star0 = diags(star0_diag)
    H = d0.T @ (diags(star1_diag) @ d0) + 1e-12*identity(d0.shape[1])
    rhs = d0.T @ (diags(star1_diag) @ omega)
    phi = lsqr(H, rhs, atol=1e-12, btol=1e-12, iter_lim=2000)[0]
    omega_h = omega - d0 @ phi  # remove exact part

    # Re-enforce period (a tiny rescale) so <period_row,omega_h>=1
    c = period_row @ omega_h
    if abs(c) > 1e-12:
        omega_h = omega_h / c
    return omega_h

# ---------- electroweak extraction ----------

def ew_from_geometry(V, E, F, d0, d1, v_higgs=246.0):
    Vn, En, Fn = V.shape[0], E.shape[0], F.shape[0]

    # scalar Hodge star (0-forms): lumped area per vertex
    star0 = np.zeros(Vn)
    A_tot = 0.0
    vert_area = np.zeros(Vn)
    for (a,b,c) in F:
        p,q,r = V[a], V[b], V[c]
        A = tri_area(p,q,r)
        A_tot += A
        vert_area[a] += A/3.0
        vert_area[b] += A/3.0
        vert_area[c] += A/3.0
    star0_diag = vert_area

    # circumcentric star on 1-forms
    star1_diag = circumcentric_star_1(V, E, F)

    # harmonic 1-forms (unit periods)
    omega_x = solve_harmonic_1form(d0, d1, star0_diag, star1_diag, None, None, V, which='x')
    omega_y = solve_harmonic_1form(d0, d1, star0_diag, star1_diag, None, None, V, which='y')

    # kinetic 2×2
    S1 = diags(star1_diag)
    Kxx = float(omega_x @ (star1_diag * omega_x))
    Kyy = float(omega_y @ (star1_diag * omega_y))
    Kxy = float(omega_x @ (star1_diag * omega_y))
    K = np.array([[Kxx, Kxy],[Kxy, Kyy]], dtype=float)

    # whiten (diagonalize)
    w, U = np.linalg.eigh(K)         # w >= 0
    w = np.clip(w, 1e-18, None)
    # couplings from eigenvalues (basis-invariant)
    g  = 2.0*np.pi / np.sqrt(w[0])
    gp = 2.0*np.pi / np.sqrt(w[1])

    # masses & observables
    mW = 0.5 * g * v_higgs
    mZ = 0.5 * np.sqrt(g**2 + gp**2) * v_higgs
    s2 = gp**2 / (g**2 + gp**2)
    rho = (mW/(mZ*np.sqrt(1.0 - s2)))**2

    return dict(
        A_tot=A_tot,
        star1_min=float(star1_diag.min()), star1_max=float(star1_diag.max()),
        Kxx=Kxx, Kyy=Kyy, Kxy=Kxy,
        lam1=w[0], lam2=w[1],
        g=g, gp=gp, mW=mW, mZ=mZ, sin2=s2, rho=rho
    )

# ---------- batch runner ----------

@dataclass
class EWResult:
    seed:int; Lx:int; Ly:int; jitter:float
    A_tot:float; Kxx:float; Kyy:float; Kxy:float
    lam1:float; lam2:float; g:float; gp:float
    mW:float; mZ:float; sin2:float; rho:float

def run_batch(n_meshes=12, Lx=12, Ly=12, jitter=0.25, v=246.0, outdir="out_ew_circum"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(n_meshes):
        seed = 2025_0901 + k
        V,E,F,d0,d1 = periodic_grid_torus(Lx,Ly,jitter=jitter, seed=seed)
        res = ew_from_geometry(V,E,F,d0,d1,v_higgs=v)
        rows.append(EWResult(
            seed=seed, Lx=Lx, Ly=Ly, jitter=jitter,
            A_tot=res['A_tot'],
            Kxx=res['Kxx'], Kyy=res['Kyy'], Kxy=res['Kxy'],
            lam1=res['lam1'], lam2=res['lam2'],
            g=res['g'], gp=res['gp'],
            mW=res['mW'], mZ=res['mZ'], sin2=res['sin2'], rho=res['rho']
        ))

    # save CSV
    csv_path = Path(outdir, "ew_circum_batch.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed","Lx","Ly","jitter","A_tot","Kxx","Kyy","Kxy","lam1","lam2","g","gprime","mW_GeV","mZ_GeV","sin2_thetaW","rho"])
        for r in rows:
            w.writerow([r.seed,r.Lx,r.Ly,r.jitter,r.A_tot,r.Kxx,r.Kyy,r.Kxy,r.lam1,r.lam2,r.g,r.gp,r.mW,r.mZ,r.sin2,r.rho])

    # quick stats
    s2 = np.array([r.sin2 for r in rows])
    rho = np.array([r.rho for r in rows])
    mW = np.array([r.mW for r in rows])
    mZ = np.array([r.mZ for r in rows])

    print(f"=== Circumcentric DEC EW over {n_meshes} meshes ===")
    print(f"sin^2(theta_W): mean={s2.mean():.6f} ± {s2.std(ddof=1):.6f}")
    print(f"rho:            mean={rho.mean():.6f} ± {rho.std(ddof=1):.6f}")
    print(f"mW (GeV):       mean={mW.mean():.3f} ± {mW.std(ddof=1):.3f}")
    print(f"mZ (GeV):       mean={mZ.mean():.3f} ± {mZ.std(ddof=1):.3f}")
    print(f"Wrote: {csv_path}")

    # plots
    fig, axs = plt.subplots(1,3, figsize=(12,3.5))
    axs[0].hist(s2, bins=10); axs[0].set_xlabel(r"$\sin^2\theta_W$"); axs[0].axvline(0.231, ls="--", lw=1)
    axs[1].hist(mW, bins=10); axs[1].set_xlabel(r"$m_W$ [GeV]")
    axs[2].hist(mZ, bins=10); axs[2].set_xlabel(r"$m_Z$ [GeV]")
    fig.tight_layout()
    fig.savefig(Path(outdir,"quick_histograms.png"), dpi=150)

    fig2, ax2 = plt.subplots(figsize=(4.5,4))
    ax2.scatter(mW, mZ, s=20, alpha=0.7)
    ax2.set_xlabel(r"$m_W$ [GeV]"); ax2.set_ylabel(r"$m_Z$ [GeV]")
    fig2.tight_layout()
    fig2.savefig(Path(outdir,"mW_vs_mZ.png"), dpi=150)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="number of meshes")
    ap.add_argument("--Lx", type=int, default=12)
    ap.add_argument("--Ly", type=int, default=12)
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--v", type=float, default=246.0, help="Higgs vev used for masses")
    ap.add_argument("--out", type=str, default="out_ew_circum")
    args = ap.parse_args()
    run_batch(n_meshes=args.n, Lx=args.Lx, Ly=args.Ly, jitter=args.jitter, v=args.v, outdir=args.out)

if __name__ == "__main__":
    main()