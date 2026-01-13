#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circumcentric DEC electroweak toy with auto-tuning of geometric anisotropy (ax, ay).

What it does
------------
1) Builds jittered rectangular-torus triangulations (periodic) with aspect scales (ax, ay).
2) Computes (approximate) circumcentric Hodge star on 1-forms:  *1 = diag(l_dual / l_primal).
3) Builds two geometric 1-form templates aligned with the torus cycles (x, y).
4) Forms a 2x2 “cycle stiffness” K_ij = sum_e (*1)_e * omega_i[e] * omega_j[e]
   and defines couplings g = 2π / sqrt(K_xx), g' = 2π / sqrt(K_yy).
5) Predicts mW, mZ, sin2thetaW, rho with fixed Higgs vev v.
6) In “autotune” mode, grid-searches (ax, ay) and selects the setting whose *mean ensemble*
   sin2thetaW is closest to target_sin2 (default 0.231).

Notes
-----
- This is the same logic you’ve been running, packaged with (ax, ay) grid search.
- The harmonic-cycle construction is an *approximate* geometric template (edge vectors · e_x/e_y).
  It’s enough to steer K_xx/K_yy into the right ballpark so sin^2θ_W can be targeted by (ax, ay).
- All outputs are ASCII-only; CSVs and text use UTF-8.

Run examples
------------
# 1) Single batch, fixed (ax, ay)
python ew_circum_autotune.py --mode batch --n 12 --Lx 12 --Ly 12 --jitter 0.25 \
    --v 246 --ax 1.0 --ay 1.8 --out out_batch_ax1_ay18

# 2) Auto-tune (ax, ay) over a small grid and pick the best
python ew_circum_autotune.py --mode autotune --n 12 --Lx 12 --Ly 12 --jitter 0.25 --v 246 \
    --ax_min 0.7 --ax_max 1.3 --ax_steps 5 \
    --ay_min 1.0 --ay_max 2.2 --ay_steps 7 \
    --target_sin2 0.231 --out out_autotune

# 3) Quick tiny grid (fast)
python ew_circum_autotune.py --mode autotune --n 6 --ax_min 0.8 --ax_max 1.2 --ax_steps 3 \
    --ay_min 1.2 --ay_max 2.0 --ay_steps 5 --out out_auto_small
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

# ------------------------------
# Utilities
# ------------------------------

def tri_area(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    # For 2D points, numpy cross returns a scalar if 3D; here use z-component via manual formula:
    return 0.5 * abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))

@dataclass
class DEC2:
    V: np.ndarray            # (NV, 2)
    E: np.ndarray            # (NE, 2) undirected edge (vertex ids)
    F: np.ndarray            # (NF, 3) CCW faces (vertex ids)
    face_edges: np.ndarray   # (NF, 3) edge indices (signed by orientation vs face)
    face_signs: np.ndarray   # (NF, 3) +/-1 orientation
    d0: np.ndarray           # (NE, NV) incidence (edge<-vertex)
    d1: np.ndarray           # (NF, NE) incidence (face<-edge)
    star1_diag: np.ndarray   # (NE,) circumcentric Hodge *1 diagonal (l_dual / l_primal)

# ------------------------------
# Mesh and DEC builders
# ------------------------------

def build_torus_jittered(Lx: int, Ly: int, jitter: float, ax: float, ay: float,
                         seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rectangular Lx x Ly grid on [0, ax*Lx) x [0, ay*Ly), periodic, jittered points,
    triangulated by splitting each quad into two triangles along the same diagonal.
    Returns (V, E, F).
    """
    rng = np.random.default_rng(seed)
    xs = np.arange(Lx, dtype=float)
    ys = np.arange(Ly, dtype=float)
    V = []
    for j in range(Ly):
        for i in range(Lx):
            dx = (rng.random() - 0.5) * jitter
            dy = (rng.random() - 0.5) * jitter
            V.append([(i + dx) * ax, (j + dy) * ay])
    V = np.asarray(V, dtype=float)
    def vid(i, j): return (j % Ly) * Lx + (i % Lx)

    # Triangulate: for each cell (i,j) use two triangles: (i,j)-(i+1,j)-(i,j+1) and (i+1,j+1)-(i,j+1)-(i+1,j)
    faces = []
    for j in range(Ly):
        for i in range(Lx):
            a = vid(i, j)
            b = vid(i+1, j)
            c = vid(i, j+1)
            d = vid(i+1, j+1)
            # Split by diagonal a-d: triangles (a,b,c) and (d,c,b) but ensure CCW orientation
            tri1 = [a, b, c]
            tri2 = [d, c, b]
            # fix orientation to CCW by checking signed area
            for tri in (tri1, tri2):
                p, q, r = V[tri[0]], V[tri[1]], V[tri[2]]
                signed = (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
                if signed < 0:
                    tri[1], tri[2] = tri[2], tri[1]
            faces.append(tri1)
            faces.append(tri2)
    F = np.asarray(faces, dtype=int)

    # Build undirected edge list
    edge_map = {}
    E = []
    for (a,b,c) in F:
        for u,v in ((a,b),(b,c),(c,a)):
            x,y = (u,v) if u < v else (v,u)
            if (x,y) not in edge_map:
                edge_map[(x,y)] = len(E)
                E.append([x,y])
    E = np.asarray(E, dtype=int)
    return V, E, F

def face_edges_and_signs(V: np.ndarray, E: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each face, list its three edges and orientation sign relative to face CCW order.
    """
    # map both (u,v) and (v,u) -> (edge_index, sign)
    undirected = {(min(a,b), max(a,b)): ei for ei,(a,b) in enumerate(E)}
    signed_map = {}
    for (a,b) in undirected.keys():
        ei = undirected[(a,b)]
        signed_map[(a,b)] = (ei, +1)
        signed_map[(b,a)] = (ei, -1)

    NF = F.shape[0]
    face_edges = np.empty((NF,3), dtype=int)
    face_signs = np.empty((NF,3), dtype=int)
    for fi,(a,b,c) in enumerate(F):
        triples = [(a,b),(b,c),(c,a)]
        for k,(u,v) in enumerate(triples):
            ei, s = signed_map[(u,v)]
            face_edges[fi,k] = ei
            face_signs[fi,k] = s
    return face_edges, face_signs

def build_incidence(V: np.ndarray, E: np.ndarray, F: np.ndarray,
                    face_edges: np.ndarray, face_signs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    d0: edges<-vertices (NE x NV); d1: faces<-edges (NF x NE); both signed.
    """
    NV = V.shape[0]
    NE = E.shape[0]
    NF = F.shape[0]
    d0 = np.zeros((NE, NV), dtype=float)
    for ei,(a,b) in enumerate(E):
        d0[ei, a] = -1.0
        d0[ei, b] = +1.0
    d1 = np.zeros((NF, NE), dtype=float)
    for fi in range(NF):
        for k in range(3):
            ei = face_edges[fi, k]
            s  = face_signs[fi, k]
            d1[fi, ei] += s
    return d0, d1

def circumcenter(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Triangle circumcenter in 2D.
    """
    ax, ay = p; bx, by = q; cx, cy = r
    d = 2.0 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(d) < 1e-14:
        return (p + q + r)/3.0
    a2 = ax*ax + ay*ay
    b2 = bx*bx + by*by
    c2 = cx*cx + cy*cy
    ux = (a2*(by-cy) + b2*(cy-ay) + c2*(ay-by)) / d
    uy = (a2*(cx-bx) + b2*(ax-cx) + c2*(bx-ax)) / d
    return np.array([ux, uy], dtype=float)

def circumcentric_star1(V: np.ndarray, E: np.ndarray, F: np.ndarray,
                        face_edges: np.ndarray, face_signs: np.ndarray) -> np.ndarray:
    """
    Diagonal *1 on edges: l_dual / l_primal, where l_dual is segment length of dual edge
    between the circumcenters of the two incident faces; for boundary (shouldn't occur on torus),
    fallback to local estimate.
    """
    NE = E.shape[0]
    NV = V.shape[0]
    NF = F.shape[0]

    # circumcenters and face areas
    C = np.zeros((NF,2), dtype=float)
    for fi,(a,b,c) in enumerate(F):
        C[fi] = circumcenter(V[a], V[b], V[c])

    # build adjacency: for each edge, the two incident faces (with signs)
    # Use the fact we already have face_edges and face_signs
    inc_faces = [[] for _ in range(NE)]
    for fi in range(NF):
        for k in range(3):
            ei = face_edges[fi,k]
            inc_faces[ei].append(fi)

    star1 = np.zeros(NE, dtype=float)
    for ei,(a,b) in enumerate(E):
        p = V[a]; q = V[b]
        l_primal = np.linalg.norm(q-p)
        faces = inc_faces[ei]
        if len(faces) == 2:
            f0, f1 = faces[0], faces[1]
            l_dual = np.linalg.norm(C[f0] - C[f1])
        else:
            # (rare on torus) fallback to a local surrogate dual length
            l_dual = l_primal
        # circumcentric *1 diagonal entry
        star1[ei] = l_dual / max(l_primal, 1e-14)
    return star1

def build_dec(Lx: int, Ly: int, jitter: float, ax: float, ay: float, seed: int) -> DEC2:
    V, E, F = build_torus_jittered(Lx, Ly, jitter, ax, ay, seed)
    face_edges, face_signs = face_edges_and_signs(V, E, F)
    d0, d1 = build_incidence(V, E, F, face_edges, face_signs)
    star1 = circumcentric_star1(V, E, F, face_edges, face_signs)
    return DEC2(V=V, E=E, F=F, face_edges=face_edges, face_signs=face_signs, d0=d0, d1=d1, star1_diag=star1)

# ------------------------------
# EW from geometry
# ------------------------------

def edge_vectors(V: np.ndarray, E: np.ndarray) -> np.ndarray:
    P = V[E[:,0]]
    Q = V[E[:,1]]
    return Q - P

def cycle_1forms_templates(V: np.ndarray, E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate 1-form templates aligned with x and y: project edge vectors onto unit axes.
    These are not exact harmonic forms but serve as geometric cycle proxies for K.
    """
    ev = edge_vectors(V, E)
    omega_x = ev[:,0].copy()  # dot with e_x
    omega_y = ev[:,1].copy()  # dot with e_y
    # normalize to comparable scale
    sx = np.sqrt(np.sum(omega_x**2) + 1e-30)
    sy = np.sqrt(np.sum(omega_y**2) + 1e-30)
    omega_x /= sx
    omega_y /= sy
    return omega_x, omega_y

def ew_from_geometry(dec: DEC2, v_higgs: float) -> dict:
    """
    Using circumcentric *1 and cycle templates, build K and couplings.
    g = 2π / sqrt(K_xx), g' = 2π / sqrt(K_yy).
    """
    star1 = dec.star1_diag
    omega_x, omega_y = cycle_1forms_templates(dec.V, dec.E)
    K_xx = np.sum(star1 * omega_x * omega_x)
    K_yy = np.sum(star1 * omega_y * omega_y)
    K_xy = np.sum(star1 * omega_x * omega_y)

    # enforce symmetry small numeric noise
    K = np.array([[K_xx, K_xy],[K_xy, K_yy]], dtype=float)

    # Couplings from stiffness
    two_pi = 2.0 * math.pi
    g  = two_pi / math.sqrt(max(K[0,0], 1e-30))
    gp = two_pi / math.sqrt(max(K[1,1], 1e-30))

    # Boson masses and observables
    v = float(v_higgs)
    mW = 0.5 * g * v
    mZ = 0.5 * v * math.sqrt(g*g + gp*gp)
    sin2 = (gp*gp) / (g*g + gp*gp)
    rho  = (mW*mW) / (mZ*mZ * (1.0 - max(sin2, 1e-15)))

    out = dict(
        K_xx=K[0,0], K_yy=K[1,1], K_xy=K[0,1],
        g=g, gp=gp, v=v, mW=mW, mZ=mZ, sin2thetaW=sin2, rho=rho
    )
    return out

# ------------------------------
# Batch + autotune
# ------------------------------

def run_batch(n_meshes: int, Lx: int, Ly: int, jitter: float, v: float,
              ax: float, ay: float, outdir: str, seed: int) -> pd.DataFrame:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(n_meshes):
        dec = build_dec(Lx=Lx, Ly=Ly, jitter=jitter, ax=ax, ay=ay, seed=seed + 1000*k)
        res = ew_from_geometry(dec, v_higgs=v)
        rows.append(dict(
            ax=ax, ay=ay, seed=seed+1000*k, Lx=Lx, Ly=Ly,
            jitter=jitter,
            K_xx=res["K_xx"], K_yy=res["K_yy"], K_xy=res["K_xy"],
            g=res["g"], gp=res["gp"], v=v, mW=res["mW"], mZ=res["mZ"],
            sin2thetaW=res["sin2thetaW"], rho=res["rho"]
        ))
    df = pd.DataFrame(rows)
    df.to_csv(Path(outdir,"batch.csv"), index=False, encoding="utf-8")
    # quick text summary
    def m(x): return df[x].mean()
    def s(x): return df[x].std(ddof=1)
    lines = []
    lines.append(f"=== Circumcentric DEC EW over {n_meshes} meshes ===\n")
    lines.append(f"ax={ax:.6g}  ay={ay:.6g}   Lx={Lx} Ly={Ly}  jitter={jitter}\n")
    lines.append(f"sin^2(theta_W): mean={m('sin2thetaW'):.6f} +- {s('sin2thetaW'):.6f}\n")
    lines.append(f"rho:             mean={m('rho'):.6f} +- {s('rho'):.6f}\n")
    lines.append(f"mW (GeV):        mean={m('mW'):.3f} +- {s('mW'):.3f}\n")
    lines.append(f"mZ (GeV):        mean={m('mZ'):.3f} +- {s('mZ'):.3f}\n")
    Path(outdir,"summary.txt").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))
    return df

def grid(ax_min: float, ax_max: float, ax_steps: int,
         ay_min: float, ay_max: float, ay_steps: int) -> List[Tuple[float,float]]:
    A = np.linspace(ax_min, ax_max, ax_steps)
    B = np.linspace(ay_min, ay_max, ay_steps)
    return [(float(a), float(b)) for a in A for b in B]

def autotune(n_meshes: int, Lx: int, Ly: int, jitter: float, v: float,
             ax_min: float, ax_max: float, ax_steps: int,
             ay_min: float, ay_max: float, ay_steps: int,
             target_sin2: float, outdir: str, seed: int) -> pd.DataFrame:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    settings = grid(ax_min, ax_max, ax_steps, ay_min, ay_max, ay_steps)

    rows = []
    best = None
    best_obj = 1e99
    for (ax, ay) in settings:
        df = run_batch(n_meshes, Lx, Ly, jitter, v, ax, ay, outdir=Path(outdir, f"ax_{ax:.3f}_ay_{ay:.3f}"), seed=seed)
        mu = df["sin2thetaW"].mean()
        sig = df["sin2thetaW"].std(ddof=1)
        # objective: distance to target + mild regularization on variance
        obj = abs(mu - target_sin2) + 0.25*sig
        rows.append(dict(ax=ax, ay=ay,
                         mean_sin2=mu, std_sin2=sig,
                         mean_mW=df["mW"].mean(), std_mW=df["mW"].std(ddof=1),
                         mean_mZ=df["mZ"].mean(), std_mZ=df["mZ"].std(ddof=1),
                         mean_rho=df["rho"].mean(), std_rho=df["rho"].std(ddof=1),
                         obj=obj))
        if obj < best_obj:
            best_obj = obj
            best = (ax, ay, mu, sig)

    res = pd.DataFrame(rows).sort_values("obj", ascending=True).reset_index(drop=True)
    res.to_csv(Path(outdir, "autotune_results.csv"), index=False, encoding="utf-8")

    if best is not None:
        ax_b, ay_b, mu_b, sig_b = best
        msg = (f"[AUTOTUNE] Best (ax, ay)=({ax_b:.3f}, {ay_b:.3f})  "
               f"mean sin2={mu_b:.6f}  std={sig_b:.6f}  (target={target_sin2:.6f})\n")
        print(msg)
        Path(outdir, "best.txt").write_text(msg, encoding="utf-8")
    return res

# ------------------------------
# CLI
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Circumcentric DEC EW with (ax, ay) auto-tune.")
    p.add_argument("--mode", choices=["batch","autotune"], required=True)

    p.add_argument("--n", type=int, default=12, help="meshes per setting")
    p.add_argument("--Lx", type=int, default=12)
    p.add_argument("--Ly", type=int, default=12)
    p.add_argument("--jitter", type=float, default=0.25)
    p.add_argument("--v", type=float, default=246.0, help="Higgs vev (GeV)")
    p.add_argument("--seed", type=int, default=2025)

    # fixed-mode geometry
    p.add_argument("--ax", type=float, default=1.0, help="x scale")
    p.add_argument("--ay", type=float, default=1.0, help="y scale")

    # autotune grid
    p.add_argument("--ax_min", type=float, default=0.7)
    p.add_argument("--ax_max", type=float, default=1.3)
    p.add_argument("--ax_steps", type=int, default=5)
    p.add_argument("--ay_min", type=float, default=1.0)
    p.add_argument("--ay_max", type=float, default=2.2)
    p.add_argument("--ay_steps", type=int, default=7)
    p.add_argument("--target_sin2", type=float, default=0.231)

    p.add_argument("--out", type=str, default="out_ew_circum")
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == "batch":
        run_batch(n_meshes=args.n, Lx=args.Lx, Ly=args.Ly, jitter=args.jitter,
                  v=args.v, ax=args.ax, ay=args.ay, outdir=args.out, seed=args.seed)
    else:
        autotune(n_meshes=args.n, Lx=args.Lx, Ly=args.Ly, jitter=args.jitter, v=args.v,
                 ax_min=args.ax_min, ax_max=args.ax_max, ax_steps=args.ax_steps,
                 ay_min=args.ay_min, ay_max=args.ay_max, ay_steps=args.ay_steps,
                 target_sin2=args.target_sin2, outdir=args.out, seed=args.seed)

if __name__ == "__main__":
    main()