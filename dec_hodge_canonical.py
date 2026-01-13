# dec_hodge_canonical.py
# DEC + Hodge on a jittered torus with canonical periods, kinetic whitening,
# and couplings from cycle lengths. FIX: include diagonal edges used by faces,
# stable 2D area, and robust CSV writing with pandas.

import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

# ---------- small utilities ----------

def triangle_area_2d(p, q, r):
    """Signed area magnitude of triangle (p,q,r) in R^2 using 2x2 determinant."""
    # 0.5 * |det([q-p, r-p])|
    u = q - p
    v = r - p
    det = u[0]*v[1] - u[1]*v[0]
    return 0.5 * abs(det)

def edge_length(p, q):
    return float(np.linalg.norm(q - p))

def sym_posdef_inv_sqrt(K):
    vals, vecs = np.linalg.eigh(K)
    vals = np.clip(vals, 1e-14, None)
    Dm12 = np.diag(1.0/np.sqrt(vals))
    return vecs @ Dm12 @ vecs.T

# ---------- DEC container ----------

@dataclass
class TorusDEC:
    V: int; E: int; F: int
    verts: np.ndarray
    edges: np.ndarray
    faces: np.ndarray
    d0: csr_matrix
    d1: csr_matrix
    star0: csr_matrix
    star1: csr_matrix
    star2: csr_matrix
    edge_len: np.ndarray
    face_area: np.ndarray
    A_tot: float
    cycles_ex: np.ndarray
    cycles_ey: np.ndarray
    cycles_sgn_x: np.ndarray
    cycles_sgn_y: np.ndarray

# ---------- build jittered torus, now with diagonals ----------

def build_torus_dec(Lx=12, Ly=12, jitter=0.25, seed=2025) -> TorusDEC:
    rng = np.random.default_rng(seed)
    xs = np.arange(Lx); ys = np.arange(Ly)
    verts = np.array([(x, y) for y in ys for x in xs], dtype=float)
    verts += (rng.random(verts.shape) - 0.5) * jitter
    V = Lx * Ly

    def vid(x, y):
        return (y % Ly) * Lx + (x % Lx)

    edges = []
    edge_map = {}  # undirected {min,max} -> eid

    def add_edge(a, b):
        a0, b0 = int(a), int(b)
        key = (a0, b0) if a0 < b0 else (b0, a0)
        if key in edge_map:
            return edge_map[key]
        eid = len(edges)
        edges.append([a0, b0])
        edge_map[key] = eid
        return eid

    # Fundamental cycles and grid edges (horiz/vert)
    cyc_ex = []; cyc_ex_sgn = []
    cyc_ey = []; cyc_ey_sgn = []

    # horizontal edges + gamma_x at y=0
    for y in range(Ly):
        for x in range(Lx):
            a = vid(x, y)
            b = vid(x+1, y)
            eid = add_edge(a, b)
            if y == 0:
                # sign for oriented path (a->b) vs stored (min->max)
                s = +1 if (edges[eid][0]==a and edges[eid][1]==b) else (-1 if (edges[eid][0]==b and edges[eid][1]==a) else +1)
                cyc_ex.append(eid); cyc_ex_sgn.append(s)

    # vertical edges + gamma_y at x=0
    for y in range(Ly):
        for x in range(Lx):
            a = vid(x, y)
            b = vid(x, y+1)
            eid = add_edge(a, b)
            if x == 0:
                s = +1 if (edges[eid][0]==a and edges[eid][1]==b) else (-1 if (edges[eid][0]==b and edges[eid][1]==a) else +1)
                cyc_ey.append(eid); cyc_ey_sgn.append(s)

    # ADD DIAGONALS used by triangular faces (both per cell; dedup via edge_map)
    for y in range(Ly):
        for x in range(Lx):
            v00 = vid(x, y)
            v10 = vid(x+1, y)
            v01 = vid(x, y+1)
            v11 = vid(x+1, y+1)
            add_edge(v11, v00)  # diag 1
            add_edge(v10, v01)  # diag 2

    edges = np.array(edges, dtype=int)
    E = edges.shape[0]

    # Faces: split each cell into two triangles (use (00,10,11) and (00,11,01))
    faces = []
    for y in range(Ly):
        for x in range(Lx):
            v00 = vid(x, y)
            v10 = vid(x+1, y)
            v01 = vid(x, y+1)
            v11 = vid(x+1, y+1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces = np.array(faces, dtype=int)
    F = faces.shape[0]

    # d0: V->E (stored orientation is min->max; incidence uses -1 at tail, +1 at head)
    rows, cols, data = [], [], []
    for eid, (a, b) in enumerate(edges):
        rows += [eid, eid]; cols += [a, b]; data += [-1.0, +1.0]
    d0 = coo_matrix((data, (rows, cols)), shape=(E, V)).tocsr()

    # undirected orientation map
    undirected = {}
    for eid, (a, b) in enumerate(edges):
        if a < b:
            undirected[(a, b)] = (eid, +1)
            undirected[(b, a)] = (eid, -1)
        else:
            undirected[(b, a)] = (eid, +1)
            undirected[(a, b)] = (eid, -1)

    # d1: E->F via oriented boundary (a->b->c->a)
    rows, cols, data = [], [], []
    for fid, (a, b, c) in enumerate(faces):
        eab = undirected[(a, b)]
        ebc = undirected[(b, c)]
        eca = undirected[(c, a)]
        for (eid, s) in (eab, ebc, eca):
            rows.append(fid); cols.append(eid); data.append(s)
    d1 = coo_matrix((data, (rows, cols)), shape=(F, E)).tocsr()

    # geometric data
    pts = verts
    edge_len = np.array([edge_length(pts[a], pts[b]) for (a, b) in edges], dtype=float)
    face_area = np.array([triangle_area_2d(pts[a], pts[b], pts[c]) for (a, b, c) in faces], dtype=float)
    A_tot = float(face_area.sum())

    # Hodge stars (barycentric for *0, edge lengths for *1, 1/area for *2)
    v_area = np.zeros(V)
    for (a, b, c), A in zip(faces, face_area):
        v_area[a] += A/3.0; v_area[b] += A/3.0; v_area[c] += A/3.0
    star0 = diags(v_area + 1e-15)
    star1 = diags(edge_len + 1e-15)
    star2 = diags(1.0 / (face_area + 1e-15))

    return TorusDEC(
        V=V, E=E, F=F,
        verts=verts, edges=edges, faces=faces,
        d0=d0, d1=d1,
        star0=star0, star1=star1, star2=star2,
        edge_len=edge_len, face_area=face_area,
        A_tot=A_tot,
        cycles_ex=np.array(cyc_ex, dtype=int),
        cycles_ey=np.array(cyc_ey, dtype=int),
        cycles_sgn_x=np.array(cyc_ex_sgn, dtype=int),
        cycles_sgn_y=np.array(cyc_ey_sgn, dtype=int),
    )

# ---------- Laplacians, harmonics, cycles ----------

def scalar_laplacian(d0, star0, star1):
    star0_inv = diags(1.0 / (star0.diagonal()))
    return star0_inv @ (d0.T @ (star1 @ d0))

def oneform_laplacian(d0, d1, star0, star1, star2):
    star1_inv = diags(1.0 / (star1.diagonal()))
    return star1_inv @ (d0 @ (star0 @ d0.T) + d1.T @ (star2 @ d1))

def harmonic_basis(L1, k=2):
    evals, evecs = eigsh(L1, k=k, sigma=0.0, which="LM")
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx]

def line_integral_on_cycle(omega, cycle_edges, cycle_sgn):
    return float(np.sum(omega[cycle_edges] * cycle_sgn))

def cycle_length(edge_len, cycle_edges):
    return float(np.sum(edge_len[cycle_edges]))

# ---------- periods, whitening, masses ----------

def canonicalize_periods(W, dec: TorusDEC):
    M = np.zeros((2,2), dtype=float)
    M[0,0] = line_integral_on_cycle(W[:,0], dec.cycles_ex, dec.cycles_sgn_x)
    M[0,1] = line_integral_on_cycle(W[:,1], dec.cycles_ex, dec.cycles_sgn_x)
    M[1,0] = line_integral_on_cycle(W[:,0], dec.cycles_ey, dec.cycles_sgn_y)
    M[1,1] = line_integral_on_cycle(W[:,1], dec.cycles_ey, dec.cycles_sgn_y)
    Minv = inv(M)
    return W @ Minv, M

def kinetic_matrix(W, star1):
    K = np.zeros((2,2), dtype=float)
    for i in range(2):
        for j in range(2):
            K[i,j] = float(W[:,i].T @ (star1 @ W[:,j]))
    return K

def whiten(Wcanon, star1):
    K = kinetic_matrix(Wcanon, star1)
    C = sym_posdef_inv_sqrt(K)
    return Wcanon @ C, K, C

def sm_masses_from_v(v, g, gp):
    mW = 0.5 * g * v
    mZ = 0.5 * np.sqrt(g**2 + gp**2) * v
    s2 = gp**2 / (g**2 + gp**2 + 1e-30)
    rho = (mW**2) / (mZ**2 * (1 - s2 + 1e-30))
    return mW, mZ, s2, rho

# ---------- main run ----------

def run_single(Lx=12, Ly=12, jitter=0.25, seed=2025,
               v_fixed=246.0, use_derived_v=False, C_const=3.9166e5,
               outdir="out_hodge_canonical"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    dec = build_torus_dec(Lx=Lx, Ly=Ly, jitter=jitter, seed=seed)

    # spectral scales
    L0 = scalar_laplacian(dec.d0, dec.star0, dec.star1)
    L1 = oneform_laplacian(dec.d0, dec.d1, dec.star0, dec.star1, dec.star2)

    vals0, _ = eigsh(L0, k=3, sigma=0.0, which="LM")
    vals0 = np.sort(vals0)
    lam1 = float(vals0[1]) if vals0[0] < 1e-10 else float(vals0[0])
    lam1_hat = lam1 * dec.A_tot

    evals1, evecs1 = harmonic_basis(L1, k=2)  # E x 2
    W = evecs1

    # canonicalize
    Wcan, M = canonicalize_periods(W, dec)

    # cycle-length couplings
    ell_x = cycle_length(dec.edge_len, dec.cycles_ex)
    ell_y = cycle_length(dec.edge_len, dec.cycles_ey)
    g_len  = 2*np.pi / max(ell_x, 1e-12)
    gp_len = 2*np.pi / max(ell_y, 1e-12)
    mW_len, mZ_len, s2_len, rho_len = sm_masses_from_v(v_fixed, g_len, gp_len)

    # whitening
    Wwhite, K, C = whiten(Wcan, dec.star1)
    # re-canon after whitening
    Wwhite_can, M2 = canonicalize_periods(Wwhite, dec)
    mW_len_w, mZ_len_w, s2_len_w, rho_len_w = sm_masses_from_v(v_fixed, g_len, gp_len)

    # optional derived v
    v_used = v_fixed
    if use_derived_v:
        v_used = C_const / np.sqrt(max(lam1_hat, 1e-16))
    mW_der, mZ_der, s2_der, rho_der = sm_masses_from_v(v_used, g_len, gp_len)

    # prints
    print("=== DEC bosons (Hodge canonical) ===")
    print(f"Lx={Lx} Ly={Ly}  V={dec.V} E={dec.E} F={dec.F}")
    print(f"A_tot={dec.A_tot:.6f}  lambda1={lam1:.6e}  lambda1_hat={lam1_hat:.6f}")

    print("\n[Periods] raw M:")
    print(np.array2string(M, formatter={'float_kind':lambda x:f"{x: .6f}"}))
    P_can = np.array([[line_integral_on_cycle(Wcan[:,0], dec.cycles_ex, dec.cycles_sgn_x),
                       line_integral_on_cycle(Wcan[:,1], dec.cycles_ex, dec.cycles_sgn_x)],
                      [line_integral_on_cycle(Wcan[:,0], dec.cycles_ey, dec.cycles_sgn_y),
                       line_integral_on_cycle(Wcan[:,1], dec.cycles_ey, dec.cycles_sgn_y)]])
    print("Canonical periods (≈ I):")
    print(np.array2string(P_can, formatter={'float_kind':lambda x:f"{x: .6f}"}))
    print(f"\nCycle lengths: ell_x={ell_x:.6f}, ell_y={ell_y:.6f}")

    print("\n[Couplings from lengths]")
    print(f"g_len={g_len:.6f}, g'_len={gp_len:.6f}")
    print("[Masses (fixed v)]")
    print(f"mW={mW_len:.6f}, mZ={mZ_len:.6f}, sin^2θW={s2_len:.6f}, rho={rho_len:.6f}")

    print("\n[Kinetic whitening]")
    print("K (pre):")
    print(np.array2string(K, formatter={'float_kind':lambda x:f"{x: .6f}"}))
    Kwhite = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            Kwhite[i,j] = float(Wwhite[:,i].T @ (dec.star1 @ Wwhite[:,j]))
    print("K (post, ≈ I):")
    print(np.array2string(Kwhite, formatter={'float_kind':lambda x:f"{x: .6f}"}))
    P_white = np.array([[line_integral_on_cycle(Wwhite_can[:,0], dec.cycles_ex, dec.cycles_sgn_x),
                         line_integral_on_cycle(Wwhite_can[:,1], dec.cycles_ex, dec.cycles_sgn_x)],
                        [line_integral_on_cycle(Wwhite_can[:,0], dec.cycles_ey, dec.cycles_sgn_y),
                         line_integral_on_cycle(Wwhite_can[:,1], dec.cycles_ey, dec.cycles_sgn_y)]])
    print("Periods after whitening (≈ I):")
    print(np.array2string(P_white, formatter={'float_kind':lambda x:f"{x: .6f}"}))
    print("[Masses after whitening (same g,g') ]")
    print(f"mW={mW_len_w:.6f}, mZ={mZ_len_w:.6f}, sin^2θW={s2_len_w:.6f}, rho={rho_len_w:.6f}")

    if use_derived_v:
        print("\n[Derived v from spectral scale]")
        print(f"v_derived={v_used:.6f}")
        print(f"mW={mW_der:.6f}, mZ={mZ_der:.6f}, sin^2θW={s2_der:.6f}, rho={rho_der:.6f}")

    # ---- robust CSV (pandas handles string/numeric mix) ----
    mode = "derived_v" if use_derived_v else "fixed_v"
    df = pd.DataFrame([{
        "Lx": Lx, "Ly": Ly, "V": dec.V, "E": dec.E, "F": dec.F,
        "A_tot": dec.A_tot, "lambda1": lam1, "lambda1_hat": lam1_hat,
        "ell_x": ell_x, "ell_y": ell_y, "g_len": g_len, "gp_len": gp_len,
        "mW_len": mW_len, "mZ_len": mZ_len, "s2_len": s2_len, "rho_len": rho_len,
        "mW_len_w": mW_len_w, "mZ_len_w": mZ_len_w, "s2_len_w": s2_len_w, "rho_len_w": rho_len_w,
        "v_used": v_used, "mode": mode
    }])
    out_csv = Path(outdir, "summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")

if __name__ == "__main__":
    run_single(Lx=12, Ly=12, jitter=0.25, seed=2025,
               v_fixed=246.0, use_derived_v=False,
               C_const=3.9166e5, outdir="out_hodge_canonical")
