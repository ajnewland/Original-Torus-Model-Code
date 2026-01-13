#!/usr/bin/env python3
# Build D_W(0) from 2x2 spinor blocks to guarantee γ5-hermiticity.

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from numpy.linalg import eigvalsh, norm

# ---- 2D Euclidean gamma matrices (Hermitian) ----
sigma1 = np.array([[0, 1],[1, 0]], dtype=complex)
sigma2 = np.array([[0,-1j],[1j, 0]], dtype=complex)
gamma1, gamma2 = sigma1, sigma2
gamma5 = np.array([[1,0],[0,-1]], dtype=complex)  # = σ3

I2 = np.eye(2, dtype=complex)

def site_index(x,y,L): return y*L + x
def sp_index(site, s): return 2*site + s  # s in {0,1}

# ---- Landau links with Uy seam; optional smearing ----
def landau_links_with_seam(L, Q=1, smear=True, smear_alpha=0.20):
    theta = 2.0*np.pi*Q/(L*L)
    y = np.arange(L)[:,None]
    Ux = np.exp(1j * theta * y) * np.ones((L,L), dtype=complex)
    Uy = np.ones((L,L), dtype=complex)
    # seam on Uy[:, L-1]
    for yy in range(L):
        Uy[yy, L-1] = np.exp(-1j * 2.0*np.pi * Q * (yy / L))
    if smear and L>1:
        Uy[:, L-1] **= (1.0 - smear_alpha)
        share = smear_alpha / (L-1)
        phase_row = np.exp(-1j * 2.0*np.pi * Q * (np.arange(L)/L))
        for x in range(L-1):
            Uy[:, x] *= phase_row**share
    return Ux, Uy

def plaquette_bulk_and_seam(Ux, Uy):
    L = Ux.shape[0]
    bulk_max, seam_max = 0.0, 0.0
    for y in range(L):
        for x in range(L):
            xp, yp = (x+1)%L, (y+1)%L
            U_p = Ux[y,x] * Uy[y,xp] * np.conj(Ux[yp,x]) * np.conj(Uy[y,x])
            dev = abs(1.0 - U_p)
            touches_x_seam = (x==L-1) or (x==L-2)
            touches_y_seam = (y==L-1) or (y==L-2)
            if touches_x_seam or touches_y_seam:
                seam_max = max(seam_max, dev)
            else:
                bulk_max = max(bulk_max, dev)
    return bulk_max, seam_max

# ---- helper: add a 2x2 block into COO triplets ----
def add_block(rows, cols, data, i_site, j_site, B2):
    for s in range(2):
        for t in range(2):
            rows.append(sp_index(i_site, s))
            cols.append(sp_index(j_site, t))
            data.append(B2[s, t])

# ---- Massless Wilson–Dirac from 2x2 blocks ----
def build_DW_massless_blocks(L, Ux, Uy, r=1.0):
    """
    D_W(0) = 2r I - 1/2 Σ_μ [ (r I - γ_μ) U_μ(x) δ_{x+μ,y} + (r I + γ_μ) U_μ^*(x-μ) δ_{x-μ,y} ].
    All 2x2 matrices assembled explicitly to ensure γ5-hermiticity.
    """
    n = 2*L*L
    rows, cols, data = [], [], []

    for y in range(L):
        for x in range(L):
            site = site_index(x,y,L)
            xp, xm = (x+1)%L, (x-1)%L
            yp, ym = (y+1)%L, (y-1)%L

            # onsite: 2r I2
            add_block(rows, cols, data, site, site, 2.0*r * I2)

            # +x hop: to (xp,y) with block -(1/2)*(r I - γ1)*U_x(y,x)
            Bpx = -0.5 * (r*I2 - gamma1) * Ux[y, x]
            add_block(rows, cols, data, site, site_index(xp,y,L), Bpx)

            # -x hop: to (xm,y) with block -(1/2)*(r I + γ1)*U_x^*(y,xm)
            Bmx = -0.5 * (r*I2 + gamma1) * np.conj(Ux[y, xm])
            add_block(rows, cols, data, site, site_index(xm,y,L), Bmx)

            # +y hop: to (x,yp) with block -(1/2)*(r I - γ2)*U_y(y,x)
            Bpy = -0.5 * (r*I2 - gamma2) * Uy[y, x]
            add_block(rows, cols, data, site, site_index(x,yp,L), Bpy)

            # -y hop: to (x,ym) with block -(1/2)*(r I + γ2)*U_y^*(ym,x)
            Bmy = -0.5 * (r*I2 + gamma2) * np.conj(Uy[ym, x])
            add_block(rows, cols, data, site, site_index(x,ym,L), Bmy)

    return coo_matrix((data,(rows,cols)), shape=(n,n)).toarray()

def gamma5_hermiticity_norm(D, L):
    Gamma5_big = np.kron(np.eye(L*L, dtype=complex), gamma5)
    return norm(D.conj().T - Gamma5_big @ D @ Gamma5_big)

# ---- Spectral flow: H(m) = Γ5 [ D_W(0) - m I ] ----
def spectral_flow(L=12, Q=1, r=1.0, smear=True, m_grid=None):
    if m_grid is None:
        m_grid = np.linspace(0.0, 2.0*r, 121)

    Ux, Uy = landau_links_with_seam(L, Q, smear=smear, smear_alpha=0.20)
    bulk_dev, seam_dev = plaquette_bulk_and_seam(Ux, Uy)
    theta = 2.0*np.pi*Q/(L*L)
    print(f"L={L}, Q={Q}, theta={theta:.6f}")
    print(f"bulk max|1-U_P| ≈ {bulk_dev:.6f}  (target {2*np.sin(theta/2):.6f})")
    print(f"seam max|1-U_P| ≈ {seam_dev:.6f}")
    print("bulk admissible (eps=0.25)? ->", bulk_dev < 0.25)

    DW0 = build_DW_massless_blocks(L, Ux, Uy, r=r)
    herm = gamma5_hermiticity_norm(DW0, L)
    print(f"γ5-hermiticity: ||D† - γ5 D γ5||_F = {herm:.3e}")

    Gamma5_big = np.kron(np.eye(L*L, dtype=complex), gamma5)
    flows = []
    for m in m_grid:
        Hm = Gamma5_big @ (DW0 - m*np.eye(DW0.shape[0], dtype=complex))
        vals = eigvalsh(Hm)
        nu = 0.5 * np.sum(np.sign(vals))
        flows.append((m, vals, nu, np.min(np.abs(vals))))
    return flows

def plot_results(flows, L, Q):
    mvals = [f[0] for f in flows]
    nus   = [f[2] for f in flows]
    mins  = [f[3] for f in flows]

    plt.figure(figsize=(7.6,4.8))
    for m, vals, _, _ in flows:
        plt.plot([m]*len(vals), vals, 'b.', ms=1.5)
    plt.axhline(0, color='k', lw=1)
    plt.xlabel(r"$m$"); plt.ylabel(r"eigs($H(m)$)")
    plt.title(f"Spectral flow (overlap kernel), L={L}, Q={Q}")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(7.2,3.0))
    plt.plot(mvals, nus, 'o-', ms=2.5)
    plt.xlabel(r"$m$"); plt.ylabel(r"$\nu(m)$")
    plt.title("Index proxy vs flow mass")
    plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(7.2,3.0))
    plt.plot(mvals, mins, '-', lw=1.6)
    plt.xlabel(r"$m$"); plt.ylabel(r"$\min|\lambda(H(m))|$")
    plt.title("Smallest gap vs flow mass")
    plt.grid(True); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    # Try L=12–14, Q=1 or 2; r=1.0; smear=True for a gentler seam
    L, Q, r = 12, 1, 1.0
    flows = spectral_flow(L=L, Q=Q, r=r, smear=True,
                          m_grid=np.linspace(0.0, 2.0*r, 121))
    plot_results(flows, L, Q)