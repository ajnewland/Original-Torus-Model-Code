#!/usr/bin/env python3
# overlap_flow_diagnostics.py
# Adds γ5-hermiticity check + robust mass scan + flow-sign toggle.

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from numpy.linalg import eigvalsh, norm

# --- 2D Euclidean gamma matrices (2-component) ---
sigma1 = np.array([[0, 1],[1, 0]], dtype=complex)
sigma2 = np.array([[0,-1j],[1j, 0]], dtype=complex)
gamma1, gamma2 = sigma1, sigma2
gamma5 = np.array([[1,0],[0,-1]], dtype=complex)

def site_index(x,y,L): return y*L + x
def sp_index(x,y,s,L): return 2*site_index(x,y,L) + s  # s∈{0,1}

# --- Landau links with Uy seam; optional smearing ---
def landau_links_with_seam(L, Q=1, smear=True, smear_alpha=0.20):
    theta = 2.0*np.pi*Q/(L*L)
    y = np.arange(L)[:,None]
    Ux = np.exp(1j * theta * y) * np.ones((L,L), dtype=complex)
    Uy = np.ones((L,L), dtype=complex)
    for yy in range(L):
        Uy[yy, L-1] = np.exp(-1j * 2.0*np.pi * Q * (yy / L))
    if smear and L>1:
        Uy[:, L-1] **= (1.0 - smear_alpha)
        share = smear_alpha / (L-1)
        phase_row = np.exp(-1j * 2.0*np.pi * Q * (np.arange(L)/L))
        for x in range(L-1):
            Uy[:, x] *= phase_row**share
    return Ux, Uy, theta

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

# --- Massless Wilson–Dirac ---
def build_DW_massless(L, Ux, Uy, r=1.0):
    n = 2*L*L
    rows, cols, data = [], [], []
    for y in range(L):
        for x in range(L):
            xp, xm = (x+1)%L, (x-1)%L
            yp, ym = (y+1)%L, (y-1)%L
            for s in range(2):
                i = sp_index(x,y,s,L)
                rows.append(i); cols.append(i); data.append(2.0*r)
                # +x
                U = Ux[y,x]
                for sp in range(2):
                    rows.append(i); cols.append(sp_index(xp,y,sp,L))
                    data.append(-0.5 * (r - gamma1[s,sp]) * U)
                # -x
                U = np.conj(Ux[y,xm])
                for sp in range(2):
                    rows.append(i); cols.append(sp_index(xm,y,sp,L))
                    data.append(-0.5 * (r + gamma1[s,sp]) * U)
                # +y
                U = Uy[y,x]
                for sp in range(2):
                    rows.append(i); cols.append(sp_index(x,yp,sp,L))
                    data.append(-0.5 * (r - gamma2[s,sp]) * U)
                # -y
                U = np.conj(Uy[ym,x])
                for sp in range(2):
                    rows.append(i); cols.append(sp_index(x,ym,sp,L))
                    data.append(-0.5 * (r + gamma2[s,sp]) * U)
    return coo_matrix((data,(rows,cols)), shape=(n,n)).toarray()

def check_gamma5_hermiticity(D):
    # Check || D† - γ5 D γ5 ||_F
    Lsq = D.shape[0]//2
    Gamma5_big = np.kron(np.eye(Lsq, dtype=complex), gamma5)
    lhs = D.conj().T
    rhs = Gamma5_big @ D @ Gamma5_big
    return norm(lhs - rhs)

# --- Spectral flow H(m) = γ5 ( ± D_W(0) ∓ m I ) ---
def spectral_flow(L=12, Q=1, r=1.0, m_grid=None, smear=True, eps_bulk=0.25, FLOW_SIGN=-1):
    """
    FLOW_SIGN = +1  -> H(m) = γ5 ( D_W(0) - m I )
               = -1 -> H(m) = γ5 ( m I - D_W(0) )
    """
    if m_grid is None:
        m_grid = np.linspace(-2.0, 4.0, 301)  # wide sweep to not miss crossings

    Ux, Uy, theta = landau_links_with_seam(L, Q, smear=smear, smear_alpha=0.20)
    bulk_dev, seam_dev = plaquette_bulk_and_seam(Ux, Uy)
    print(f"L={L}, Q={Q}, theta={theta:.6f}")
    print(f"bulk max|1-U_P| ≈ {bulk_dev:.6f}  (target 2 sin(theta/2) = {2*np.sin(theta/2):.6f})")
    print(f"seam max|1-U_P| ≈ {seam_dev:.6f}  (seam >> bulk expected)")
    print(f"bulk admissible (eps={eps_bulk})? -> {bulk_dev < eps_bulk}")

    DW0 = build_DW_massless(L, Ux, Uy, r=r)
    g5_norm = check_gamma5_hermiticity(DW0)
    print(f"γ5-hermiticity check: ||D† - γ5 D γ5||_F = {g5_norm:.3e}")

    Gamma5_big = np.kron(np.eye(L*L, dtype=complex), gamma5)

    flows = []
    minabs_list = []
    for m in m_grid:
        if FLOW_SIGN > 0:
            Hm = Gamma5_big @ (DW0 - m*np.eye(DW0.shape[0], dtype=complex))
        else:
            Hm = Gamma5_big @ (m*np.eye(DW0.shape[0], dtype=complex) - DW0)
        vals = eigvalsh(Hm)
        nu = 0.5 * np.sum(np.sign(vals))
        minabs = np.min(np.abs(vals))
        flows.append((m, vals, nu, minabs))
        minabs_list.append(minabs)
    return (bulk_dev, seam_dev, theta, g5_norm), flows

def plot_results(flows, L, Q, title_tag=""):
    mvals = [f[0] for f in flows]
    nus   = [f[2] for f in flows]
    mins  = [f[3] for f in flows]

    # spectral flow
    plt.figure(figsize=(7.4,4.8))
    for m, vals, _, _ in flows:
        plt.plot([m]*len(vals), vals, 'b.', ms=1.5)
    plt.axhline(0, color='k', lw=1)
    plt.xlabel(r"$m$"); plt.ylabel(r"eigs($H(m)$)")
    plt.title(f"Spectral flow (overlap kernel), L={L}, Q={Q} {title_tag}")
    plt.tight_layout(); plt.show()

    # index vs m
    plt.figure(figsize=(7.2,3.0))
    plt.plot(mvals, nus, 'o-', ms=2.5)
    plt.xlabel(r"$m$"); plt.ylabel(r"$\nu(m)$")
    plt.title("Index proxy vs flow mass")
    plt.grid(True); plt.tight_layout(); plt.show()

    # min |lambda| vs m (shows pinch locations)
    plt.figure(figsize=(7.2,3.0))
    plt.plot(mvals, mins, '-', lw=1.5)
    plt.xlabel(r"$m$"); plt.ylabel(r"$\min|\lambda(H(m))|$")
    plt.title("Smallest gap vs flow mass")
    plt.grid(True); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    # --- knobs ---
    L, Q, r      = 12, 1, 1.0
    smear        = True
    eps_bulk     = 0.25
    FLOW_SIGN    = +1     # try +1 first; if no plateau, try -1
    m_grid       = np.linspace(-2.0, 4.0, 301)

    (bulk_dev, seam_dev, theta, g5_norm), flows = spectral_flow(
        L=L, Q=Q, r=r, m_grid=m_grid, smear=smear, eps_bulk=eps_bulk, FLOW_SIGN=FLOW_SIGN
    )
    plot_results(flows, L, Q, title_tag=f"(FLOW_SIGN={FLOW_SIGN})")