#!/usr/bin/env python3
"""
spectral_flow_admissibility.py

Spectral-flow + admissibility diagnostic for Wilson-Dirac on a 2D torus with U(1) flux.

Usage:
    python spectral_flow_admissibility.py

Configurable parameters in the CONFIG section.
"""

import os, math, numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import kron, identity, csr_matrix, csc_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

# -------------------------
# CONFIG
# -------------------------
OUTDIR = "spectral_flow_out"
os.makedirs(OUTDIR, exist_ok=True)

L = 6                     # lattice linear size (try 6,8,10)
Q = 1                     # integer flux
flux_style = "dual_plaquette"  # "dual_plaquette" or "branch_cut" or "uniform_dirac"
m0_vals = np.linspace(-2.5, 1.0, 61)  # sweep of regulator mass
nev = 8                   # number of low eigenvalues to compute per m0
use_sparse = True         # attempt sparse solver (eigsh); fallback if it fails
eigsh_tol = 1e-6
admissibility_eps = 0.1   # threshold for admissibility (max |1-U_p| < eps)
save_csv = True
# -------------------------

# -------------------------
# Gamma matrices (4x4)
# -------------------------
# Minimal 2D-like Dirac in 4x4 representation (enough for chirality/gamma5)
gamma1 = np.array([[0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, -1, 0, 0],
                   [-1, 0, 0, 0]], dtype=complex)
gamma2 = np.array([[0, 0, 0, -1j],
                   [0, 0, 1j, 0],
                   [0, 1j, 0, 0],
                   [-1j, 0, 0, 0]], dtype=complex)
gamma5 = np.diag([1, 1, -1, -1]).astype(complex)

# -------------------------
# Lattice & gauge helpers
# -------------------------
def idx_site(x, y, L):
    return (x % L) + L * (y % L)

def build_square_links(L, style="dual_plaquette", Q=1):
    """
    Return Ux, Uy arrays (shape LxL) of complex link variables implementing a flux Q.
    Styles:
      - 'dual_plaquette': distribute 2πQ evenly on the 3 edges of a triangular face
         (on square lattice we emulate by inserting flux on one elementary plaquette evenly).
      - 'branch_cut': insert a Dirac-string style flux along a short path of horizontal links.
      - 'uniform_dirac': simpler Dirac twist across a row (classic textbook).
    """
    Ux = np.ones((L, L), dtype=complex)
    Uy = np.ones((L, L), dtype=complex)

    if style == "dual_plaquette":
        # put flux 2πQ localized on plaquette at (0,0) distributed equally to its 4 edges
        # (for square lattice we split into four links)
        phi = 2.0 * math.pi * Q
        # distribute quarter-phase to each of the 4 links bounding the square at (0,0)
        Ux[0, 0] *= np.exp(1j * phi / 4.0)      # right from (0,0)
        Uy[1, 0] *= np.exp(1j * phi / 4.0)      # up from (1,0)
        Ux[0, 1] *= np.exp(1j * phi / 4.0)      # right from (0,1)
        Uy[0, 0] *= np.exp(1j * phi / 4.0)      # up from (0,0)
    elif style == "branch_cut":
        # make a short branch along x from x=0..path_len-1 at y=0
        path_len = max(1, L // 3)
        total = 2.0 * math.pi * Q
        per_link = total / path_len
        for x in range(path_len):
            Ux[x, 0] *= np.exp(1j * per_link)
    elif style == "uniform_dirac":
        # classic: put phase along last column in Uy to create net flux
        for x in range(L):
            Uy[x, L-1] *= np.exp(2j * math.pi * Q * (x / float(L)))
    else:
        raise ValueError("Unknown flux style")
    return Ux, Uy

def plaquettes_from_links(Ux, Uy):
    """
    Compute plaquette product for each elementary square plaquette (site at lower-left corner).
    Ux[x,y] is link from (x,y) -> (x+1,y)
    Uy[x,y] is link from (x,y) -> (x,y+1)
    Plaquette at (x,y): Ux[x,y] * Uy[x+1,y] * conj(Ux[x,y+1]) * conj(Uy[x,y])
    """
    L = Ux.shape[0]
    prods = np.zeros((L, L), dtype=complex)
    for x in range(L):
        for y in range(L):
            up = Ux[x, y]
            ur = Uy[(x+1)%L, y]
            dl = np.conjugate(Ux[x, (y+1)%L])
            dl2 = np.conjugate(Uy[x, y])
            prods[x, y] = up * ur * dl * dl2
    return prods

# -------------------------
# Wilson-Dirac builder (2D square lattice, 4-spinor per site)
# -------------------------
def build_wilson_dirac(L, Ux, Uy, m0, r=1.0):
    """
    Construct a sparse Wilson-Dirac operator D (4V x 4V) in CSR format.
    This implementation uses a 4-component spinor at each site (block 4x4).
    """
    V = L * L
    n = 4 * V
    # We'll construct in COO-like lists first
    rows, cols, data = [], [], []
    eye4 = np.eye(4, dtype=complex)

    for x in range(L):
        for y in range(L):
            site = idx_site(x, y, L)
            base_row = 4 * site
            # Onsite term
            for a in range(4):
                rows.append(base_row + a); cols.append(base_row + a); data.append((m0 + 2.0 * r))
            # Hopping +x
            xp, yp = (x + 1) % L, y
            neigh = idx_site(xp, yp, L); base_col = 4 * neigh
            # gamma1 coupling (use gamma1 as direction 1)
            block_f = 0.5 * ((r * eye4) - gamma1) * Ux[x, y]
            block_b = 0.5 * ((r * eye4) + gamma1) * np.conjugate(Ux[(x - 1) % L, y])
            for a in range(4):
                for b in range(4):
                    rows.append(base_row + a); cols.append(base_col + b); data.append(block_f[a, b])
                    rows.append(base_row + a); cols.append(base_col - 4 + b); data.append(0)  # placeholder no-op
            # But the above double-write is messy; instead write explicit blocks:
            # Forward +x
            for a in range(4):
                for b in range(4):
                    rows.append(base_row + a); cols.append(base_col + b); data.append(block_f[a, b])
            # Backward -x (from neighbor at x-1,y into current)
            xn, yn = (x - 1) % L, y
            neigh2 = idx_site(xn, yn, L); base_col2 = 4 * neigh2
            block_b = 0.5 * ((r * eye4) + gamma1) * np.conjugate(Ux[xn, yn])
            for a in range(4):
                for b in range(4):
                    rows.append(base_row + a); cols.append(base_col2 + b); data.append(block_b[a, b])
            # +y
            xp, yp = x, (y + 1) % L
            neigh = idx_site(xp, yp, L); base_col = 4 * neigh
            block_fy = 0.5 * ((r * eye4) - gamma2) * Uy[x, y]
            for a in range(4):
                for b in range(4):
                    rows.append(base_row + a); cols.append(base_col + b); data.append(block_fy[a, b])
            # -y
            xn, yn = x, (y - 1) % L
            neigh2 = idx_site(xn, yn, L); base_col2 = 4 * neigh2
            block_by = 0.5 * ((r * eye4) + gamma2) * np.conjugate(Uy[xn, yn])
            for a in range(4):
                for b in range(4):
                    rows.append(base_row + a); cols.append(base_col2 + b); data.append(block_by[a, b])

    # Build CSR matrix
    from scipy.sparse import coo_matrix
    D = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    # Ensure Hermitian-ish symmetry by averaging with adjoint (small numerical safety)
    D = 0.5 * (D + D.getH())
    return D

# -------------------------
# Spectral flow & crossings
# -------------------------
def compute_low_eigenvalues(H, nev=8, use_sparse=True):
    n = H.shape[0]
    if use_sparse and nev < n - 2:
        try:
            # use shift-invert? here we request smallest magnitude eigenvalues using sigma=0 approach
            vals, vecs = eigsh(csc_matrix(H), k=nev, which='SM', tol=eigsh_tol, maxiter=500)
            vals = np.sort(vals)
            return vals
        except Exception as e:
            # fallback to dense
            pass
    # dense fallback
    vals = eigh(H, eigvals_only=True)
    vals = np.sort(vals)[:nev]
    return vals

def count_zero_crossings(m0_vals, spectra):
    """
    Count sign changes across m0 for each eigen-branch and return total crossings.
    Also return per-branch crossing counts.
    """
    spectra = np.array(spectra)  # shape (nm0, nev)
    nm0, nev = spectra.shape
    branch_counts = np.zeros(nev, dtype=int)
    for k in range(nev):
        ev = spectra[:, k]
        signs = np.sign(ev)
        # treat zeros as sign 0; detect sign flips ignoring zeros
        flips = np.where(np.sign(ev[:-1]) * np.sign(ev[1:]) < 0)[0]
        branch_counts[k] = len(flips)
    return int(np.sum(branch_counts)), branch_counts

# -------------------------
# Main run
# -------------------------
def main():
    print("Spectral-flow + admissibility diagnostic")
    print(f"L={L}, Q={Q}, flux_style={flux_style}, m0 range {m0_vals[0]}..{m0_vals[-1]} ({len(m0_vals)} points)")

    # build links
    Ux, Uy = build_square_links(L, style=flux_style, Q=Q)
    pls = plaquettes_from_links(Ux, Uy)
    maxdev = np.max(np.abs(1.0 - pls))
    print("max |1 - U_P| =", maxdev)
    admissible = (maxdev < admissibility_eps)
    print("admissible (eps=%.3g)? -> %s" % (admissibility_eps, admissible))

    # Prepare Gamma5 full
    V = L * L
    Gamma5 = kron(identity(V, format='csr'), csr_matrix(gamma5))

    spectra = []
    minabs = []
    for i, m0 in enumerate(m0_vals):
        # build D_W with argument appropriate sign (we used D with +m0 onsite earlier)
        D = build_wilson_dirac(L, Ux, Uy, -m0, r=1.0)
        HW = (Gamma5 @ D).toarray()  # small L ok; for larger L use sparse workflows
        vals = np.linalg.eigvalsh(HW)
        vals = np.sort(vals)
        low = vals[:nev]
        spectra.append(low)
        minabs.append(np.min(np.abs(vals)))
        print(f"m0={m0:.4f}  minabs={minabs[-1]:.3e}  low[0..{nev-1}] ~ {np.round(low,4)}")

    spectra = np.array(spectra)
    total_crossings, branch_counts = count_zero_crossings(m0_vals, spectra)
    print("Total zero crossings (sum over branches) ≈", total_crossings)
    print("Per-branch counts:", branch_counts)

    # Save CSV
    if save_csv:
        import csv
        csvpath = os.path.join(OUTDIR, f"spectral_L{L}_Q{Q}_{flux_style}.csv")
        with open(csvpath, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["m0"] + [f"eig_{k}" for k in range(nev)] + ["minabs"]
            writer.writerow(header)
            for i, m0 in enumerate(m0_vals):
                row = [m0] + list(spectra[i].tolist()) + [minabs[i]]
                writer.writerow(row)
        print("Wrote CSV to", csvpath)

    # Plot spectral flow
    plt.figure(figsize=(8,5))
    for k in range(spectra.shape[1]):
        plt.plot(m0_vals, spectra[:, k], '-', lw=1.5, label=f'λ_{k}' if k < 3 else None)
    plt.axhline(0, color='k', lw=1)
    # shade admissible region (if admissible, shade entire x range in greenish)
    if admissible:
        plt.axvspan(m0_vals[0], m0_vals[-1], color='palegreen', alpha=0.15, label='admissible (all m0)')
    else:
        # annotate max dev
        plt.text(0.02, 0.95, f"max|1-U_p|={maxdev:.3e}", transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')

    plt.xlabel(r"$m_0$")
    plt.ylabel("Eigenvalues of $H_W$")
    plt.title(f"Spectral flow L={L}, Q={Q}, crossings≈{total_crossings}")
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=8)
    pngpath = os.path.join(OUTDIR, f"spectral_L{L}_Q{Q}_{flux_style}.png")
    plt.tight_layout()
    plt.savefig(pngpath, dpi=200)
    print("Saved spectral plot to", pngpath)
    plt.show()

if __name__ == "__main__":
    main()