# ckm_koide_enhanced_audit.py
# ------------------------------------------------------------
# Usage (example):
#   python ckm_koide_enhanced_audit.py \
#     --locks "C:/path/to/all_particles_locked.csv" \
#     --latent-globs "C:/path/to/New Results/**/*cycle_rows_r.csv" \
#     --outdir "C:/path/to/grand_audit_ckm_koide"
#
# Notes:
# - Accepts multiple --latent-globs arguments.
# - Expects locks CSV to have columns: species, ax, ay, z_pred (or z), logm
# - Latent files: tries columns among {ax, ay, z_pred, z}; ignores rows w/ missing ax,ay
# - Writes: ckm_fit.csv, ckm_features.csv, koide_z.csv, summary.txt
# ------------------------------------------------------------

import argparse, os, glob, math, json
import numpy as np
import pandas as pd

# -----------------------
# Utilities
# -----------------------
PDG_CKM = np.array([
    [0.974, 0.225, 0.0037],
    [0.225, 0.973, 0.041 ],
    [0.0087,0.041, 0.999 ]
], dtype=float)
UP = ["u","c","t"]
DOWN = ["d","s","b"]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def rmse(A,B):
    return float(np.sqrt(((A-B)**2).mean()))

def sinkhorn_knopp(K, max_iter=500, tol=1e-12):
    K = np.maximum(K, 1e-300)  # avoid zeros
    r = np.ones(K.shape[0])
    c = np.ones(K.shape[1])
    for _ in range(max_iter):
        r_old, c_old = r.copy(), c.copy()
        r = 1.0 / (K @ c)
        c = 1.0 / (K.T @ r)
        if max(np.max(np.abs(r-r_old)), np.max(np.abs(c-c_old))) < tol:
            break
    return np.diag(r) @ K @ np.diag(c)

# -----------------------
# RBF interpolant for z(ax,ay)
# -----------------------
class RBF2D:
    """
    Simple thin-plate-spline RBF: phi(r) = r^2 * log(r + eps)
    Provides z(ax,ay) and gradient via analytic derivatives.
    """
    def __init__(self, ax, ay, z, lam=1e-8, eps=1e-12):
        self.ax = np.asarray(ax, float)
        self.ay = np.asarray(ay, float)
        self.z  = np.asarray(z,  float)
        self.n  = len(self.z)
        self.lam = lam
        self.eps = eps
        # Fit coefficients for: f(x) = sum_i w_i phi(||x-x_i||) + b0 + b1*x + b2*y
        X = np.column_stack([self.ax, self.ay])
        # Build K
        K = np.zeros((self.n, self.n), float)
        for i in range(self.n):
            dx = X[i,0] - X[:,0]
            dy = X[i,1] - X[:,1]
            r = np.sqrt(dx*dx + dy*dy) + eps
            K[i,:] = (r**2) * np.log(r)
        # Regularize diagonal
        K.flat[::self.n+1] += lam
        # Poly terms
        P = np.column_stack([np.ones(self.n), self.ax, self.ay])
        # Solve augmented system
        # [ K  P ] [w] = [z]
        # [ P' 0 ] [a]   [0]
        A = np.zeros((self.n+3, self.n+3), float)
        A[:self.n,:self.n] = K
        A[:self.n, self.n:] = P
        A[self.n:, :self.n] = P.T
        b = np.zeros(self.n+3, float)
        b[:self.n] = self.z
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        self.w = sol[:self.n]
        self.a = sol[self.n:]  # [b0, b1, b2]
        self.X = X

    def eval(self, x, y):
        dx = x - self.X[:,0]
        dy = y - self.X[:,1]
        r = np.sqrt(dx*dx + dy*dy) + self.eps
        phi = (r**2) * np.log(r)
        return float(np.dot(self.w, phi) + self.a[0] + self.a[1]*x + self.a[2]*y)

    def grad(self, x, y):
        dx = x - self.X[:,0]
        dy = y - self.X[:,1]
        r = np.sqrt(dx*dx + dy*dy) + self.eps
        # d/dr [ r^2 log r ] = 2r log r + r
        dphidr = 2.0*r*np.log(r) + r
        # grad phi = dphidr * grad r = dphidr * (dx/r, dy/r) = (dx*(2 log r + 1), dy*(2 log r + 1))
        fac = (2.0*np.log(r) + 1.0)
        dphidx = dx * fac
        dphidy = dy * fac
        gx = float(np.dot(self.w, dphidx) + self.a[1])
        gy = float(np.dot(self.w, dphidy) + self.a[2])
        return gx, gy

# -----------------------
# Loaders
# -----------------------
def load_locks(path):
    df = pd.read_csv(path)
    # Pick z column preference
    zcol = "z_pred" if "z_pred" in df.columns else ("z_target" if "z_target" in df.columns else ("z" if "z" in df.columns else None))
    if zcol is None:
        raise ValueError("No z columns in locks file. Expected one of: z_pred, z_target, z")
    need = ["species","ax","ay",zcol]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in locks file")
    out = df[["species","ax","ay",zcol,"logm"]].copy()
    out.rename(columns={zcol:"z"}, inplace=True)
    out.dropna(subset=["ax","ay","z"], inplace=True)
    return out

def load_latent_from_globs(globs):
    rows = []
    for g in globs:
        for p in glob.glob(g, recursive=True):
            try:
                d = pd.read_csv(p)
            except Exception:
                continue
            # choose z column
            zcol = "z_pred" if "z_pred" in d.columns else ("z" if "z" in d.columns else None)
            if zcol is None:
                continue
            # Ensure ax, ay
            if "ax" not in d.columns or "ay" not in d.columns:
                # some files have empty ax; skip those rows
                continue
            dd = d[["ax","ay",zcol]].copy()
            dd.dropna(subset=["ax","ay",zcol], inplace=True)
            dd.rename(columns={zcol:"z"}, inplace=True)
            if len(dd):
                rows.append(dd)
    if not rows:
        raise ValueError("No latent points found from the provided globs.")
    latent = pd.concat(rows, ignore_index=True)
    # Drop obvious duplicates
    latent = latent.drop_duplicates()
    return latent

# -----------------------
# Feature builder
# -----------------------
def build_ckm_features(locks_df, latent_df):
    # RBF fit over latent points
    rbf = RBF2D(latent_df["ax"].values, latent_df["ay"].values, latent_df["z"].values, lam=1e-7)
    # Extract species we need
    sub = locks_df.set_index("species")
    needed = UP + DOWN
    for s in needed:
        if s not in sub.index:
            raise ValueError(f"species {s} missing in locks file")
    # Pack coords, z, rho, grad
    S = {}
    for s in needed:
        ax = float(sub.loc[s,"ax"])
        ay = float(sub.loc[s,"ay"])
        z  = float(sub.loc[s,"z"])
        rho = ay/ax
        gx, gy = rbf.grad(ax, ay)
        S[s] = dict(ax=ax, ay=ay, z=z, rho=rho, gx=gx, gy=gy)

    # Build matrices
    D = np.zeros((3,3)); Dz = np.zeros((3,3)); Drho = np.zeros((3,3)); Theta = np.zeros((3,3))
    for i, ui in enumerate(UP):
        for j, dj in enumerate(DOWN):
            ax1, ay1 = S[ui]["ax"], S[ui]["ay"]
            ax2, ay2 = S[dj]["ax"], S[dj]["ay"]
            D[i,j] = float(np.hypot(ax1-ax2, ay1-ay2))
            Dz[i,j] = abs(S[ui]["z"] - S[dj]["z"])
            Drho[i,j] = abs(S[ui]["rho"] - S[dj]["rho"])
            g1 = np.array([S[ui]["gx"], S[ui]["gy"]], float)
            g2 = np.array([S[dj]["gx"], S[dj]["gy"]], float)
            n1 = np.linalg.norm(g1); n2 = np.linalg.norm(g2)
            if n1<1e-14 or n2<1e-14:
                Theta[i,j] = 1.0
            else:
                cosang = float(np.clip(np.dot(g1,g2)/(n1*n2), -1.0, 1.0))
                Theta[i,j] = 1.0 - cosang
    feats = dict(D=D, Dz=Dz, Drho=Drho, Theta=Theta, species_state=S)
    return feats, rbf

def fit_ckm(feats, pdg=PDG_CKM, coarse=True):
    D, Dz, Drho, Theta = feats["D"], feats["Dz"], feats["Drho"], feats["Theta"]

    def build_P(k1,k2,k3,k4):
        S = -k1*(D**2) - k2*Dz - k3*Drho - k4*Theta
        K = np.exp(S - S.max())
        return sinkhorn_knopp(K)

    best = None
    if coarse:
        k1_grid = np.linspace(20, 1200, 30)
        k2_grid = np.linspace(0, 300, 16)
        k3_grid = np.linspace(0, 300, 16)
        k4_grid = np.linspace(0, 300, 16)
        for k1 in k1_grid:
            for k2 in k2_grid[::3]:
                for k3 in k3_grid[::3]:
                    for k4 in k4_grid[::3]:
                        P = build_P(k1,k2,k3,k4)
                        e = rmse(P, pdg)
                        if (best is None) or (e < best[0]):
                            corr = float(np.corrcoef(P.flatten(), pdg.flatten())[0,1])
                            best = (e, corr, (k1,k2,k3,k4), P)
    # small local polish around coarse
    if best is not None:
        e0, c0, (k1,k2,k3,k4), P0 = best
        for s1 in [0.5, 0.8, 1.2, 1.5]:
            for s2 in [0.5, 0.8, 1.2, 1.5]:
                for s3 in [0.5, 0.8, 1.2, 1.5]:
                    for s4 in [0.5, 0.8, 1.2, 1.5]:
                        kk = (k1*s1, k2*s2, k3*s3, k4*s4)
                        P = build_P(*kk)
                        e = rmse(P, pdg)
                        if e < best[0]:
                            corr = float(np.corrcoef(P.flatten(), pdg.flatten())[0,1])
                            best = (e, corr, kk, P)
    return best  # (rmse, corr, (k1,k2,k3,k4), P)

# -----------------------
# Koide in z (with family shift)
# -----------------------
def koide_Q(vals):
    vals = np.asarray(vals, float)
    if np.any(vals <= 0): return np.nan
    return float((np.sum(np.sqrt(vals))**2) / (3.0*np.sum(vals)))

def best_koide_shift(ztriplet):
    z = np.asarray(ztriplet, float)
    zmin = float(z.min())
    # scan z0 below zmin so (z - z0) > 0
    z0s = np.linspace(zmin - 1.0, zmin - 1e-4, 4000)
    Qs  = []
    for z0 in z0s:
        Qs.append(koide_Q(z - z0))
    Qs = np.array(Qs)
    idx = np.nanargmin(np.abs(Qs - 2.0/3.0))
    return z0s[idx], float(Qs[idx])

def do_koide(locks_df, outdir):
    sub = locks_df.set_index("species")["z"]
    fams = {
        "lepton": ["e","mu","tau"],
        "up": ["u","c","t"],
        "down": ["d","s","b"]
    }
    rows = []
    for name, trip in fams.items():
        zvals = [float(sub[s]) for s in trip]
        z0, Q = best_koide_shift(zvals)
        rows.append(dict(family=name, z0=z0, Q=Q, z1=zvals[0], z2=zvals[1], z3=zvals[2]))
    koide_df = pd.DataFrame(rows)
    koide_df.to_csv(os.path.join(outdir, "koide_z.csv"), index=False)
    return koide_df

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locks", required=True, help="Path to all_particles_locked.csv")
    ap.add_argument("--latent-globs", nargs="+", required=True, help="One or more glob patterns for latent CSVs")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    locks = load_locks(args.locks)
    latent = load_latent_from_globs(args.latent_globs)

    # Build CKM features (D, Dz, Drho, Theta) and fit
    feats, rbf = build_ckm_features(locks, latent)
    best = fit_ckm(feats, PDG_CKM, coarse=True)
    if best is None:
        raise RuntimeError("CKM fit failed to produce any candidate.")
    err, corr, (k1,k2,k3,k4), P = best

    # Save CKM results
    ckm_df = pd.DataFrame(P, index=UP, columns=DOWN)
    ckm_df.to_csv(os.path.join(outdir, "ckm_fit.csv"))
    # Save features (flattened)
    feat_rows = []
    for i,u in enumerate(UP):
        for j,d in enumerate(DOWN):
            feat_rows.append(dict(
                up=u, down=d,
                D=feats["D"][i,j],
                Dz=feats["Dz"][i,j],
                Drho=feats["Drho"][i,j],
                Theta=feats["Theta"][i,j],
                P=P[i,j],
                PDG=PDG_CKM[i,j]
            ))
    pd.DataFrame(feat_rows).to_csv(os.path.join(outdir, "ckm_features.csv"), index=False)

    # Koide in z (with family shifts)
    koide_df = do_koide(locks, outdir)

    # Summary
    summary = {
        "ckm_rmse": err,
        "ckm_corr": corr,
        "k_params": {"k1":k1, "k2":k2, "k3":k3, "k4":k4},
        "notes": "Theta uses gradient misalignment from RBF-interpolated z(ax,ay). Sinkhorn enforces doubly-stochastic rows/cols."
    }
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(json.dumps(summary, indent=2))

    print("=== CKM enhanced fit ===")
    print("RMSE:", err, " Corr:", corr)
    print("k1,k2,k3,k4:", k1,k2,k3,k4)
    print("Saved:", os.path.join(outdir, "ckm_fit.csv"))
    print("Saved:", os.path.join(outdir, "ckm_features.csv"))
    print("=== Koide-in-z (with shift) ===")
    print(koide_df.to_string(index=False))
    print("Summary:", os.path.join(outdir, "summary.txt"))

if __name__ == "__main__":
    main()