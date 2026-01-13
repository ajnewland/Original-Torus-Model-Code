# ckm_koide_enhanced_audit_MIN.py
import os, sys, json, math
import argparse
import numpy as np
import pandas as pd

PDG_CKM = np.array([
    [0.974, 0.225, 0.0037],
    [0.225, 0.973, 0.041 ],
    [0.0087,0.041, 0.999 ]
], dtype=float)
UP = ["u","c","t"]
DOWN = ["d","s","b"]

def rmse(A,B): return float(np.sqrt(((A-B)**2).mean()))

def sinkhorn_knopp(K, max_iter=500, tol=1e-12):
    K = np.maximum(K, 1e-300)
    r = np.ones(K.shape[0]); c = np.ones(K.shape[1])
    for _ in range(max_iter):
        r_old, c_old = r.copy(), c.copy()
        r = 1.0 / (K @ c); c = 1.0 / (K.T @ r)
        if max(np.max(np.abs(r-r_old)), np.max(np.abs(c-c_old))) < tol: break
    return np.diag(r) @ K @ np.diag(c)

class RBF2D:
    def __init__(self, ax, ay, z, lam=1e-7, eps=1e-12):
        self.ax = np.asarray(ax,float); self.ay = np.asarray(ay,float)
        self.z  = np.asarray(z,float);  self.n  = len(self.z)
        self.eps = eps
        X = np.column_stack([self.ax,self.ay])
        K = np.zeros((self.n,self.n),float)
        for i in range(self.n):
            dx = X[i,0]-X[:,0]; dy = X[i,1]-X[:,1]
            r = np.sqrt(dx*dx+dy*dy)+eps
            K[i,:] = (r**2)*np.log(r)
        K.flat[::self.n+1] += lam
        P = np.column_stack([np.ones(self.n), self.ax, self.ay])
        A = np.zeros((self.n+3,self.n+3),float)
        A[:self.n,:self.n] = K
        A[:self.n, self.n:] = P
        A[self.n:, :self.n] = P.T
        b = np.zeros(self.n+3,float); b[:self.n] = self.z
        sol = np.linalg.lstsq(A,b,rcond=None)[0]
        self.w = sol[:self.n]; self.a = sol[self.n:]; self.X = X
    def grad(self, x, y):
        dx = x - self.X[:,0]; dy = y - self.X[:,1]
        r = np.sqrt(dx*dx + dy*dy) + self.eps
        fac = (2.0*np.log(r) + 1.0)
        dphidx = dx * fac; dphidy = dy * fac
        gx = float(np.dot(self.w, dphidx) + self.a[1])
        gy = float(np.dot(self.w, dphidy) + self.a[2])
        return gx, gy

def load_locks(path):
    df = pd.read_csv(path)
    zcol = "z_pred" if "z_pred" in df.columns else ("z_target" if "z_target" in df.columns else ("z" if "z" in df.columns else None))
    if zcol is None: raise ValueError("locks file: no z_pred/z_target/z column")
    need = ["species","ax","ay",zcol,"logm"]
    for c in need:
        if c not in df.columns: raise ValueError(f"locks missing column: {c}")
    out = df[["species","ax","ay",zcol,"logm"]].copy()
    out.rename(columns={zcol:"z"}, inplace=True)
    out.dropna(subset=["ax","ay","z"], inplace=True)
    return out

def load_latent_dir(latent_dir):
    rows = []
    files = []
    for fn in os.listdir(latent_dir):
        if not fn.lower().startswith("cycle_rows_r"): continue
        if not fn.lower().endswith(".csv"): continue
        p = os.path.join(latent_dir, fn)
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        zcol = "z_pred" if "z_pred" in d.columns else ("z" if "z" in d.columns else None)
        if zcol is None: continue
        if "ax" not in d.columns or "ay" not in d.columns: continue
        use = d[["ax","ay",zcol]].copy()
        use.rename(columns={zcol:"z"}, inplace=True)
        use.dropna(subset=["ax","ay","z"], inplace=True)
        if len(use):
            rows.append(use)
            files.append(p)
    if not rows:
        raise ValueError(f"No usable cycle_rows_r*.csv in {latent_dir}")
    merged = pd.concat(rows, ignore_index=True).drop_duplicates()
    print(f"[latent] loaded {len(files)} files, {len(merged)} rows")
    for p in files[:5]: print("  e.g.", p)
    return merged

def build_features(locks_df, latent_df):
    rbf = RBF2D(latent_df["ax"].values, latent_df["ay"].values, latent_df["z"].values)
    sub = locks_df.set_index("species")
    for s in UP+DOWN:
        if s not in sub.index: raise ValueError(f"{s} missing in locks")
    S = {}
    for s in UP+DOWN:
        ax = float(sub.loc[s,"ax"]); ay = float(sub.loc[s,"ay"])
        z = float(sub.loc[s,"z"]); rho = ay/ax
        gx, gy = rbf.grad(ax, ay)
        S[s] = dict(ax=ax,ay=ay,z=z,rho=rho,gx=gx,gy=gy)
    D = np.zeros((3,3)); Dz = np.zeros((3,3)); Drho = np.zeros((3,3)); Theta = np.zeros((3,3))
    for i, ui in enumerate(UP):
        for j, dj in enumerate(DOWN):
            ax1,ay1 = S[ui]["ax"],S[ui]["ay"]; ax2,ay2 = S[dj]["ax"],S[dj]["ay"]
            D[i,j] = float(np.hypot(ax1-ax2, ay1-ay2))
            Dz[i,j] = abs(S[ui]["z"] - S[dj]["z"])
            Drho[i,j] = abs(S[ui]["rho"] - S[dj]["rho"])
            g1 = np.array([S[ui]["gx"], S[ui]["gy"]], float)
            g2 = np.array([S[dj]["gx"], S[dj]["gy"]], float)
            n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
            if n1<1e-14 or n2<1e-14: Theta[i,j] = 1.0
            else:
                cosang = float(np.clip(np.dot(g1,g2)/(n1*n2), -1.0, 1.0))
                Theta[i,j] = 1.0 - cosang
    return dict(D=D,Dz=Dz,Drho=Drho,Theta=Theta)

def fit_ckm(feats, pdg=PDG_CKM):
    D, Dz, Drho, Theta = feats["D"], feats["Dz"], feats["Drho"], feats["Theta"]
    def build_P(k1,k2,k3,k4):
        S = -k1*(D**2) - k2*Dz - k3*Drho - k4*Theta
        K = np.exp(S - S.max())
        return sinkhorn_knopp(K)

    best = None
    for k1 in np.linspace(20,1200,30):
        for k2 in np.linspace(0,300,16)[::3]:
            for k3 in np.linspace(0,300,16)[::3]:
                for k4 in np.linspace(0,300,16)[::3]:
                    P = build_P(k1,k2,k3,k4)
                    e = rmse(P, pdg)
                    if (best is None) or (e < best[0]):
                        corr = float(np.corrcoef(P.flatten(), pdg.flatten())[0,1])
                        best = (e, corr, (k1,k2,k3,k4), P)
    # polish
    e0,c0,(k1,k2,k3,k4),P0 = best
    for s1 in [0.5,0.8,1.2,1.5]:
        for s2 in [0.5,0.8,1.2,1.5]:
            for s3 in [0.5,0.8,1.2,1.5]:
                for s4 in [0.5,0.8,1.2,1.5]:
                    kk = (k1*s1,k2*s2,k3*s3,k4*s4)
                    P = build_P(*kk)
                    e = rmse(P,pdg)
                    if e < best[0]:
                        corr = float(np.corrcoef(P.flatten(), pdg.flatten())[0,1])
                        best = (e, corr, kk, P)
    return best

def koide_Q(vals):
    vals = np.asarray(vals,float)
    if np.any(vals <= 0): return np.nan
    return float((np.sum(np.sqrt(vals))**2) / (3.0*np.sum(vals)))

def best_koide_shift(z):
    z = np.asarray(z,float)
    zmin = float(z.min())
    z0s = np.linspace(zmin - 1.0, zmin - 1e-4, 4000)
    Qs = []
    for z0 in z0s: Qs.append(koide_Q(z - z0))
    Qs = np.array(Qs)
    idx = np.nanargmin(np.abs(Qs - 2.0/3.0))
    return z0s[idx], float(Qs[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locks", required=True)
    ap.add_argument("--latent-dir", required=True, help="Directory with cycle_rows_r*.csv files")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    locks = load_locks(args.locks)
    latent = load_latent_dir(args.latent_dir)

    feats = build_features(locks, latent)
    err, corr, (k1,k2,k3,k4), P = fit_ckm(feats, PDG_CKM)

    # save CKM and features
    pd.DataFrame(P, index=UP, columns=DOWN).to_csv(os.path.join(args.outdir,"ckm_fit.csv"))
    rows = []
    for i,u in enumerate(UP):
        for j,d in enumerate(DOWN):
            rows.append(dict(up=u,down=d,
                             D=feats["D"][i,j], Dz=feats["Dz"][i,j],
                             Drho=feats["Drho"][i,j], Theta=feats["Theta"][i,j],
                             P=P[i,j], PDG=PDG_CKM[i,j]))
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir,"ckm_features.csv"), index=False)

    # Koide-in-z
    zmap = locks.set_index("species")["z"]
    koide_rows = []
    for fam,name in [(["e","mu","tau"],"lepton"), (["u","c","t"],"up"), (["d","s","b"],"down")]:
        zvals = [float(zmap[s]) for s in fam]
        z0, Q = best_koide_shift(zvals)
        koide_rows.append(dict(family=name, z0=z0, Q=Q, z1=zvals[0], z2=zvals[1], z3=zvals[2]))
    pd.DataFrame(koide_rows).to_csv(os.path.join(args.outdir,"koide_z.csv"), index=False)

    with open(os.path.join(args.outdir,"summary.txt"),"w") as f:
        f.write(json.dumps({
            "ckm_rmse": err, "ckm_corr": corr,
            "k_params": {"k1":k1,"k2":k2,"k3":k3,"k4":k4}
        }, indent=2))
    print("=== Done ===")
    print("CKM RMSE:", err, " Corr:", corr)
    print("k1..k4:", k1,k2,k3,k4)
    print("Wrote:", os.path.join(args.outdir,"ckm_fit.csv"))
    print("Wrote:", os.path.join(args.outdir,"ckm_features.csv"))
    print("Wrote:", os.path.join(args.outdir,"koide_z.csv"))
    print("Wrote:", os.path.join(args.outdir,"summary.txt"))

if __name__ == "__main__":
    main()