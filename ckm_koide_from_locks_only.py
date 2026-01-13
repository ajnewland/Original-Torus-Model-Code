import os, argparse, json, numpy as np, pandas as pd

# PDG-ish magnitudes (swap with your preferred PDG set if desired)
PDG_CKM = np.array([
    [0.974, 0.225, 0.0037],
    [0.225, 0.973, 0.041 ],
    [0.0087,0.041,  0.999 ]
], dtype=float)

UP   = ["u","c","t"]
DOWN = ["d","s","b"]

def rmse(A,B):
    A = np.asarray(A, float); B = np.asarray(B, float)
    return float(np.sqrt(((A-B)**2).mean()))

def sinkhorn_knopp(K, max_iter=600, tol=1e-12):
    K = np.maximum(K, 1e-300)
    r = np.ones(K.shape[0]); c = np.ones(K.shape[1])
    for _ in range(max_iter):
        r_old, c_old = r.copy(), c.copy()
        r = 1.0/(K @ c); c = 1.0/(K.T @ r)
        if max(np.max(np.abs(r-r_old)), np.max(np.abs(c-c_old))) < tol:
            break
    return np.diag(r) @ K @ np.diag(c)

def load_locks(path):
    df = pd.read_csv(path)
    zcol = None
    for cand in ["z_pred","z_target","z"]:
        if cand in df.columns:
            zcol = cand; break
    if zcol is None:
        raise ValueError("locks file must have one of: z_pred, z_target, z")
    for req in ["species","ax","ay",zcol]:
        if req not in df.columns:
            raise ValueError(f"locks missing column: {req}")
    use = df[["species","ax","ay",zcol]].dropna()
    use = use.rename(columns={zcol:"z"})
    use["rho"] = use["ay"]/use["ax"]
    return use

def build_features_from_locks(df):
    sub = df.set_index("species")
    for s in UP+DOWN:
        if s not in sub.index: raise ValueError(f"{s} missing in locks")
    D = np.zeros((3,3)); Dz = np.zeros((3,3)); Drho = np.zeros((3,3))
    for i,u in enumerate(UP):
        for j,d in enumerate(DOWN):
            ax1,ay1,z1,rho1 = [float(sub.loc[u,k]) for k in ["ax","ay","z","rho"]]
            ax2,ay2,z2,rho2 = [float(sub.loc[d,k]) for k in ["ax","ay","z","rho"]]
            D[i,j]    = np.hypot(ax1-ax2, ay1-ay2)
            Dz[i,j]   = abs(z1 - z2)
            Drho[i,j] = abs(rho1 - rho2)
    return dict(D=D, Dz=Dz, Drho=Drho)

def fit_ckm(feats, pdg=PDG_CKM):
    D, Dz, Drho = feats["D"], feats["Dz"], feats["Drho"]
    def build_P(k1,k2,k3):
        S = -k1*(D**2) - k2*Dz - k3*Drho
        K = np.exp(S - S.max())
        return sinkhorn_knopp(K)
    best = None
    for k1 in np.linspace(20,1200,30):
        for k2 in np.linspace(0,300,16)[::3]:
            for k3 in np.linspace(0,300,16)[::3]:
                P = build_P(k1,k2,k3)
                e = rmse(P, pdg)
                if (best is None) or (e < best[0]):
                    corr = float(np.corrcoef(P.flatten(), pdg.flatten())[0,1])
                    best = (e, corr, (k1,k2,k3), P)
    # polish locally
    e0,c0,(k1,k2,k3),P0 = best
    for s1 in [0.7,0.9,1.1,1.3]:
        for s2 in [0.7,0.9,1.1,1.3]:
            for s3 in [0.7,0.9,1.1,1.3]:
                kk = (k1*s1,k2*s2,k3*s3)
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
    for z0 in z0s:
        Qs.append(koide_Q(z - z0))
    Qs = np.array(Qs)
    idx = np.nanargmin(np.abs(Qs - 2.0/3.0))
    return z0s[idx], float(Qs[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locks", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    locks = load_locks(args.locks)
    feats = build_features_from_locks(locks)
    err, corr, (k1,k2,k3), P = fit_ckm(feats, PDG_CKM)

    # outputs
    pd.DataFrame(P, index=UP, columns=DOWN).to_csv(os.path.join(args.outdir,"ckm_fit.csv"))
    rows = []
    for i,u in enumerate(UP):
        for j,d in enumerate(DOWN):
            rows.append(dict(up=u,down=d,
                D=feats["D"][i,j], Dz=feats["Dz"][i,j], Drho=feats["Drho"][i,j],
                P=P[i,j], PDG=PDG_CKM[i,j]))
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir,"ckm_features.csv"), index=False)

    # Koide in z (shifted)
    zmap = locks.set_index("species")["z"]
    koide_rows = []
    for fam,name in [(["e","mu","tau"],"lepton"), (["u","c","t"],"up"), (["d","s","b"],"down")]:
        if not set(fam).issubset(zmap.index): continue
        zvals = [float(zmap[s]) for s in fam]
        z0, Q = best_koide_shift(zvals)
        koide_rows.append(dict(family=name, z0=z0, Q=Q, z1=zvals[0], z2=zvals[1], z3=zvals[2]))
    pd.DataFrame(koide_rows).to_csv(os.path.join(args.outdir,"koide_z.csv"), index=False)

    with open(os.path.join(args.outdir,"summary.txt"),"w") as f:
        f.write(json.dumps({"ckm_rmse": err, "ckm_corr": corr,
                            "k_params": {"k1":k1,"k2":k2,"k3":k3}}, indent=2))
    print("=== Done ===")
    print("CKM RMSE:", err, " Corr:", corr)
    print("k1..k3:", k1,k2,k3)
    print("Wrote:", os.path.join(args.outdir,"ckm_fit.csv"))
    print("Wrote:", os.path.join(args.outdir,"ckm_features.csv"))
    print("Wrote:", os.path.join(args.outdir,"koide_z.csv"))
    print("Wrote:", os.path.join(args.outdir,"summary.txt"))

if __name__ == "__main__":
    main()