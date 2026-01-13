# anchor_free_isotonic_auto.py
# Anchor-free, geometry-only isotonic mapping of z -> mass for ~6-10 species.
# Fit target is monotone *rank/quantile*, not PDG. No external anchors.

import argparse, math
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

# Optional PDG dict for diagnostics ONLY (not used to fit)
PDG = {
    "e":0.000511, "mu":0.10566, "tau":1.77686,
    "u":0.0022, "d":0.0047, "s":0.095, "c":1.27, "b":4.18, "t":172.76
}

def load_z(path):
    df = pd.read_csv(path)
    zcol = 'z' if 'z' in df.columns else [c for c in df.columns if c.lower() in ('z','latent_z','z_norm','r_norm')][0]
    if 'species' not in df.columns:
        df['species'] = [f"sp{i}" for i in range(len(df))]
    df = df[['species', zcol]].rename(columns={zcol:'z'}).dropna().drop_duplicates()
    return df.sort_values('z').reset_index(drop=True)

def extend_to_n(df, N):
    if len(df) >= N:
        return df.iloc[:N].copy()
    # create (N - len(df)) virtual species by linear interpolation along z
    z = df['z'].values
    if len(z) == 1:
        # only one point; synthesize a small ladder around it
        zs = np.linspace(df['z'].iloc[0]-0.5, df['z'].iloc[0]+0.5, N)
        sp = [*(df['species'].tolist()), *[f"virt{i}" for i in range(N-1)]]
        out = pd.DataFrame({'species':sp[:N], 'z':zs})
        return out.sort_values('z').reset_index(drop=True)
    # interpolate evenly across the observed z-range
    zmin, zmax = float(z.min()), float(z.max())
    zs = np.linspace(zmin, zmax, N)
    # keep existing species nearest to grid points; create virtuals for gaps
    merged = []
    used = set()
    for k, zt in enumerate(zs):
        idx = int(np.argmin(np.abs(z - zt)))
        if idx not in used:
            merged.append((df['species'].iloc[idx], float(z[idx])))
            used.add(idx)
        else:
            merged.append((f"virt{k}", float(zt)))
    out = pd.DataFrame(merged, columns=['species','z']).sort_values('z').reset_index(drop=True)
    return out

def anchor_free_isotonic(df, logm_min=-6.0, logm_max=3.0):
    """
    Fit z -> q via isotonic regression to *quantiles* (rank/N) (anchor-free).
    Then map q -> log m linearly between [logm_min, logm_max].
    """
    z = df['z'].values
    N = len(df)
    # strict monotone target: mid-quantiles (0,1)->(q1,...,qN)
    # use (rank-0.5)/N for symmetry
    q = (np.arange(1, N+1) - 0.5) / N
    iz = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
    iz.fit(z, q)
    qhat = iz.predict(z)
    logm = logm_min + (logm_max - logm_min) * qhat
    m = np.exp(logm)
    out = df.copy()
    out['q_iso'] = qhat
    out['logm_anchorfree'] = logm
    out['m_anchorfree_GeV'] = m
    return out, dict(logm_min=logm_min, logm_max=logm_max)

def nearest_pdg(m):
    if m <= 0: return None, None, None
    names = list(PDG.keys()); vals = np.array([PDG[k] for k in names])
    logs = np.log(vals)
    li = math.log(m)
    j = int(np.argmin(np.abs(logs - li)))
    return names[j], float(vals[j]), abs(li - logs[j])

def main():
    ap = argparse.ArgumentParser(description="Anchor-free isotonic z->mass for ~6-10 species")
    ap.add_argument('latent_or_extended_csv', help="CSV with columns: species,z (others ignored)")
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--N', type=int, default=10, help="target number of species (auto-extend if needed)")
    ap.add_argument('--logm_min', type=float, default=-6.0, help="lower log-mass bound (GeV)")
    ap.add_argument('--logm_max', type=float, default=3.0, help="upper log-mass bound (GeV)")
    ap.add_argument('--report', action='store_true', help="write diagnostic nearest-PDG comparison (not used in fit)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df0 = load_z(args.latent_or_extended_csv)
    dfN = extend_to_n(df0, args.N)

    spec = anchor_free_isotonic(dfN, logm_min=args.logm_min, logm_max=args.logm_max)
    dfm, meta = spec
    dfm.to_csv(outdir/'anchor_free_isotonic_masses.csv', index=False)

    with open(outdir/'anchor_free_isotonic_meta.txt','w', encoding='utf-8') as f:
        f.write(f"Anchor-free isotonic map\n")
        f.write(f"Input species count: {len(df0)}  -> Extended to: {len(dfN)}\n")
        f.write(f"logm_min={meta['logm_min']}  logm_max={meta['logm_max']}\n")

    print(f"[OK] wrote {outdir/'anchor_free_isotonic_masses.csv'}")

    if args.report:
        rep = []
        for _,row in dfm.iterrows():
            name, mpdg, err = nearest_pdg(row['m_anchorfree_GeV'])
            rep.append(dict(species=row['species'], z=row['z'],
                            m_anchorfree_GeV=row['m_anchorfree_GeV'],
                            nearest_PDG=name, PDG_GeV=mpdg, log_err=err))
        pd.DataFrame(rep).to_csv(outdir/'anchor_free_isotonic_vs_pdg_report.csv', index=False)
        print(f"[OK] wrote {outdir/'anchor_free_isotonic_vs_pdg_report.csv'}")

if __name__ == '__main__':
    main()
