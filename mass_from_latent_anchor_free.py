import argparse, json, math, pandas as pd
from pathlib import Path
import numpy as np

PDG_V = 246.0  # GeV (Higgs vev, for reporting Yukawas)

def load_latent(path):
    df = pd.read_csv(path)
    # be forgiving: accept columns like ['species','z'] or just ['z']
    if 'z' not in df.columns:
        # try common alternatives
        zcol = [c for c in df.columns if c.lower() in ('z','latent_z','r_norm','z_norm')]
        if not zcol:
            raise SystemExit(f"No z-like column found in {path}, columns={df.columns.tolist()}")
        df = df.rename(columns={zcol[0]:'z'})
    if 'species' not in df.columns:
        # fabricate species labels if missing
        df['species'] = [f"sp{i}" for i in range(len(df))]
    return df[['species','z']].copy()

def ladder(df, a=1.0, b=0.0, m0=1.0):
    # log m = a z + b + log m0  -> m in GeV if m0 in GeV
    logm = a*df['z'].values + b + math.log(m0)
    m = np.exp(logm)
    y = math.sqrt(2.0)/PDG_V * m
    return pd.DataFrame({'species':df['species'], 'z':df['z'], 'm_ladder_GeV':m, 'y_ladder':y})

def powerlaw(df, alpha=1.0, beta=0.0, gamma=4.0, mscale=1.0):
    s = 1.0/(1.0+np.exp(-(alpha*df['z'].values + beta)))
    m = mscale * np.power(s, gamma)
    y = math.sqrt(2.0)/PDG_V * m
    return pd.DataFrame({'species':df['species'], 'z':df['z'], 'm_power_GeV':m, 'y_power':y})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('latent_csv', help='path to latent_z.csv')
    ap.add_argument('--out', required=True, help='output folder')
    ap.add_argument('--a', type=float, default=1.0)
    ap.add_argument('--b', type=float, default=0.0)
    ap.add_argument('--m0', type=float, default=1.0)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--gamma', type=float, default=4.0)
    ap.add_argument('--mscale', type=float, default=1.0)
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    dfz = load_latent(args.latent_csv)

    df1 = ladder(dfz, a=args.a, b=args.b, m0=args.m0)
    df2 = powerlaw(dfz, alpha=args.alpha, beta=args.beta, gamma=args.gamma, mscale=args.mscale)

    m = dfz.merge(df1, on=['species','z']).merge(df2, on=['species','z'])
    m.to_csv(outdir/'anchor_free_masses.csv', index=False)
    print(f"[OK] wrote {outdir/'anchor_free_masses.csv'}")

if __name__ == '__main__':
    main()
