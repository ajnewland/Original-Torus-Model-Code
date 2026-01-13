import pandas as pd, numpy as np, sys, os
p = sys.argv[1]
df = pd.read_csv(p)
assert {'ax','ay','z','z_pred'}.issubset(df.columns), "CSV must have ax,ay,z,z_pred"
df['dz'] = df['z_pred'] - df['z']
print("=== Hot latent summary ===")
for k in ['dz','z','z_pred']:
    s = df[k].describe()
    print(f"{k}: mean={s['mean']:.6f}, std={s['std']:.6f}, min={s['min']:.6f}, max={s['max']:.6f}")
flips = df[(df['z']*df['z_pred']<0)]
print(f"\nSign flips count: {len(flips)}")
if len(flips):
    print(flips[['ax','ay','z','z_pred','dz']].head(12).to_string(index=False))