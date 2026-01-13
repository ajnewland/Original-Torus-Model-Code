import pandas as pd, sys
inp, outp = sys.argv[1], sys.argv[2]
df = pd.read_csv(inp)
# keep a generous boson neighborhood
band = df[(df['ay'].between(0.84, 0.95)) & (df['S_star'].between(0.22, 0.26))]
band.to_csv(outp, index=False)
print(f"wrote {len(band)} rows -> {outp}")