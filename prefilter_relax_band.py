import pandas as pd
import sys

if len(sys.argv) < 3:
    print("Usage: python prefilter_relax_band.py <in.csv> <out.csv> [aymin aymax] [zmin zmax]")
    sys.exit(1)

inp, outp = sys.argv[1], sys.argv[2]
aymin = float(sys.argv[3]) if len(sys.argv) > 3 else 0.82
aymax = float(sys.argv[4]) if len(sys.argv) > 4 else 0.96
zmin  = float(sys.argv[5]) if len(sys.argv) > 5 else 0.215
zmax  = float(sys.argv[6]) if len(sys.argv) > 6 else 0.265

df = pd.read_csv(inp)

needed = ["ax","ay","S_star","r"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in input: {missing}")

band = df[(df["ay"].between(aymin, aymax)) & (df["S_star"].between(zmin, zmax))].copy()
if band.empty:
    raise SystemExit("No rows in the chosen band; widen ay/z windows.")

# Nudge boundary points in ax, ay, S_star so they are strictly interior
eps = 1e-9
def nudge(col):
    lo, hi = band[col].min(), band[col].max()
    band.loc[band[col] <= lo + 1e-15, col] = lo + eps
    band.loc[band[col] >= hi - 1e-15, col] = hi - eps

for c in ["ax", "ay", "S_star"]:
    nudge(c)

band.to_csv(outp, index=False)
print(f"Prefiltered {len(band)} points into {outp}")
print(f"ax range:  [{band['ax'].min()}, {band['ax'].max()}]")
print(f"ay range:  [{band['ay'].min()}, {band['ay'].max()}]")
print(f"S* range:  [{band['S_star'].min()}, {band['S_star'].max()}]")