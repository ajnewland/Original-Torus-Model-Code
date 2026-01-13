# -*- coding: utf-8 -*-
import re, sys
from pathlib import Path
import pandas as pd

def parse_num(tok: str):
    # handle patterns like "2p58" or "1p08" -> 2.58 / 1.08
    tok = tok.strip().lower().replace("p", ".")
    return float(tok)

def main():
    if len(sys.argv) != 3:
        print("usage: python repair_latent_ax_from_src.py in_latent.csv out_latent_fixed.csv")
        sys.exit(1)

    inp, outp = sys.argv[1], sys.argv[2]
    df = pd.read_csv(inp, sep=None, engine="python", dtype=str, na_filter=False)
    # normalize column names a bit
    df.columns = [c.strip().lstrip("\ufeff").lower() for c in df.columns]

    if "src" not in df.columns:
        print("ERROR: no 'src' column to parse ax/ay from.")
        sys.exit(2)

    # Ensure ax/ay/z columns exist
    if "ax" not in df.columns: df["ax"] = ""
    if "ay" not in df.columns: df["ay"] = ""
    if "z"  not in df.columns:
        print("ERROR: no 'z' column present.")
        sys.exit(3)

    ax_out, ay_out = [], []
    for s, ax_s, ay_s in zip(df["src"], df["ax"], df["ay"]):
        ax_val, ay_val = None, None

        # try existing numeric values first
        try:
            if ax_s.strip():
                ax_val = float(ax_s)
        except:
            ax_val = None
        try:
            if ay_s.strip():
                ay_val = float(ay_s)
        except:
            ay_val = None

        # if missing, parse from src path
        if ax_val is None:
            m = re.search(r'ax([0-9p\.]+)', s, flags=re.IGNORECASE)
            if m:
                ax_val = parse_num(m.group(1))
        if ay_val is None:
            m = re.search(r'ay([0-9p\.]+)', s, flags=re.IGNORECASE)
            if m:
                ay_val = parse_num(m.group(1))

        ax_out.append(ax_val)
        ay_out.append(ay_val)

    df["ax"] = ax_out
    df["ay"] = ay_out

    # drop rows that still don't have usable ax/ay/z
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df = df[pd.to_numeric(df["ax"], errors="coerce").notna() &
            pd.to_numeric(df["ay"], errors="coerce").notna() &
            df["z"].notna()].copy()

    # keep only needed columns for the predictor
    keep = [c for c in ["src","ax","ay","z","r"] if c in df.columns]
    df[keep].to_csv(outp, index=False)
    print(f"[OK] repaired latent written to: {outp}  (rows={len(df)})")

if __name__ == "__main__":
    main()