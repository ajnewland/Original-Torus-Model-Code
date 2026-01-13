# relationships_audit.py
# One-shot audit over fermions/neutrinos/bosons with optional grids.
# Windows/CMD friendly. No external deps beyond: python>=3.9, pandas, numpy.
# (matplotlib not required; we emit Markdown + CSV text summaries.)

import argparse
import math
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ----------------------------
# Small utilities
# ----------------------------

def read_csv_safely(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        # Gentle strip; no deprecated applymap
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        return df
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None

def ensure_cols(df: pd.DataFrame, need: List[str], label: str):
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}. Found: {list(df.columns)}")

def continued_fraction_rational(x: float, max_den: int = 64) -> Tuple[int,int,float]:
    """Best rational p/q approx with denominator <= max_den (Farey/CFE mix)."""
    # Guard
    if not np.isfinite(x):
        return (0,1,0.0)
    from fractions import Fraction
    frac = Fraction(x).limit_denominator(max_den)
    p, q = frac.numerator, frac.denominator
    return p, q, p/q

def euclid_dist(ax1, ay1, ax2, ay2) -> float:
    return float(math.hypot(ax1-ax2, ay1-ay2))

def sector_of_species(name: str) -> str:
    n = name.lower()
    if n in ["u","c","t"]:
        return "up"
    if n in ["d","s","b"]:
        return "down"
    if n in ["e","mu","tau","tauon","tau_lepton"]:
        return "leptons"
    if n.startswith("nu"):
        return "neutrinos"
    if n in ["h","w","z","gamma","photon","g","gluon"]:
        return "bosons"
    return "other"

def safe_float_col(df: pd.DataFrame, name_options: List[str]) -> Optional[str]:
    for n in name_options:
        if n in df.columns:
            return n
    return None

def write_table(path: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def md_header(f, title: str, level: int = 2):
    f.write("\n" + "#"*level + f" {title}\n\n")

# ----------------------------
# Load core tables
# ----------------------------

def load_species_table(path: str, label: str) -> pd.DataFrame:
    df = read_csv_safely(path)
    if df is None:
        raise ValueError(f"{label}: cannot read file {path}")
    # Try to standardize column names a bit
    # Expect at least: species, ax, ay; prefer z_target else z_pred; mass column optional
    needed_min = ["species","ax","ay"]
    ensure_cols(df, needed_min, label)
    # Standardize 'species'
    df["species"] = df["species"].astype(str)
    # Pick z column
    zcol = None
    for cand in ["z_target","z","z_pred"]:
        if cand in df.columns:
            zcol = cand
            break
    if zcol is None:
        raise ValueError(f"{label}: need a z-like column (one of z_target,z,z_pred)")
    df = df.copy()
    df["z_use"] = pd.to_numeric(df[zcol], errors="coerce")
    # mass col
    mcol = safe_float_col(df, ["m_GeV","mass_GeV","mass","m"])
    if mcol is None:
        # ok, but warn
        print(f"[INFO] {label}: no mass column found; slope/ordering tests that need mass will be skipped.")
        df["m_GeV"] = np.nan
    else:
        df["m_GeV"] = pd.to_numeric(df[mcol], errors="coerce")
    # Useful ratios
    df["ay_over_ax"] = df["ay"] / df["ax"]
    df["sector"] = df["species"].map(sector_of_species)
    return df

# ----------------------------
# (1) Winding numbers (continued fractions)
# ----------------------------

def winding_report(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_all.iterrows():
        p,q,pq = continued_fraction_rational(float(r["ay_over_ax"]), max_den=64)
        rows.append({
            "species": r["species"],
            "ay_over_ax": float(r["ay_over_ax"]),
            "p": p, "q": q, "p_over_q": pq,
            "abs_err": abs(float(r["ay_over_ax"])-pq)
        })
    out = pd.DataFrame(rows)
    return out

# ----------------------------
# (3) CKM-like distances
# ----------------------------

def ckm_proxy(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def pick(names):
        return df_all[df_all["species"].str.lower().isin(names)].copy()
    up  = pick(["u","c","t"]).sort_values("species")
    down = pick(["d","s","b"]).sort_values("species")
    if len(up)!=3 or len(down)!=3:
        return pd.DataFrame(), pd.DataFrame()
    D = np.zeros((3,3))
    for i, (_, ui) in enumerate(up.iterrows()):
        for j, (_, dj) in enumerate(down.iterrows()):
            D[i,j] = euclid_dist(ui["ax"], ui["ay"], dj["ax"], dj["ay"])
    # Row-normalized inverse distance (avoid divide-by-zero)
    invD = 1.0 / np.clip(D, 1e-12, None)
    row_sums = invD.sum(axis=1, keepdims=True)
    V = invD / np.clip(row_sums, 1e-12, None)
    dm = pd.DataFrame(D, index=list(up["species"]), columns=list(down["species"]))
    vm = pd.DataFrame(V, index=list(up["species"]), columns=list(down["species"]))
    return dm, vm

# ----------------------------
# (4) Koide-like in z (shift-invariant)
# ----------------------------

def koide_z_shift_invariant(zvals: List[float]) -> Optional[float]:
    z = np.array(zvals, dtype=float)
    if len(z) != 3 or not np.all(np.isfinite(z)):
        return None
    # Shift to positive, add tiny epsilon
    eps = 1e-12
    z_shift = z - np.min(z) + eps
    # If still numerically nonpositive, bail
    if np.any(z_shift <= 0):
        return None
    num = float(z_shift.sum())
    den = float(np.sum(np.sqrt(z_shift)))**2
    if den == 0:
        return None
    return num/den

def koide_family(df: pd.DataFrame, names: List[str]) -> Optional[float]:
    sub = df[df["species"].str.lower().isin(names)]
    if len(sub) != 3:
        return None
    return koide_z_shift_invariant(list(sub.sort_values("species")["z_use"]))

# ----------------------------
# (5) Mirror-centroid (needs bosons)
# ----------------------------

def centroid(df: pd.DataFrame) -> Tuple[float,float]:
    return (float(df["ax"].mean()), float(df["ay"].mean()))

# ----------------------------
# (6) Morse counting on grids
# ----------------------------

def load_grid(path: str) -> Optional[Tuple[np.ndarray,np.ndarray,np.ndarray]]:
    df = read_csv_safely(path)
    if df is None:
        return None
    # accept z or z_pred
    zcol = "z" if "z" in df.columns else ("z_pred" if "z_pred" in df.columns else None)
    if zcol is None:
        print(f"[WARN] Grid missing z/z_pred: {path}")
        return None
    try:
        ax = np.sort(df["ax"].unique())
        ay = np.sort(df["ay"].unique())
        # pivot
        Z = df.pivot(index="ay", columns="ax", values=zcol).sort_index().values
        # Ensure shapes match
        if Z.shape != (len(ay), len(ax)):
            print(f"[WARN] Grid pivot shape mismatch in {path}")
            return None
        return ax, ay, Z
    except Exception as e:
        print(f"[WARN] Failed grid pivot {path}: {e}")
        return None

def morse_count(Z: np.ndarray) -> Tuple[int,int,int]:
    """Count interior minima/maxima/saddles by 4-neighborhood comparisons."""
    ny, nx = Z.shape
    mins = maxs = sads = 0
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            nb = [Z[j-1,i], Z[j+1,i], Z[j,i-1], Z[j,i+1]]
            center = Z[j,i]
            less = sum(center < v for v in nb)
            greater = sum(center > v for v in nb)
            if less == 4:
                mins += 1
            elif greater == 4:
                maxs += 1
            else:
                # crude saddle proxy
                sads += 1
    return mins, maxs, sads

# ----------------------------
# (7) Charge-weighted z (fermions)
# ----------------------------

CHARGE = {
    "u": 2/3, "c": 2/3, "t": 2/3,
    "d": -1/3, "s": -1/3, "b": -1/3,
    "e": -1.0, "mu": -1.0, "tau": -1.0,
    "nu": 0.0, "nu1": 0.0, "nu2": 0.0, "nu3": 0.0,
}

def sum_Qz(df_f: pd.DataFrame) -> float:
    tot = 0.0
    for _, r in df_f.iterrows():
        sp = r["species"].lower()
        q = CHARGE.get(sp, 0.0)
        z = float(r["z_use"]) if np.isfinite(r["z_use"]) else 0.0
        tot += q*z
    return float(tot)

# ----------------------------
# (8) Sector slopes: log m vs z
# ----------------------------

def sector_slopes(df_all: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sector in ["up","down","leptons","neutrinos","bosons"]:
        sub = df_all[(df_all["sector"]==sector) & np.isfinite(df_all["m_GeV"]) & np.isfinite(df_all["z_use"])]
        if len(sub) < 2:
            continue
        x = sub["z_use"].values.astype(float)
        y = np.log(np.clip(sub["m_GeV"].values.astype(float), 1e-300, None))
        # raw slope
        A = np.vstack([x, np.ones_like(x)]).T
        alpha, beta = np.linalg.lstsq(A, y, rcond=None)[0]
        yhat = alpha*x + beta
        # R^2
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2)) if len(y)>1 else 0.0
        R2 = 1.0 - ss_res/ss_tot if ss_tot>0 else np.nan

        row = {"sector": sector, "alpha_raw": alpha, "beta_raw": beta, "R2": R2, "n": len(sub)}

        # neutrino normalized slope too
        if sector == "neutrinos" and len(sub) >= 2:
            xn = (x - np.mean(x)) / (np.std(x) if np.std(x)>0 else 1.0)
            A2 = np.vstack([xn, np.ones_like(xn)]).T
            alpha_n, beta_n = np.linalg.lstsq(A2, y, rcond=None)[0]
            row["alpha_norm"] = float(alpha_n)
            row["beta_norm"]  = float(beta_n)
        out.append(row)
    return pd.DataFrame(out)

# ----------------------------
# (9) No-anchors ordering: Spearman ρ
# ----------------------------

def sector_ordering(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sector in ["up","down","leptons","neutrinos","bosons"]:
        sub = df_all[(df_all["sector"]==sector) & np.isfinite(df_all["m_GeV"]) & np.isfinite(df_all["z_use"])]
        if len(sub) < 2:
            continue
        rho = sub["z_use"].rank().corr(np.log(np.clip(sub["m_GeV"],1e-300,None)).rank(), method="spearman")
        rows.append({"sector": sector, "spearman_rho": float(rho), "n": int(len(sub))})
    return pd.DataFrame(rows)

# ----------------------------
# (10) Boson band width ratio → sin^2θ proxy
# ----------------------------

def local_band_width(ax: np.ndarray, ay: np.ndarray, Z: np.ndarray, ax0: float, ay0: float) -> Optional[Tuple[float,float]]:
    """
    A simple local width proxy: pick the closest grid index to (ax0,ay0).
    Around it, take a small window and compute stdev along x and y of Z-level set
    weights. To keep it robust, we just return std_x = std of ax in window,
    std_y = std of ay in window (uniform weights).
    """
    if ax.size<5 or ay.size<5:
        return None
    ix = int(np.argmin(np.abs(ax-ax0)))
    iy = int(np.argmin(np.abs(ay-ay0)))
    # 11x11 neighborhood if possible
    dx = 5
    x0 = max(0, ix-dx); x1 = min(len(ax)-1, ix+dx)
    y0 = max(0, iy-dx); y1 = min(len(ay)-1, iy+dx)
    axw = ax[x0:x1+1]
    ayw = ay[y0:y1+1]
    if len(axw)<3 or len(ayw)<3:
        return None
    std_x = float(np.std(axw))
    std_y = float(np.std(ayw))
    return std_x, std_y

def band_ratio_to_sin2(std_x: float, std_y: float) -> float:
    # Interpreting ratio as mixing: s = std_y / (std_x + std_y)
    denom = std_x + std_y
    return float(std_y/denom) if denom>0 else np.nan

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="All-in-one relationships audit.")
    ap.add_argument("--fermions_csv", required=True)
    ap.add_argument("--neutrinos_csv", default=None)
    ap.add_argument("--bosons_csv", default=None)

    ap.add_argument("--grid_W", default=None)
    ap.add_argument("--grid_Z", default=None)
    ap.add_argument("--grid_H", default=None)

    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load core tables
    df_f = load_species_table(args.fermions_csv, "fermions_csv")
    df_n = load_species_table(args.neutrinos_csv, "neutrinos_csv") if args.neutrinos_csv else None
    df_b = load_species_table(args.bosons_csv, "bosons_csv") if args.bosons_csv else None

    # Combined for cross-sector analyses
    frames = [df_f]
    if df_n is not None: frames.append(df_n)
    if df_b is not None: frames.append(df_b)
    df_all = pd.concat(frames, ignore_index=True)

    # 1) Winding rationals
    wind = winding_report(df_all)
    write_table(os.path.join(args.outdir, "winding_rationals.csv"), wind)

    # 3) CKM-like matrix
    dm, vm = ckm_proxy(df_all)
    if not dm.empty:
        write_table(os.path.join(args.outdir, "ckm_distance_matrix.csv"), dm)
        write_table(os.path.join(args.outdir, "ckm_inverse_distance_proxy.csv"), vm)

    # 4) Koide-like in z (shift-invariant)
    koide_rows = []
    fams = {
        "charged_leptons": ["e","mu","tau"],
        "up_quarks": ["u","c","t"],
        "down_quarks": ["d","s","b"],
    }
    for fam, names in fams.items():
        val = koide_family(df_all, names)
        koide_rows.append({"family": fam, "Qz_shift_invariant": val})
    koide_df = pd.DataFrame(koide_rows)
    write_table(os.path.join(args.outdir, "koide_z_shift_invariant.csv"), koide_df)

    # 5) Mirror centroid
    cent_lines = []
    fc = centroid(df_f)
    cent_lines.append(f"fermion centroid: ({fc[0]:.6f}, {fc[1]:.6f})")
    bc = None
    if df_b is not None and len(df_b)>0:
        bc = centroid(df_b)
        cent_lines.append(f"boson centroid  : ({bc[0]:.6f}, {bc[1]:.6f})")

    # 6) Morse on grids
    morse_rows = []
    for tag, p in [("W", args.grid_W), ("Z", args.grid_Z), ("H", args.grid_H)]:
        if p:
            g = load_grid(p)
            if g is not None:
                _,_,Z = g
                mins,maxs,sads = morse_count(Z)
                morse_rows.append({"grid": tag, "mins": mins, "maxs": maxs, "saddles": sads})
    morse_df = pd.DataFrame(morse_rows)
    if not morse_df.empty:
        write_table(os.path.join(args.outdir, "morse_counts.csv"), morse_df)

    # 7) Sum Q z (fermions only)
    qz_sum = sum_Qz(df_f)

    # 8) Sector slopes
    slopes_df = sector_slopes(df_all)
    write_table(os.path.join(args.outdir, "sector_slopes.csv"), slopes_df)

    # 9) No-anchors ordering (Spearman ρ)
    ord_df = sector_ordering(df_all)
    write_table(os.path.join(args.outdir, "noanchors_ordering.csv"), ord_df)

    # 10) Boson band width ratio -> sin^2 proxy (needs grids + boson centers)
    band_rows = []
    if df_b is not None:
        # Try match by species names W/Z/H if present with ax,ay
        centers = {}
        for _, r in df_b.iterrows():
            s = r["species"].upper()
            if s in ["W","Z","H"]:
                centers[s] = (float(r["ax"]), float(r["ay"]))
        for s, grid_path in [("W", args.grid_W), ("Z", args.grid_Z), ("H", args.grid_H)]:
            if grid_path and s in centers:
                g = load_grid(grid_path)
                if g is not None:
                    axv, ayv, Z = g
                    ax0, ay0 = centers[s]
                    ww = local_band_width(axv, ayv, Z, ax0, ay0)
                    if ww is not None:
                        stdx, stdy = ww
                        s_est = band_ratio_to_sin2(stdx, stdy)
                        band_rows.append({"species": s, "std_ax": stdx, "std_ay": stdy, "sin2_proxy": s_est})
    if band_rows:
        write_table(os.path.join(args.outdir, "boson_bandwidth_sin2_proxy.csv"), pd.DataFrame(band_rows))

    # ------------- Markdown SUMMARY -------------
    summary_path = os.path.join(args.outdir, "SUMMARY.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        md_header(f, "Relationships Audit Summary", level=1)

        md_header(f, "Inputs")
        f.write(f"- Fermions CSV: `{args.fermions_csv}`\n")
        if args.neutrinos_csv: f.write(f"- Neutrinos CSV: `{args.neutrinos_csv}`\n")
        if args.bosons_csv:   f.write(f"- Bosons CSV: `{args.bosons_csv}`\n")
        if args.grid_W: f.write(f"- W grid: `{args.grid_W}`\n")
        if args.grid_Z: f.write(f"- Z grid: `{args.grid_Z}`\n")
        if args.grid_H: f.write(f"- H grid: `{args.grid_H}`\n")

        md_header(f, "1) Winding Numbers (continued fractions)")
        f.write("Saved: `winding_rationals.csv`\n\n")
        f.write(wind.to_string(index=False) + "\n\n")

        md_header(f, "3) CKM distance proxy")
        if dm.empty:
            f.write("_Skipped (need u,c,t and d,s,b present)._\n")
        else:
            f.write("Distance matrix (ax,ay Euclidean):\n\n")
            f.write(dm.to_string() + "\n\n")
            f.write("Row-normalized inverse-distance (|V| proxy):\n\n")
            f.write(vm.to_string() + "\n\n")

        md_header(f, "4) Koide-like in z (shift-invariant)")
        f.write("Saved: `koide_z_shift_invariant.csv`\n\n")
        f.write(koide_df.to_string(index=False) + "\n\n")

        md_header(f, "5) Centroids")
        for line in cent_lines:
            f.write(line + "\n")

        md_header(f, "6) Morse counts on grids")
        if morse_df.empty:
            f.write("_No grids provided or could not parse._\n")
        else:
            f.write(morse_df.to_string(index=False) + "\n")

        md_header(f, "7) Charge-weighted sum Q*z (fermions)")
        f.write(f"`Sum Q*z = {qz_sum:.6f}`\n")

        md_header(f, "8) Sector slopes: log m vs z")
        f.write("Saved: `sector_slopes.csv`\n\n")
        f.write(slopes_df.to_string(index=False) + "\n\n")
        if "neutrinos" in list(slopes_df["sector"]):
            f.write("_For neutrinos, `alpha_norm`/`beta_norm` use locally normalized z._\n")

        md_header(f, "9) No-anchors ordering (Spearman ρ)")
        f.write("Saved: `noanchors_ordering.csv`\n\n")
        f.write(ord_df.to_string(index=False) + "\n\n")

        md_header(f, "10) Boson band width → sin²θ proxy")
        if band_rows:
            f.write("Saved: `boson_bandwidth_sin2_proxy.csv`\n\n")
            f.write(pd.DataFrame(band_rows).to_string(index=False) + "\n")
        else:
            f.write("_Skipped (need boson CSV with W/Z/H centers and matching grids)._")

    print(f"[DONE] Wrote summary: {summary_path}")
    print(f"[DONE] CSVs in: {args.outdir}")

if __name__ == "__main__":
    main()