# grand_audit.py
# One-stop audit of geometric relationships + plateau/phase structure
# using your existing CSV results. Windows/CMD friendly.
#
# Outputs: a folder with summary CSVs + quick-look PNGs.
#
# Usage examples (CMD):
#  python "...\grand_audit.py" ^
#    --fermions "...\Predicted Masses\all_particles_locked.csv" ^
#    --bosonW_region "...\Predicted Masses\plateau_scan_bosons\boson_refined_W" ^
#    --bosonZ_region "...\Predicted Masses\plateau_scan_bosons\boson_refined_Z" ^
#    --bosonH_region "...\Predicted Masses\plateau_scan_bosons\boson_refined_H" ^
#    --dark_regions "...\Predicted Masses\plateau_scan_darkband\dark_band_mirror_map_fine" ^
#                   "...\Predicted Masses\plateau_scan_darkband\dark_band_push_1" ^
#                   "...\Predicted Masses\plateau_scan_darkband\dark_band_push_2" ^
#    --outdir "...\Predicted Masses\grand_audit_out"
#
# Minimal deps: numpy, pandas, matplotlib

import os, glob, math, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- small utils -----------------

def _read_csv(path):
   df = pd.read_csv(path)
   df.columns = [c.strip() for c in df.columns]
   return df

def _ensure_dir(d):
   os.makedirs(d, exist_ok=True)

def _save_df(df, path):
   df.to_csv(path, index=False)
   print(f"[WROTE] {path}")

def _euclid(a,b):
   return float(np.sqrt(np.sum((np.asarray(a)-np.asarray(b))**2)))

def _read_ax_ay(df):
   cols = {c.lower():c for c in df.columns}
   ax = cols.get("ax") or cols.get("ax0") or cols.get("a_x") or cols.get("x")
   ay = cols.get("ay") or cols.get("ay0") or cols.get("a_y") or cols.get("y")
   if not ax or not ay: raise ValueError("No ax/ay columns found.")
   out = df[[ax,ay]].copy()
   out.columns = ["ax","ay"]
   return out

def _infer_step(vals):
   u = np.unique(np.round(vals,10))
   d = np.diff(u)
   d = d[d>0]
   return float(np.min(d)) if d.size else np.nan

# --------------- Part A: relationships ---------------

def winding_rationals(locked_df):
   df = locked_df.copy()
   df.columns = [c.lower() for c in df.columns]
   need = ["species","ax","ay"]
   assert all(c in df.columns for c in need), f"locked CSV must have {need}"
   df["ay_over_ax"] = df["ay"] / df["ax"]
   out_rows=[]
   for _,r in df.iterrows():
       x = r["ay_over_ax"]
       # short continued fraction best rationals (cap denominators)
       best = (None,None,1e9)
       for q in range(2, 65):
           p = round(x*q)
           err = abs(x - p/q)
           if err < best[2]:
               best = (p,q,err)
       out_rows.append(dict(species=r["species"], ay_over_ax=x, p=best[0], q=best[1], p_over_q=best[0]/best[1], abs_err=best[2]))
   return pd.DataFrame(out_rows)

def ckm_distance_proxy(locked_df):
   # expects quark species & their ax,ay
   df = locked_df.copy()
   df.columns = [c.lower() for c in df.columns]
   up = df[df["species"].isin(["u","c","t"])][["species","ax","ay"]].set_index("species")
   dn = df[df["species"].isin(["d","s","b"])][["species","ax","ay"]].set_index("species")
   if len(up)!=3 or len(dn)!=3:
       return None, None
   D = pd.DataFrame(index=up.index, columns=dn.index, dtype=float)
   for i,(su,ru) in enumerate(up.iterrows()):
       for j,(sd,rd) in enumerate(dn.iterrows()):
           D.loc[su,sd] = _euclid([ru.ax,ru.ay],[rd.ax,rd.ay])
   # inverse-distance row-normalized proxy
   inv = 1.0/(D+1e-9)
   V = inv.div(inv.sum(axis=1), axis=0)
   return D, V

def koide_in_z(locked_df):
   # Koide-like in latent z if present
   df = locked_df.copy()
   df.columns = [c.lower() for c in df.columns]
   if "z" not in df.columns: return pd.DataFrame([dict(family="NA", Qz=np.nan, note="no z column")])
   fams = {
       "charged_leptons":["e","mu","tau"],
       "up_quarks":["u","c","t"],
       "down_quarks":["d","s","b"]
   }
   rows=[]
   for fam, names in fams.items():
       sub = df[df["species"].isin(names)][["species","z"]].set_index("species")
       if len(sub)!=3:
           rows.append(dict(family=fam, Qz=np.nan, note="missing species"))
           continue
       z = sub["z"].values.astype(float)
       if np.any(z<=0):
           rows.append(dict(family=fam, Qz=np.nan, note="nonpositive z encountered"))
           continue
       r = np.sqrt(z)
       Q = (r.sum()**2)/(3.0*np.sum(z))
       rows.append(dict(family=fam, Qz=float(Q), note="ok"))
   return pd.DataFrame(rows)

def sector_slopes(locked_df):
   # piecewise linear log m vs z per sector if both present
   df = locked_df.copy()
   df.columns = [c.lower() for c in df.columns]
   need = ["species","z","m_gev"]
   if not all(c in df.columns for c in need):
       return pd.DataFrame([dict(sector="NA", alpha=np.nan, beta=np.nan, R2=np.nan, n=np.nan)])
   # sector mapping
   def sector_of(s):
       if s in ["e","mu","tau"]: return "leptons"
       if s in ["u","c","t"]:     return "up"
       if s in ["d","s","b"]:     return "down"
       if s in ["w","z","h","g","photon","gluon"]: return "bosons"
       if s.startswith("nu"):     return "neutrinos"
       return "other"
   df["sector"] = df["species"].map(sector_of)
   rows=[]
   for sec,sub in df.groupby("sector"):
       sub = sub.dropna(subset=["z","m_gev"])
       if len(sub)<3:
           rows.append(dict(sector=sec, alpha=np.nan, beta=np.nan, R2=np.nan, n=len(sub)))
           continue
       x = sub["z"].values
       y = np.log(sub["m_gev"].values)
       A = np.vstack([x, np.ones_like(x)]).T
       alpha,beta = np.linalg.lstsq(A,y,rcond=None)[0]
       yhat = alpha*x+beta
       ssr = np.sum((yhat-y.mean())**2)
       sst = np.sum((y-y.mean())**2)
       R2  = float(ssr/sst) if sst>0 else np.nan
       rows.append(dict(sector=sec, alpha=float(alpha), beta=float(beta), R2=R2, n=len(sub)))
   return pd.DataFrame(rows)

def noanchors_ordering(locked_df):
   # monotone rank agreement between z and masses
   df = locked_df.copy()
   df.columns = [c.lower() for c in df.columns]
   if not {"species","z","m_gev"}.issubset(df.columns):
       return pd.DataFrame([dict(sector="NA", spearman_rho=np.nan, n=np.nan)])
   def sec(s):
       if s in ["e","mu","tau"]: return "leptons"
       if s in ["u","c","t"]:    return "up"
       if s in ["d","s","b"]:    return "down"
       if s.startswith("nu"):    return "neutrinos"
       if s in ["w","z","h"]:    return "bosons"
       return "other"
   df["sector"]=df["species"].map(sec)
   from scipy.stats import spearmanr
   rows=[]
   for sec,sub in df.groupby("sector"):
       sub=sub.dropna(subset=["z","m_gev"])
       if len(sub)<3:
           rows.append(dict(sector=sec, spearman_rho=np.nan, n=len(sub)))
       else:
           rho,_=spearmanr(sub["z"].values, np.log(sub["m_gev"].values))
           rows.append(dict(sector=sec, spearman_rho=float(rho), n=len(sub)))
   return pd.DataFrame(rows)

# --------------- Part B: plateau/phase ---------------

def find_plateau_csv(region_dir):
   cands=[]
   for pat in ["*plateau_points.csv", "*_plateau_points.csv", "plateau_points.csv", "*peaks_or_boundary.csv"]:
       cands+=glob.glob(os.path.join(region_dir, pat))
   return cands[0] if cands else None

def read_plateau_points(path):
   df=_read_csv(path)
   return _read_ax_ay(df)

def connected_components(points):
   if points.empty: return [],[]
   ax_step=_infer_step(points["ax"].values)
   ay_step=_infer_step(points["ay"].values)
   if not np.isfinite(ax_step): ax_step = 1e-3
   if not np.isfinite(ay_step): ay_step = 1e-3
   # grid index
   ax0=points["ax"].min(); ay0=points["ay"].min()
   gx=np.round((points["ax"].values-ax0)/ax_step).astype(int)
   gy=np.round((points["ay"].values-ay0)/ay_step).astype(int)
   buckets={}
   for i,(ix,iy) in enumerate(zip(gx,gy)):
       buckets.setdefault((ix,iy), []).append(i)
   nbrs=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
   visited=np.zeros(len(points),bool)
   comps=[]
   cents=[]
   for i in range(len(points)):
       if visited[i]: continue
       stack=[i]; visited[i]=True; comp=[i]
       while stack:
           cur=stack.pop()
           ix,iy=gx[cur],gy[cur]
           for dx,dy in nbrs:
               key=(ix+dx,iy+dy)
               if key in buckets:
                   for j in buckets[key]:
                       if not visited[j]:
                           visited[j]=True
                           stack.append(j); comp.append(j)
       comps.append(comp)
       sub=points.iloc[comp]
       cents.append((float(sub["ax"].median()), float(sub["ay"].median())))
   return comps,cents

def nearest_stats(points):
   if len(points)<2:
       return dict(n_points=len(points), dmin_med=np.nan, dmin_mean=np.nan, dmin_min=np.nan, dmin_max=np.nan)
   P=points[["ax","ay"]].values
   dmins=[]
   for i in range(len(P)):
       d2=np.sum((P-P[i])**2,axis=1); d2[i]=np.inf
       dmins.append(np.sqrt(np.min(d2)))
   dmins=np.array(dmins)
   return dict(n_points=len(points),
               dmin_med=float(np.median(dmins)),
               dmin_mean=float(np.mean(dmins)),
               dmin_min=float(np.min(dmins)),
               dmin_max=float(np.max(dmins)))

def pairwise_min_dist(setA,setB):
   if setA.empty or setB.empty: return np.inf
   A=setA[["ax","ay"]].values; B=setB[["ax","ay"]].values
   best=np.inf
   for a in A:
       d2=np.sum((B-a)**2,axis=1)
       m=np.min(d2)
       if m<best: best=m
   return float(np.sqrt(best))

# --------------- main ---------------

def main():
   ap=argparse.ArgumentParser(description="Grand audit: relationships + plateau analysis")
   ap.add_argument("--fermions", required=True, help="Locked particles CSV (species, ax, ay, z, m_GeV ...)")
   ap.add_argument("--bosonW_region", default="")
   ap.add_argument("--bosonZ_region", default="")
   ap.add_argument("--bosonH_region", default="")
   ap.add_argument("--dark_regions", nargs="*", default=[])
   ap.add_argument("--outdir", required=True)
   args=ap.parse_args()

   _ensure_dir(args.outdir)

   # ====== Part A: relationships on locked set ======
   locked=_read_csv(args.fermions)

   WR = winding_rationals(locked)
   _save_df(WR, os.path.join(args.outdir,"winding_rationals.csv"))

   CKM_D, CKM_V = ckm_distance_proxy(locked)
   if CKM_D is not None:
       _save_df(CKM_D.reset_index(), os.path.join(args.outdir,"ckm_distance_matrix.csv"))
       _save_df(CKM_V.reset_index(), os.path.join(args.outdir,"ckm_inverse_distance_proxy.csv"))

   KO = koide_in_z(locked)
   _save_df(KO, os.path.join(args.outdir,"koide_z_shift_invariant.csv"))

   SL = sector_slopes(locked)
   _save_df(SL, os.path.join(args.outdir,"sector_slopes.csv"))

   NA = noanchors_ordering(locked)
   _save_df(NA, os.path.join(args.outdir,"noanchors_ordering.csv"))

   # ====== Part B: plateau/phase for bosons and dark ======
   region_specs = []
   if args.bosonW_region: region_specs.append(("W", args.bosonW_region))
   if args.bosonZ_region: region_specs.append(("Z", args.bosonZ_region))
   if args.bosonH_region: region_specs.append(("H", args.bosonH_region))
   for i,rd in enumerate(args.dark_regions):
       region_specs.append((f"dark_{i+1}", rd))

   summary_rows=[]
   set_points = {}

   for label, rd in region_specs:
       pcsv = find_plateau_csv(rd)
       if not pcsv or not os.path.isfile(pcsv):
           print(f"[WARN] Region {label}: no plateau CSV found in {rd}")
           continue
       pts = read_plateau_points(pcsv)
       comps, cents = connected_components(pts)
       nstats = nearest_stats(pts)
       _save_df(pd.DataFrame(dict(ax=pts["ax"], ay=pts["ay"])),
                os.path.join(args.outdir, f"{label}_plateau_points_copy.csv"))

       # component table
       crows=[]
       for ci,comp in enumerate(comps):
           sub=pts.iloc[comp]
           s = nearest_stats(sub)
           crows.append(dict(region=label, comp_id=ci, **s,
                             ax_centroid=float(sub["ax"].median()),
                             ay_centroid=float(sub["ay"].median())))
       compdf=pd.DataFrame(crows)
       _save_df(compdf, os.path.join(args.outdir, f"{label}_components.csv"))

       # plot
       fig,ax=plt.subplots(figsize=(5.2,4.2),dpi=120)
       ax.scatter(pts["ax"], pts["ay"], s=6, alpha=0.6)
       if cents:
           cx=[c[0] for c in cents]; cy=[c[1] for c in cents]
           ax.scatter(cx,cy,s=24,c="red",marker="x",label="centroids")
           ax.legend()
       ax.set_title(f"{label} plateau points")
       ax.set_xlabel("ax"); ax.set_ylabel("ay"); fig.tight_layout()
       fig.savefig(os.path.join(args.outdir, f"{label}_plateau_plot.png"))
       plt.close(fig)

       summary_rows.append(dict(region=label, n_components=len(comps), **nstats,
                                source_plateau=pcsv))
       set_points[label]=pts

   if summary_rows:
       SUM = pd.DataFrame(summary_rows).sort_values("region")
       _save_df(SUM, os.path.join(args.outdir, "plateau_summary.csv"))
       # pairwise min distance across all included sets
       labels=list(set_points.keys())
       M=np.zeros((len(labels),len(labels)))
       for i,a in enumerate(labels):
           for j,b in enumerate(labels):
               if i==j: M[i,j]=0.0
               else:    M[i,j]=pairwise_min_dist(set_points[a], set_points[b])
       PM = pd.DataFrame(M, index=labels, columns=labels)
       _save_df(PM, os.path.join(args.outdir,"pairwise_plateau_min_distance.csv"))

   # ------ tiny overview file
   lines = []
   lines.append("# Grand audit outputs")
   lines.append("")
   lines.append("* winding_rationals.csv — rational approximations of ay/ax per species.")
   if CKM_D is not None:
       lines.append("* ckm_distance_matrix.csv — geometric distance matrix (up vs down).")
       lines.append("* ckm_inverse_distance_proxy.csv — row-normalized inverse-distance (|V| proxy).")
   lines.append("* koide_z_shift_invariant.csv — Koide-like check in latent z (where z>0).")
   lines.append("* sector_slopes.csv — per-sector log m vs z linear fits.")
   lines.append("* noanchors_ordering.csv — rank agreement (z vs mass) per sector.")
   if summary_rows:
       lines.append("* plateau_summary.csv — per-region components + internal spacing.")
       lines.append("* pairwise_plateau_min_distance.csv — isolation between regions.")
       lines.append("* <region>_components.csv and <region>_plateau_plot.png — details & quick look.")
   with open(os.path.join(args.outdir,"SUMMARY.md"),"w",encoding="utf-8") as f:
       f.write("\n".join(lines))
   print(f"[WROTE] {os.path.join(args.outdir,'SUMMARY.md')}")

if __name__ == "__main__":
   main()