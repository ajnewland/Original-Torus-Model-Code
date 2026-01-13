#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (script content same as prior cell; rewriting entirely)
import argparse, os, numpy as np, pandas as pd
def idw_interp(xp, yp, zp, X, Y, power=2, eps=1e-12):
    Zi = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dx = xp - X[i, j]; dy = yp - Y[i, j]
            dist2 = dx*dx + dy*dy
            if np.any(dist2 < 1e-14): Zi[i, j] = zp[np.argmin(dist2)]
            else:
                w = 1.0 / np.power(dist2 + eps, power/2.0); Zi[i, j] = np.sum(w * zp) / np.sum(w)
    return Zi
def gaussian_2d(x, y, x0, y0, sigma): return np.exp(-((x-x0)**2+(y-y0)**2)/(2.0*sigma**2))
def nearest_unitary(A): U,s,Vh=np.linalg.svd(A,full_matrices=False); return U@Vh
def jarlskog_from_unitary(V):
    return np.imag(V[0,0]*V[1,1]*np.conj(V[0,1])*np.conj(V[1,0]))
def main():
    ap=argparse.ArgumentParser(description="Predict CKM from torsion-phase overlaps")
    ap.add_argument("--torsion_csv",required=True); ap.add_argument("--locked_csv",required=True)
    ap.add_argument("--sigma",type=float,default=0.025); ap.add_argument("--grid_n",type=int,default=220)
    ap.add_argument("--outdir",default="ckm_out"); args=ap.parse_args(); os.makedirs(args.outdir,exist_ok=True)
    tor=pd.read_csv(args.torsion_csv); tor.columns=[c.lower() for c in tor.columns]
    for req in ["species","sector","ax","ay","t_eff"]:
        if req not in tor.columns: raise ValueError(f"torsion_csv must include '{req}'")
    locked=pd.read_csv(args.locked_csv); locked.columns=[c.lower() for c in locked.columns]
    def get_pos(sp):
        rows=tor.loc[tor["species"]==sp]; 
        if rows.empty: raise ValueError(f"Species '{sp}' not in torsion_csv.")
        r=rows.iloc[0]; return float(r["ax"]),float(r["ay"])
    up_species=["u","c","t"]; down_species=["d","s","b"]
    up_pos=np.array([get_pos(s) for s in up_species]); down_pos=np.array([get_pos(s) for s in down_species])
    ax_min,ax_max=float(tor["ax"].min()),float(tor["ax"].max()); ay_min,ay_max=float(tor["ay"].min()),float(tor["ay"].max())
    pad_x=0.02*(ax_max-ax_min if ax_max>ax_min else 1.0); pad_y=0.02*(ay_max-ay_min if ay_max>ay_min else 1.0)
    x=np.linspace(ax_min-pad_x,ax_max+pad_x,args.grid_n); y=np.linspace(ay_min-pad_y,ay_max+pad_y,args.grid_n)
    X,Y=np.meshgrid(x,y)
    Tv=tor["t_eff"].to_numpy(float); Xp=tor["ax"].to_numpy(float); Yp=tor["ay"].to_numpy(float)
    Tgrid=idw_interp(Xp,Yp,Tv,X,Y,power=2); dTx,dTy=np.gradient(Tgrid,x,y,edge_order=2); phi=np.arctan2(dTy,dTx)
    sigma=float(args.sigma); G_up=[gaussian_2d(X,Y,ax0,ay0,sigma) for (ax0,ay0) in up_pos]
    G_down=[gaussian_2d(X,Y,ax0,ay0,sigma) for (ax0,ay0) in down_pos]
    E=np.exp(1j*phi); A=np.zeros((3,3),dtype=complex)
    for i in range(3):
        for j in range(3):
            W=G_up[i]*G_down[j]; num=np.sum(W*E); den=np.sqrt(np.sum(G_up[i]**2)*np.sum(G_down[j]**2)); A[i,j]=num/den
    V=nearest_unitary(A); V_abs=np.abs(V)
    pred_df=pd.DataFrame(V_abs,index=["u","c","t"],columns=["d","s","b"])
    pred_path=os.path.join(args.outdir,"predicted_ckm.csv"); pred_df.to_csv(pred_path,float_format="%.6f")
    V_pdg=np.array([[0.97401,0.22650,0.00361],[0.22636,0.97320,0.04053],[0.00854,0.03978,0.999172]])
    comp=pd.DataFrame({
        "pred_Vud":[V_abs[0,0]],"PDG_Vud":[V_pdg[0,0]],"delta_Vud":[V_abs[0,0]-V_pdg[0,0]],
        "pred_Vus":[V_abs[0,1]],"PDG_Vus":[V_pdg[0,1]],"delta_Vus":[V_abs[0,1]-V_pdg[0,1]],
        "pred_Vub":[V_abs[0,2]],"PDG_Vub":[V_pdg[0,2]],"delta_Vub":[V_abs[0,2]-V_pdg[0,2]],
        "pred_Vcd":[V_abs[1,0]],"PDG_Vcd":[V_pdg[1,0]],"delta_Vcd":[V_abs[1,0]-V_pdg[1,0]],
        "pred_Vcs":[V_abs[1,1]],"PDG_Vcs":[V_pdg[1,1]],"delta_Vcs":[V_abs[1,1]-V_pdg[1,1]],
        "pred_Vcb":[V_abs[1,2]],"PDG_Vcb":[V_pdg[1,2]],"delta_Vcb":[V_abs[1,2]-V_pdg[1,2]],
        "pred_Vtd":[V_abs[2,0]],"PDG_Vtd":[0.00854],"delta_Vtd":[V_abs[2,0]-0.00854],
        "pred_Vts":[V_abs[2,1]],"PDG_Vts":[0.03978],"delta_Vts":[V_abs[2,1]-0.03978],
        "pred_Vtb":[V_abs[2,2]],"PDG_Vtb":[0.999172],"delta_Vtb":[V_abs[2,2]-0.999172],
    })
    comp_path=os.path.join(args.outdir,"ckm_comparison.csv"); comp.to_csv(comp_path,index=False,float_format="%.6f")
    J=np.imag(V[0,0]*V[1,1]*np.conj(V[0,1])*np.conj(V[1,0]))
    print("=== Geometric CKM Prediction ===")
    print("Sigma:",sigma," Grid:",args.grid_n,"x",args.grid_n)
    print("\n|V_CKM|:\n",pred_df.to_string(float_format=lambda v:f"{v:0.6f}"))
    print("\nApprox. Jarlskog J =",f"{J:.6e}")
    print("Saved:",pred_path,"\nSaved:",comp_path)
if __name__=="__main__":
    main()
