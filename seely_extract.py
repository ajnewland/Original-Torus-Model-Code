#!/usr/bin/env python3
import numpy as np, scipy.linalg as la, scipy.optimize as opt, matplotlib.pyplot as plt
from math import pi
import sys, os

def build_periodic_triangulation(L):
    def vid(i,j): return (i%L) + L*(j%L)
    faces=[]
    for j in range(L):
        for i in range(L):
            v00=vid(i,j); v10=vid(i+1,j); v01=vid(i,j+1); v11=vid(i+1,j+1)
            faces.append([v00,v10,v11]); faces.append([v00,v11,v01])
    edge_set={}; edges=[]
    for f in faces:
        for a,b in zip(f, np.roll(f,-1)):
            key=tuple(sorted((a,b)))
            if key not in edge_set:
                edge_set[key]=len(edges); edges.append((key[0],key[1]))
    return edges, faces

def build_simple_hodge(L, edges, faces):
    nV=L*L; nE=len(edges)
    Af = np.ones(len(faces))*(1.0/(2*L*L))
    Lp = np.ones(nE)*(1.0/L); Ls = np.ones(nE)*(1.0/L)
    w1 = Ls/np.maximum(Lp,1e-12); w0 = np.ones(nV)*(1.0/(L*L))
    return w0,w1,Af

def build_DEC_Laplacian(L):
    edges, faces = build_periodic_triangulation(L)
    # build incidence E matrix (nE x nV)
    nV = L*L
    edge_index = {edges[i]:i for i in range(len(edges))}
    E = np.zeros((len(edges), nV))
    for ei,(u,v) in enumerate(edges):
        E[ei, v] += 1
        E[ei, u] -= 1
    w0,w1,Af = build_simple_hodge(L, edges, faces)
    W0_inv = np.diag(1.0/np.maximum(w0,1e-12))
    W1 = np.diag(w1)
    delta1 = W0_inv @ E.T @ W1
    L0 = delta1 @ E
    L0 = 0.5*(L0 + L0.T)
    # ensure nonnegative
    vals = la.eigvalsh(L0)
    vals[vals < 0] = 0.0
    return L0, vals

def heat_trace(vals, t): return np.sum(np.exp(-t * vals))

def model_K(t, d_s, a0, a1):
    return (4*pi*t)**(-d_s/2.0) * (a0 + a1 * t)

def fit_heat_trace(vals, tvals):
    Kvals = np.array([heat_trace(vals, t) for t in tvals])
    def f_log(t, d_s, a0, a1):
        return np.log(np.maximum(model_K(t, d_s, a0, a1), 1e-30))
    # initial guess
    p0 = [3.0, Kvals[0]*(4*pi*tvals[0])**(1.5), 0.01*Kvals[0]]
    popt, pcov = opt.curve_fit(f_log, tvals, np.log(Kvals+1e-30), p0=p0, maxfev=20000)
    return popt, pcov, tvals, Kvals

if __name__ == "__main__":
    outdir = "seeley_out"
    os.makedirs(outdir, exist_ok=True)
    for L in [6,8,10]:
        L0, vals = build_DEC_Laplacian(L)
        tvals = np.logspace(-3, -0.1, 20)   # adjust small-t window as needed
        popt, pcov, tvals, Kvals = fit_heat_trace(vals, tvals)
        d_s, a0, a1 = popt
        print(f"L={L}: d_s={d_s:.4f}, a0={a0:.4g}, a1={a1:.4g}")
        np.savetxt(os.path.join(outdir, f"L{L}_eigs.csv"), vals, delimiter=",")
        np.savetxt(os.path.join(outdir, f"L{L}_K.csv"), np.vstack([tvals, Kvals]).T, delimiter=",", header="t,K")
        import matplotlib.pyplot as plt
        plt.loglog(tvals, Kvals, 'o', label='K(t)')
        plt.loglog(tvals, model_K(tvals, *popt), '-', label=f'fit ds={d_s:.3f}')
        plt.legend(); plt.xlabel('t'); plt.ylabel('K(t)')
        plt.savefig(os.path.join(outdir, f"L{L}_Kfit.png")); plt.clf()