#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo heat-kernel & spectral-dimension analysis
=====================================================

Reproduces the "Rule of Three" plateaus by computing the heat trace
Theta(t) = Tr[exp(-t L)] on ensembles of graphs that represent the
spinfoam micro-geometry, then d_s(t) = -2 d log Theta / d log t.

Two backends:
 - "eigen": exact using Laplacian eigenvalues (small graphs).
 - "hutch": stochastic trace estimator (large graphs), using expm_multiply.

You can build graphs either procedurally ("fractal") or from a latent
point cloud ("latent") using k-nearest-neighbour edges.
"""

import argparse, os, json
import numpy as np
import pandas as pd

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt


# ----------------------------- Graph builders -----------------------------

def build_fractal_graph(n=600, k=6, p_branch=0.25, rewire=0.0, seed=42):
   """
   Procedural "fractal-like" graph:
     1) Start from random 2D points.
     2) Connect k nearest neighbours (k-NN).
     3) Stochastically "branch" some edges by inserting a vertex (prob p_branch)
        to mimic Koch-like refinements and reduce effective dimension.
     4) Optionally small-world rewire (fraction 'rewire').

   Returns: scipy CSR adjacency matrix (unweighted, symmetric).
   """
   rng = np.random.default_rng(seed)
   P = rng.uniform(0, 1, size=(n, 2))
   tree = cKDTree(P)
   nn_d, nn_i = tree.query(P, k=k+1)  # includes self
   rows, cols = [], []
   for i in range(n):
       for j in nn_i[i,1:]:
           rows.append(i); cols.append(int(j))
           rows.append(int(j)); cols.append(i)
   A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n,n)).tocsr()
   A = (A + A.T); A.data[:] = 1.0
   A.setdiag(0); A.eliminate_zeros()

   # stochastic "branch" some edges: split edge i-j by inserting new node v
   to_split = []
   Acoo = A.tocoo()
   for i,j in zip(Acoo.row, Acoo.col):
       if i < j and rng.random() < p_branch:
           to_split.append((i,j))
   m = len(to_split)
   if m > 0:
       newN = n + m
       A2 = sp.lil_matrix((newN, newN))
       A2[:n,:n] = A
       idx_new = np.arange(n, newN)
       for k_, (i,j) in enumerate(to_split):
           v = idx_new[k_]
           A2[i,j] = 0; A2[j,i] = 0
           A2[i,v] = 1; A2[v,i] = 1
           A2[v,j] = 1; A2[j,v] = 1
       A = A2.tocsr()
       n = newN

   # optional small-world rewiring
   if rewire > 0:
       A = A.tolil()
       rng_edges = list(zip(*A.nonzero()))
       rng.shuffle(rng_edges)
       num_rewire = int(rewire * len(rng_edges) / 2)
       for (i,j) in rng_edges[:num_rewire]:
           if i == j: continue
           A[i,j] = 0; A[j,i] = 0
           k_idx = rng.integers(0, n-1)
           if k_idx == i: k_idx = (k_idx+1) % n
           A[i,k_idx] = 1; A[k_idx,i] = 1
       A = A.tocsr()
       A.setdiag(0); A.eliminate_zeros()

   return A


def build_knn_graph_from_latent(latent_csv, k=6):
   """
   Build an undirected k-NN graph from a latent CSV with columns 'ax','ay'.
   """
   df = pd.read_csv(latent_csv)
   cols = {c.lower(): c for c in df.columns}
   def pick(*names):
       for n in names:
           if n in cols: return cols[n]
       return None
   cx = pick("ax","a_x","alpha_x"); cy = pick("ay","a_y","alpha_y")
   if cx is None or cy is None:
       raise ValueError("latent CSV must contain ax/ay (or a_x/a_y/alpha_x/alpha_y)")
   P = df[[cx,cy]].to_numpy(float)
   n = P.shape[0]
   tree = cKDTree(P)
   nn_d, nn_i = tree.query(P, k=min(k+1, n))
   rows, cols = [], []
   for i in range(n):
       for j in nn_i[i,1:]:
           rows.append(i); cols.append(int(j))
           rows.append(int(j)); cols.append(i)
   A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n,n))
   A = (A + A.T); A.data[:] = 1.0
   A.setdiag(0); A.eliminate_zeros()
   return A


# ----------------------------- Laplacian & heat kernel -----------------------------

def laplacian_from_adj(A):
   d = np.array(A.sum(axis=1)).ravel()
   D = sp.diags(d)
   L = D - A
   return L


def heat_trace_eigen(L, t_vals, k_max=None):
   """
   Exact/partial spectrum method. For small graphs, compute all eigenvalues.
   For larger graphs, compute k smallest eigenvalues (dominant at large t).
   """
   n = L.shape[0]
   if k_max is None or k_max >= n:
       evals = np.linalg.eigvalsh(L.toarray())
   else:
       k = min(max(2, k_max), n-2)
       evals = spla.eigsh(L, k=k, which='SM', return_eigenvectors=False)
   evals = np.clip(evals, 0, None)
   Theta = np.array([np.sum(np.exp(-t*evals)) for t in t_vals], dtype=float)
   return Theta


def heat_trace_hutch(L, t_vals, m=64, seed=0):
   """
   Hutchinson trace estimator with expm_multiply:
       Tr[exp(-tL)] ≈ (1/m) Σ v^T expm(-tL) v,  v ∈ {±1}^n
   """
   rng = np.random.default_rng(seed)
   n = L.shape[0]
   Theta = np.zeros_like(t_vals, dtype=float)
   for j in range(m):
       v = rng.choice([-1.0, 1.0], size=n)
       for idx, t in enumerate(t_vals):
           y = spla.expm_multiply((-t) * L, v)
           Theta[idx] += float(v @ y)
   Theta /= m
   return Theta


def spectral_dimension(Theta, t_vals):
   logt = np.log(t_vals)
   logs = np.log(Theta)
   dlogs = np.gradient(logs, logt)
   return -2.0 * dlogs


# ----------------------------- Monte Carlo driver -----------------------------

def run_ensemble(args):
   os.makedirs(args.outdir, exist_ok=True)
   t_vals = np.logspace(args.tmin_exp, args.tmax_exp, args.nt, base=10.0)
   all_theta, all_ds, meta = [], [], []

   for r in range(args.runs):
       seed = args.seed + r
       if args.mode == "fractal":
           A = build_fractal_graph(n=args.nodes, k=args.k, p_branch=args.p_branch, rewire=args.rewire, seed=seed)
       elif args.mode == "latent":
           if not args.latent:
               raise ValueError("--latent CSV required for --mode latent")
           A = build_knn_graph_from_latent(args.latent, k=args.k)
       else:
           raise ValueError("Unknown mode")

       L = laplacian_from_adj(A)
       if args.method == "eigen":
           Theta = heat_trace_eigen(L, t_vals, k_max=args.k_max)
       else:
           Theta = heat_trace_hutch(L, t_vals, m=args.hutch_m, seed=seed)
       ds = spectral_dimension(Theta, t_vals)

       all_theta.append(Theta)
       all_ds.append(ds)
       meta.append({"seed": seed, "n": int(A.shape[0]), "m": int(A.nnz//2)})

   all_theta = np.vstack(all_theta)
   all_ds = np.vstack(all_ds)
   meta_df = pd.DataFrame(meta)

   # ensemble stats
   theta_mean = all_theta.mean(axis=0)
   theta_lo = np.percentile(all_theta, 16, axis=0)
   theta_hi = np.percentile(all_theta, 84, axis=0)

   ds_mean = all_ds.mean(axis=0)
   ds_lo = np.percentile(all_ds, 16, axis=0)
   ds_hi = np.percentile(all_ds, 84, axis=0)

   # save CSVs
   pd.DataFrame({"t": t_vals, "theta_mean": theta_mean, "theta_lo": theta_lo, "theta_hi": theta_hi}) \
       .to_csv(os.path.join(args.outdir, "theta_mean.csv"), index=False)
   pd.DataFrame({"t": t_vals, "ds_mean": ds_mean, "ds_lo": ds_lo, "ds_hi": ds_hi}) \
       .to_csv(os.path.join(args.outdir, "ds_mean.csv"), index=False)
   meta_df.to_csv(os.path.join(args.outdir, "ensemble_meta.csv"), index=False)

   # plot
   fig, ax = plt.subplots(2,1, figsize=(6,7), sharex=True, dpi=150)
   ax[0].loglog(t_vals, theta_mean/theta_mean[0], lw=2)
   ax[0].fill_between(t_vals, theta_lo/theta_mean[0], theta_hi/theta_mean[0], alpha=0.2)
   ax[0].set_ylabel(r"$\Theta(t)/\Theta(t_{\min})$")
   ax[0].grid(True, which="both", alpha=0.2)

   ax[1].semilogx(t_vals, ds_mean, lw=2)
   ax[1].fill_between(t_vals, ds_lo, ds_hi, alpha=0.2)
   for x in [10**args.split1, 10**args.split2, 10**args.split3]:
       ax[1].axvline(x, ls="--", color="gray", alpha=0.7)
   ax[1].set_xlabel(r"Diffusion scale $t$")
   ax[1].set_ylabel(r"$d_s(t)$")
   ax[1].grid(True, which="both", alpha=0.2)

   plt.tight_layout()
   fig.savefig(os.path.join(args.outdir, "spectral_dimension_rule_of_three.png"))
   plt.close(fig)

   print(json.dumps({
       "outdir": args.outdir,
       "runs": args.runs,
       "nt": args.nt,
       "nodes": args.nodes,
       "mode": args.mode,
       "method": args.method
   }, indent=2))


def parse_args(argv=None):
   ap = argparse.ArgumentParser(description="Monte Carlo heat-kernel & spectral-dimension analysis")
   ap.add_argument("--mode", choices=["fractal","latent"], default="fractal",
                   help="graph source: procedural fractal or latent kNN")
   ap.add_argument("--latent", type=str, default=None, help="latent CSV with ax,ay (for --mode latent)")
   ap.add_argument("--nodes", type=int, default=600, help="number of base points (fractal mode)")
   ap.add_argument("--k", type=int, default=6, help="k-NN degree (fractal or latent)")
   ap.add_argument("--p_branch", type=float, default=0.25, help="edge-branch prob to mimic fractal refinement")
   ap.add_argument("--rewire", type=float, default=0.00, help="small-world rewiring fraction")
   ap.add_argument("--runs", type=int, default=40, help="number of Monte Carlo realisations")
   ap.add_argument("--seed", type=int, default=42, help="random seed base")
   ap.add_argument("--method", choices=["eigen","hutch"], default="hutch", help="Theta trace method")
   ap.add_argument("--k_max", type=int, default=400, help="if method=eigen, number of smallest eigenvalues to use")
   ap.add_argument("--hutch_m", type=int, default=64, help="if method=hutch, number of Hutchinson probe vectors")
   ap.add_argument("--tmin_exp", type=float, default=-4.0, help="log10 t_min")
   ap.add_argument("--tmax_exp", type=float, default=3.0, help="log10 t_max")
   ap.add_argument("--nt", type=int, default=400, help="number of t samples")
   ap.add_argument("--split1", type=float, default=-3.0, help="log10 t for UV/transition marker")
   ap.add_argument("--split2", type=float, default=0.0, help="log10 t for transition/IR marker")
   ap.add_argument("--split3", type=float, default=1.0, help="log10 t for IR marker")
   ap.add_argument("--outdir", type=str, default="hk_out", help="output directory")
   return ap.parse_args(argv)


if __name__ == "__main__":
   args = parse_args()
   run_ensemble(args)