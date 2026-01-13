#!/usr/bin/env bash
set -euxo pipefail
: "${HK_NODES:=2400}"
: "${HK_K:=6}"
: "${HK_PBRANCH:=0.28}"
: "${HK_REWIRE:=0.08}"
: "${HK_RUNS:=64}"
: "${HK_HUTCH_M:=256}"
: "${HK_TMIN_EXP:=-6}"
: "${HK_TMAX_EXP:=3}"
: "${HK_NT:=600}"
: "${HK_OUTDIR:=/var/hk_runs/out}"
: "${HK_SEED:=12345}"
: "${HK_GIANT_MIN:=0.98}"
: "${HK_CLUSTER_MIN:=0.02}"
: "${HK_CLUSTER_MAX:=0.40}"
: "${HK_GCS_DST:=}"

apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3 python3-venv python3-pip build-essential git screen

python3 -m venv /opt/hkenv
source /opt/hkenv/bin/activate
pip install --upgrade pip wheel
pip install numpy scipy networkx tqdm pandas

mkdir -p /var/hk_runs/src "$HK_OUTDIR"
chmod -R 777 /var/hk_runs

cat >/var/hk_runs/src/heat_kernel_cloudmc.py <<'PY'
import argparse, os
import numpy as np, pandas as pd, networkx as nx
import scipy.sparse as sp, scipy.sparse.linalg as spla
from pathlib import Path
from tqdm import tqdm

def make_fractal_graph(n, k, p_branch, rewire, seed):
    rng = np.random.default_rng(seed)
    G = nx.watts_strogatz_graph(n, k, rewire, seed=int(rng.integers(1<<31)))
    to_add=[]
    for u in G.nodes():
        if rng.random()<p_branch:
            v=len(G)+len(to_add)
            to_add.append((u,v))
    G.add_edges_from(to_add)
    return G

def realism_filters(G, giant_min=0.98, cmin=0.02, cmax=0.40):
    if len(G)==0: return False
    comps=sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    giant_frac=comps[0]/len(G)
    if giant_frac<giant_min: return False
    c=nx.average_clustering(G)
    return (cmin<=c<=cmax)

def laplacian_sparse(G):
    n=G.number_of_nodes()
    idx={u:i for i,u in enumerate(G.nodes())}
    rows,cols=[],[]
    for u,v in G.edges():
        i,j=idx[u],idx[v]
        rows+=[i,j]; cols+=[j,i]
    A=sp.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)).tocsr()
    deg=np.asarray(A.sum(axis=1)).ravel()
    L=sp.diags(deg)-A
    return L

def hutch_trace_expm(L,t,m,rng):
    fLm=lambda v: spla.expm_multiply((-t)*L, v)
    n=L.shape[0]; acc=0.0
    for _ in range(m):
        v=rng.choice([-1.0,1.0],size=n).astype(np.float64)
        acc += np.dot(v, fLm(v))
    return acc/m

def logspace(tmin_exp,tmax_exp,nt):
    return np.logspace(tmin_exp,tmax_exp,nt)

def save_row(path,d):
    p=Path(path); new=not p.exists()
    pd.DataFrame([d]).to_csv(p,mode='a',index=False,header=new)

def gsutil_copy(src,dst):
    if not dst: return
    os.system(f'gsutil -q cp -n {src} {dst}/')

def run(args):
    rng=np.random.default_rng(args.seed)
    t_vals=logspace(args.tmin_exp,args.tmax_exp,args.nt)
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    meta=out/'ensemble_meta.csv'
    theta_part=out/'theta_partial.csv'
    theta_mean_path=out/'theta_mean.csv'
    ds_path=out/'ds_mean.csv'

    done=set()
    if meta.exists():
        try: done=set(pd.read_csv(meta)['run_id'].astype(int).tolist())
        except: pass

    Theta_acc=np.zeros_like(t_vals); Theta_cnt=0
    if theta_part.exists():
        tp=pd.read_csv(theta_part)
        if not tp.empty:
            for rid,sub in tp.groupby('run_id'):
                Theta_acc+=np.array(sub.sort_values('t')['theta'])
                Theta_cnt+=1

    from tqdm import tqdm
    pbar=tqdm(total=args.runs,desc="Ensemble runs",ncols=100)
    pbar.update(len(done))

    for run_id in range(args.runs):
        if run_id in done: continue
        for _ in range(10):
            G=make_fractal_graph(args.nodes,args.k,args.p_branch,args.rewire,
                                 int(rng.integers(1<<31)))
            if realism_filters(G,args.giant_min,args.cluster_min,args.cluster_max):
                break
        else:
            save_row(meta,{'run_id':run_id,'status':'skipped_no_realistic_graph'})
            pbar.update(1); continue

        L=laplacian_sparse(G)
        rrng=np.random.default_rng(int(rng.integers(1<<31)))
        thetas=[hutch_trace_expm(L,t,args.hutch_m,rrng) for t in t_vals]

        pd.DataFrame({'run_id':run_id,'t':t_vals,'theta':thetas}).to_csv(
            theta_part,mode='a',index=False,header=not theta_part.exists())
        gsutil_copy(str(theta_part), args.gcs_dst)

        Theta_acc+=np.array(thetas); Theta_cnt+=1
        save_row(meta,{
          'run_id':run_id,'status':'ok',
          'n_nodes':G.number_of_nodes(),'n_edges':G.number_of_edges(),
          'giant_frac':max(len(c) for c in nx.connected_components(G))/G.number_of_nodes(),
          'clustering':nx.average_clustering(G)
        })
        gsutil_copy(str(meta), args.gcs_dst)

        theta_mean=Theta_acc/max(1,Theta_cnt)
        pd.DataFrame({'t':t_vals,'theta_mean':theta_mean}).to_csv(theta_mean_path,index=False)
        gsutil_copy(str(theta_mean_path), args.gcs_dst)

        pbar.update(1)

    pbar.close()
    theta_mean=Theta_acc/max(1,Theta_cnt)
    pd.DataFrame({'t':t_vals,'theta_mean':theta_mean}).to_csv(theta_mean_path,index=False)
    gsutil_copy(str(theta_mean_path), args.gcs_dst)

    logt=np.log(t_vals); logTh=np.log(theta_mean+1e-300)
    dlog=np.gradient(logTh,logt); ds=-2.0*dlog
    pd.DataFrame({'t':t_vals,'ds':ds}).to_csv(ds_path,index=False)
    gsutil_copy(str(ds_path), args.gcs_dst)

def main():
    import argparse, os
    ap=argparse.ArgumentParser()
    ap.add_argument('--nodes',type=int,default=int(os.environ.get('HK_NODES',2400)))
    ap.add_argument('--k',type=int,default=int(os.environ.get('HK_K',6)))
    ap.add_argument('--p_branch',type=float,default=float(os.environ.get('HK_PBRANCH',0.28)))
    ap.add_argument('--rewire',type=float,default=float(os.environ.get('HK_REWIRE',0.08)))
    ap.add_argument('--runs',type=int,default=int(os.environ.get('HK_RUNS',64)))
    ap.add_argument('--hutch_m',type=int,default=int(os.environ.get('HK_HUTCH_M',256)))
    ap.add_argument('--tmin_exp',type=float,default=float(os.environ.get('HK_TMIN_EXP',-6)))
    ap.add_argument('--tmax_exp',type=float,default=float(os.environ.get('HK_TMAX_EXP',3)))
    ap.add_argument('--nt',type=int,default=int(os.environ.get('HK_NT',600)))
    ap.add_argument('--outdir',type=str,default=os.environ.get('HK_OUTDIR','/var/hk_runs/out'))
    ap.add_argument('--seed',type=int,default=int(os.environ.get('HK_SEED',12345)))
    ap.add_argument('--giant_min',type=float,default=float(os.environ.get('HK_GIANT_MIN',0.98)))
    ap.add_argument('--cluster_min',type=float,default=float(os.environ.get('HK_CLUSTER_MIN',0.02)))
    ap.add_argument('--cluster_max',type=float,default=float(os.environ.get('HK_CLUSTER_MAX',0.40)))
    ap.add_argument('--gcs_dst',type=str,default=os.environ.get('HK_GCS_DST',''))
    args=ap.parse_args()
    run(args)

if __name__=='__main__': main()
PY

SCREEN_NAME="hkjob"
if ! screen -list | grep -q "$SCREEN_NAME"; then
  screen -dmS "$SCREEN_NAME" bash -lc "
    source /opt/hkenv/bin/activate
    python /var/hk_runs/src/heat_kernel_cloudmc.py \
      --nodes $HK_NODES --k $HK_K --p_branch $HK_PBRANCH --rewire $HK_REWIRE \
      --runs $HK_RUNS --hutch_m $HK_HUTCH_M \
      --tmin_exp $HK_TMIN_EXP --tmax_exp $HK_TMAX_EXP --nt $HK_NT \
      --outdir $HK_OUTDIR --seed $HK_SEED \
      --giant_min $HK_GIANT_MIN --cluster_min $HK_CLUSTER_MIN --cluster_max $HK_CLUSTER_MAX \
      --gcs_dst \"$HK_GCS_DST\"
  "
fi
echo "Startup complete. Attach with: screen -r $SCREEN_NAME"
