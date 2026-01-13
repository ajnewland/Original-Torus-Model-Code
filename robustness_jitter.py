import argparse, csv, math, random
from collections import defaultdict

SECTORS={
    "leptons":["e","mu","tau"],
    "up":["u","c","t"],
    "down":["d","s","b"],
    "neutrinos":["nu1","nu2","nu3"],
    "bosons":["W","Z","H"],
}

def read_locked(p):
    rows=[]
    with open(p, newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            s=row.get("species","").strip()
            try:
                m=float(row["m_GeV"])
                z=float(row.get("z_pred", row.get("z", "nan")))
            except:
                continue
            rows.append({"species":s,"m":m,"logm":math.log(m), "z":z})
    return rows

def spearman(xs, ys):
    def rank(vs):
        pairs=sorted((v,i) for i,v in enumerate(vs))
        r=[0]*len(vs)
        i=0
        while i<len(vs):
            j=i
            while j+1<len(vs) and pairs[j+1][0]==pairs[i][0]:
                j+=1
            rr=(i+j)/2+1
            for k in range(i,j+1):
                r[pairs[k][1]]=rr
            i=j+1
        return r
    rx=rank(xs); ry=rank(ys)
    mx=sum(rx)/len(rx); my=sum(ry)/len(ry)
    num=sum((rx[i]-mx)*(ry[i]-my) for i in range(len(rx)))
    den=(sum((rx[i]-mx)**2 for i in range(len(rx))) * sum((ry[i]-my)**2 for i in range(len(ry))))**0.5
    return num/den if den>0 else 0.0

def sector_order(names, zmap):
    return tuple(sorted(names, key=lambda n: zmap[n]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--R", type=int, default=1000, help="replicates")
    ap.add_argument("--sigma_z", type=float, default=5e-4, help="additive jitter std")
    ap.add_argument("--sigma_affine", type=float, default=1e-4, help="affine jitter std")
    ap.add_argument("--seed", type=int, default=777)
    args=ap.parse_args()
    rng=random.Random(args.seed)

    rows=read_locked(args.csv)
    by_name={r["species"]:r for r in rows}
    base_z={s:r["z"] for s,r in by_name.items()}
    base_orders={}
    for sec,names in SECTORS.items():
        avail=[n for n in names if n in by_name]
        if len(avail)>=3:
            base_orders[sec]=sector_order(avail, base_z)

    def run_block(kind):
        # kind: "add" or "affine"
        per_sec=[]
        order_stable=[]
        for _ in range(args.R):
            zjit={}
            if kind=="add":
                for s,z in base_z.items():
                    eps = rng.gauss(0.0, args.sigma_z)
                    zjit[s]= z + eps
            else:
                delta_s = rng.gauss(0.0, args.sigma_affine)
                delta_o = rng.gauss(0.0, args.sigma_affine)
                for s,z in base_z.items():
                    zjit[s]= (1.0+delta_s)*z + delta_o
            # metrics
            for sec,names in SECTORS.items():
                avail=[n for n in names if n in by_name]
                if len(avail)<3: continue
                xs=[zjit[n] for n in avail]
                ys=[by_name[n]["logm"] for n in avail]
                rho=spearman(xs,ys)
                per_sec.append((kind,sec,rho))
                # ranking stability
                if sec in base_orders:
                    ord_now=sector_order(avail, zjit)
                    order_stable.append((kind,sec, 1 if ord_now==base_orders[sec] else 0))
        return per_sec, order_stable

    add_sec, add_ord = run_block("add")
    aff_sec, aff_ord = run_block("affine")

    # Summaries
    with open(args.outdir+"/robustness_jitter_hist.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["kind","sector","spearman"])
        for row in add_sec+aff_sec: w.writerow(row)
    # Stability
    by_sec={"add":defaultdict(list), "affine":defaultdict(list)}
    for k,sec,val in add_ord: by_sec["add"][sec].append(val)
    for k,sec,val in aff_ord: by_sec["affine"][sec].append(val)

    with open(args.outdir+"/robustness_jitter_summary.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["kind","sector","median_rho","p05_rho","p95_rho","ranking_stability"])
        for kind, bag in [("add",add_sec),("affine",aff_sec)]:
            by=dict()
            for k,sec,rho in bag:
                by.setdefault(sec,[]).append(rho)
            for sec, arr in by.items():
                arr.sort()
                med=arr[len(arr)//2]
                p05=arr[int(0.05*len(arr))]
                p95=arr[int(0.95*len(arr))-1]
                stab = sum(by_sec[kind].get(sec,[]))/max(1,len(by_sec[kind].get(sec,[])))
                w.writerow([kind,sec,med,p05,p95,stab])

    print("[DONE] robustness_jitter_summary.csv, robustness_jitter_hist.csv")

if __name__=="__main__":
    main()