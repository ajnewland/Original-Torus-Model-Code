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

def ols_fit(xs, ys):
    n=len(xs)
    mx=sum(xs)/n; my=sum(ys)/n
    sxx=sum((x-mx)**2 for x in xs)
    sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    if sxx==0: return 0.0
    a=sxy/sxx; b=my-a*mx
    yhat=[a*x+b for x in xs]
    ssr=sum((yh-my)**2 for yh in yhat)
    sst=sum((y-my)**2 for y in ys)
    return ssr/sst if sst>0 else 0.0

def composite_score(by_name):
    sector_rhos=[]
    for sec, names in SECTORS.items():
        avail=[n for n in names if n in by_name]
        if len(avail)<3: continue
        xs=[by_name[n]["z"] for n in avail]
        ys=[by_name[n]["logm"] for n in avail]
        sector_rhos.append(spearman(xs,ys))
    # global R2 on all present
    xs_all=[v["z"] for v in by_name.values()]
    ys_all=[v["logm"] for v in by_name.values()]
    r2=ols_fit(xs_all, ys_all) if len(by_name)>=3 else 0.0
    if not sector_rhos:
        return r2
    return sum(sector_rhos)/len(sector_rhos) + r2

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--nperm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    args=ap.parse_args()
    rng=random.Random(args.seed)

    rows=read_locked(args.csv)
    by_name={r["species"]:r for r in rows}

    # Real score
    s_real=composite_score(by_name)

    # Prepare sector pools
    sector_pools={}
    for sec,names in SECTORS.items():
        avail=[n for n in names if n in by_name]
        if len(avail)>=3:
            sector_pools[sec]=avail

    # Build arrays for shuffling
    perm_scores=[]
    for _ in range(args.nperm):
        by_perm={k:v.copy() for k,v in by_name.items()}
        for sec, avail in sector_pools.items():
            logs=[by_name[n]["logm"] for n in avail]
            rng.shuffle(logs)
            for n,val in zip(avail, logs):
                by_perm[n]["logm"]=val
        perm_scores.append(composite_score(by_perm))

    # p-value
    ge=sum(1 for s in perm_scores if s>=s_real)
    p=(1+ge)/(1+args.nperm)

    # Write outputs
    with open(args.outdir+"/permutation_scores.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["type","score"])
        w.writerow(["real", s_real])
        for s in perm_scores:
            w.writerow(["perm", s])
    with open(args.outdir+"/permutation_summary.txt","w") as f:
        f.write(f"real_score,{s_real}\n")
        f.write(f"n_perm,{args.nperm}\n")
        f.write(f"p_value,{p}\n")
    print("[DONE] permutation_scores.csv, permutation_summary.txt")

if __name__=="__main__":
    main()