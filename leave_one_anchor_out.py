import argparse, csv, math, random
from collections import defaultdict
import statistics as stats

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

SECTORS={
    "leptons":["e","mu","tau"],
    "up":["u","c","t"],
    "down":["d","s","b"],
    "neutrinos":["nu1","nu2","nu3"],
    "bosons":["W","Z","H"],
}

def ols_fit(xs, ys):
    n=len(xs)
    mx=sum(xs)/n; my=sum(ys)/n
    sxx=sum((x-mx)**2 for x in xs)
    sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    if sxx==0: return 0.0, my, 0.0
    a=sxy/sxx; b=my-a*mx
    # R^2
    yhat=[a*x+b for x in xs]
    ssr=sum((yh-my)**2 for yh in yhat)
    sst=sum((y-my)**2 for y in ys)
    r2= ssr/sst if sst>0 else 0.0
    return a,b,r2

def bootstrap_pred_err(xs, ys, x0, y0, B=1000, rng=random.Random(123)):
    a,b,r2=ols_fit(xs,ys)
    yhat=a*x0+b
    resid=[y-(a*x+b) for x,y in zip(xs,ys)]
    if not resid: return yhat, abs(yhat-y0), 0,0
    errs=[]
    for _ in range(B):
        rbs=[rng.choice(resid) for __ in range(len(resid))]
        yhat_bs=(a*x0+b)+stats.mean(rbs)
        errs.append(abs(yhat_bs-y0))
    errs.sort()
    lo=errs[int(0.025*B)]
    hi=errs[int(0.975*B)]
    return yhat, abs(yhat-y0), lo, hi

def spearman(xs, ys):
    def rank(vs):
        # average ranks for ties
        pairs=sorted((v,i) for i,v in enumerate(vs))
        r=[0]*len(vs)
        i=0
        while i<len(vs):
            j=i
            while j+1<len(vs) and pairs[j+1][0]==pairs[i][0]:
                j+=1
            rr= (i+j)/2+1
            for k in range(i,j+1):
                r[pairs[k][1]]=rr
            i=j+1
        return r
    rx=rank(xs); ry=rank(ys)
    mx=sum(rx)/len(rx); my=sum(ry)/len(ry)
    num=sum((rx[i]-mx)*(ry[i]-my) for i in range(len(rx)))
    den=(sum((rx[i]-mx)**2 for i in range(len(rx))) * sum((ry[i]-my)**2 for i in range(len(ry))))**0.5
    return num/den if den>0 else 0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="all_particles_locked.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    args=ap.parse_args()

    rows=read_locked(args.csv)
    by_name={r["species"]:r for r in rows}

    with open(args.outdir+"/leave_one_anchor_out_summary.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["sector","withheld","z","logm_true","logm_pred","abs_err","rel_err","abs_err_lo","abs_err_hi","spearman_sector","r2_sector"])
        for sec, names in SECTORS.items():
            avail=[n for n in names if n in by_name]
            if len(avail)<3:
                continue
            xs=[by_name[n]["z"] for n in avail]
            ys=[by_name[n]["logm"] for n in avail]
            rho=spearman(xs,ys)
            a,b,r2=ols_fit(xs,ys)
            for hold in avail:
                fit_names=[n for n in avail if n!=hold]
                xfit=[by_name[n]["z"] for n in fit_names]
                yfit=[by_name[n]["logm"] for n in fit_names]
                x0=by_name[hold]["z"]; y0=by_name[hold]["logm"]
                yhat, ae, lo, hi=bootstrap_pred_err(xfit,yfit,x0,y0,B=args.boot)
                rel=abs((math.exp(yhat)-math.exp(y0))/math.exp(y0))
                w.writerow([sec, hold, x0, y0, yhat, ae, rel, lo, hi, rho, r2])

    print("[DONE] leave_one_anchor_out_summary.csv")

if __name__=="__main__":
    main()