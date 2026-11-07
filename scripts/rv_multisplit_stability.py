"""Repeated-split stability check: is the ~+1.4pp effect real but underpowered,
or is it an artifact of one lucky/unlucky split seed?"""
import json, sys, torch
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
from revua_cir.significance import mcnemar_exact, split_indices
import rv_cache_analysis as rca

recs=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8") if l.strip()]
for r in recs: r["effective_rv"]=rca._effective_rv_signal(r).tolist()
lams=(0.05,0.10,0.20,0.30,0.50,0.70,1.00,1.50,2.00,3.00,5.00,8.00)
ks=(1,)
rows=[]
for seed in range(20):
    ti,ri=split_indices(len(recs),0.5,seed=seed)
    T=[{**recs[i],"verification_score":recs[i]["effective_rv"]} for i in ti]
    R=[{**recs[i],"verification_score":recs[i]["effective_rv"]} for i in ri]
    best=max(lams,key=lambda l: rca._evaluate_scoring_fn(T,rca._bounded_additive_score_fn(l),ks)["recall"][1])
    res=rca._evaluate_scoring_fn(R,rca._bounded_additive_score_fn(best),ks)
    base=100.0*sum(res["base_hits"])/len(res["base_hits"])
    new=100.0*sum(res["new_hits"])/len(res["new_hits"])
    p=mcnemar_exact(res["help_count"],res["harm_count"])
    rows.append((seed,best,new-base,p))
    print(f"seed={seed:2d} lambda={best:4.2f} delta={new-base:+.2f}pp p={p:.4f} {'SIG' if p<0.05 else ''}")
d=[r[2] for r in rows]; ps=[r[3] for r in rows]
print(f"\ndelta: mean={sum(d)/len(d):+.2f}pp min={min(d):+.2f} max={max(d):+.2f}")
print(f"all positive: {all(x>0 for x in d)}")
print(f"significant in {sum(1 for x in ps if x<0.05)}/20 splits")
print(f"selected lambda distribution: {sorted(set(r[1] for r in rows))}")
