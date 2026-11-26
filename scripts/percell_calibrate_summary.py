"""Score the per-cell calibration over every cell that has been calibrated.

Merges runs/percal/eval.json (the first four cells) with eval_more.json (the six the pool
already covered) and reports one table: per cell, then pooled over all of them with a paired
bootstrap whose units are (cell, seed, target).

It also tests, now at n = 10 instead of n = 4, the suspicion recorded in findings.md: that the
measured skill tracks how well the two counters agree on that cell, i.e. that a low number can be
the instrument rather than the method. A correlation here is evidence about the MEASUREMENT, and
is never used to adjust a reported number.
"""
import json, os, sys
import numpy as np
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/percal"
T = list(range(10, 30, 2))
rows = json.load(open(f"{OUT}/eval.json"))
more = f"{OUT}/eval_more.json"
if os.path.exists(more):
    rows += json.load(open(more))
cells = sorted({(x["color"], x["shape"]) for x in rows})
bar = min(np.mean([abs(t-k) for t in T]) for k in range(5, 40))
rng = np.random.default_rng(0)

def stats(sel):
    b = {(x["color"],x["shape"],x["seed"],x["target"]): abs(x["owl"]-x["target"]) for x in sel if x["arm"]=="baseline"}
    s = {(x["color"],x["shape"],x["seed"],x["target"]): abs(x["owl"]-x["target"]) for x in sel if x["arm"]=="sampler"}
    ks = sorted(b); bb = np.array([b[k] for k in ks]); ss = np.array([s[k] for k in ks])
    boot = np.array([(bb[i]-ss[i]).mean() for i in (rng.integers(0,len(ks),len(ks)) for _ in range(10000))])
    mu = np.array([np.mean([x["owl"] for x in sel if x["arm"]=="sampler" and x["target"]==t]) for t in T])
    o = np.array([x["owl"] for x in sel if x["arm"]=="sampler"])
    cl = np.array([x["col"] for x in sel if x["arm"]=="sampler"])
    agree = float(np.mean(np.abs(o-cl)))
    return dict(base=bb.mean(), dphi=ss.mean(), skill=(bar-ss.mean())/bar,
                corr=np.corrcoef(T,mu)[0,1], slope=np.polyfit(T,mu,1)[0], imp=(bb-ss).mean(),
                ci=np.percentile(boot,[2.5,97.5]), p=(boot<=0).mean(), mu=mu,
                oc=np.corrcoef(o,cl)[0,1], agree=agree)

print(f"\n===== PER-CELL CALIBRATED SAMPLER, {len(cells)} CELLS =====")
print("  count absent from the prompt, 8 held-out seeds, every image double-counted")
print(f"  best constant predictor MAE {bar:.2f}; skill 0.000 means no better than ignoring n\n")
hdr = f"  {'cell':>15} {'base':>6} {'Dphi':>6} {'skill':>7} {'corr':>6} {'slope':>6} {'improve':>8} {'95% CI':>15} {'p':>7} {'o/c':>5} {'|o-c|':>6}"
print(hdr)
per = {}
for color, shape in cells:
    r = stats([x for x in rows if x["color"]==color and x["shape"]==shape]); per[(color,shape)] = r
    print(f"  {color+'/'+shape:>15} {r['base']:6.2f} {r['dphi']:6.2f} {r['skill']:+7.3f} {r['corr']:+6.2f} "
          f"{r['slope']:6.2f} {r['imp']:+8.2f} [{r['ci'][0]:+5.2f},{r['ci'][1]:+5.2f}] {r['p']:7.4f} "
          f"{r['oc']:+5.2f} {r['agree']:6.1f}")
g = stats(rows)
print(f"\n  {'POOLED':>15} {g['base']:6.2f} {g['dphi']:6.2f} {g['skill']:+7.3f} {g['corr']:+6.2f} "
      f"{g['slope']:6.2f} {g['imp']:+8.2f} [{g['ci'][0]:+5.2f},{g['ci'][1]:+5.2f}] {g['p']:7.4f} "
      f"{g['oc']:+5.2f} {g['agree']:6.1f}")
print(f"\n  per-target mean counted, pooled: " + " ".join(f"{m:.0f}" for m in g["mu"]))
print(f"  degeneracy guard: {len(set(np.round(g['mu']).astype(int)))}/10 distinct, "
      f"{sum(g['mu'][i+1]>=g['mu'][i] for i in range(9))}/9 monotone")
pos = sum(1 for r in per.values() if r["skill"] > 0)
sig = sum(1 for r in per.values() if r["p"] < 0.05 and r["imp"] > 0)
print(f"\n  {pos}/{len(cells)} cells beat the constant predictor; "
      f"{sig}/{len(cells)} improve on untouched noise at p < 0.05")
sk = np.array([per[c]["skill"] for c in cells]); ag = np.array([per[c]["agree"] for c in cells])
oc = np.array([per[c]["oc"] for c in cells])
print(f"\n  measured skill vs counter DISAGREEMENT |OWLv2-colour|: r = {np.corrcoef(sk,ag)[0,1]:+.2f} (n={len(cells)})")
print(f"  measured skill vs counter correlation:                 r = {np.corrcoef(sk,oc)[0,1]:+.2f}")
print("  This is evidence about the instrument, not a correction: no reported number uses it.")
json.dump({f"{c}/{s}": {k: (v.tolist() if isinstance(v,np.ndarray) else float(v))
                        for k,v in per[(c,s)].items()} for c,s in cells},
          open(f"{OUT}/summary.json","w"), indent=1)
