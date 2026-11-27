"""Score the blend-0.85 calibration: skill, significance, and the diversity it costs.

Units for the paired bootstrap are (cell, seed, target), the sampler against untouched noise on
the same seed and target. Diversity is the across-seed pixel sd at a fixed target, reported as a
fraction of the baseline arm's, so the price of the blend is visible beside every skill number.
"""
import json, os, sys
import numpy as np
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/percal2"
T = list(range(10, 30, 2))
rows = json.load(open(f"{OUT}/eval.json"))
cells = sorted({(x["color"], x["shape"]) for x in rows})
blends = sorted({x["blend"] for x in rows if x["arm"] == "sampler"})
bar = min(np.mean([abs(t-k) for t in T]) for k in range(5, 40))
rng = np.random.default_rng(0)

def stats(sel_b, sel_s):
    key = lambda x: (x["color"],x["shape"],x["seed"],x["target"])
    b = {key(x): abs(x["owl"]-x["target"]) for x in sel_b}
    s = {key(x): abs(x["owl"]-x["target"]) for x in sel_s}
    ks = sorted(set(b) & set(s)); bb = np.array([b[k] for k in ks]); ss = np.array([s[k] for k in ks])
    boot = np.array([(bb[i]-ss[i]).mean() for i in (rng.integers(0,len(ks),len(ks)) for _ in range(10000))])
    mu = np.array([np.mean([x["owl"] for x in sel_s if x["target"]==t]) for t in T])
    o = np.array([x["owl"] for x in sel_s]); cl = np.array([x["col"] for x in sel_s])
    dv = np.mean([x["div"] for x in sel_s]); db = np.mean([x["div"] for x in sel_b])
    return dict(base=bb.mean(), dphi=ss.mean(), skill=(bar-ss.mean())/bar,
                corr=np.corrcoef(T,mu)[0,1], slope=np.polyfit(T,mu,1)[0], imp=(bb-ss).mean(),
                ci=np.percentile(boot,[2.5,97.5]), p=(boot<=0).mean(), mu=mu,
                oc=np.corrcoef(o,cl)[0,1], div=dv/db)

print(f"\n===== PER-CELL CALIBRATION, BLEND SWEPT, FRESH SEED BANK =====")
print("  count absent from the prompt; 8 seeds used neither in training nor in choosing the blend")
print(f"  best constant predictor MAE {bar:.2f}\n")
for bl in blends:
    print(f"  --- blend {bl:.2f} ---")
    print(f"  {'cell':>15} {'base':>6} {'Dphi':>6} {'skill':>7} {'corr':>6} {'slope':>6} {'improve':>8} "
          f"{'95% CI':>15} {'p':>7} {'div':>5}")
    per = {}
    for color, shape in cells:
        sb = [x for x in rows if x["color"]==color and x["shape"]==shape and x["arm"]=="baseline"]
        ss = [x for x in rows if x["color"]==color and x["shape"]==shape and x["arm"]=="sampler" and x["blend"]==bl]
        r = stats(sb, ss); per[(color,shape)] = r
        print(f"  {color+'/'+shape:>15} {r['base']:6.2f} {r['dphi']:6.2f} {r['skill']:+7.3f} {r['corr']:+6.2f} "
              f"{r['slope']:6.2f} {r['imp']:+8.2f} [{r['ci'][0]:+5.2f},{r['ci'][1]:+5.2f}] {r['p']:7.4f} {r['div']:5.2f}")
    g = stats([x for x in rows if x["arm"]=="baseline"],
              [x for x in rows if x["arm"]=="sampler" and x["blend"]==bl])
    print(f"\n  {'POOLED':>15} {g['base']:6.2f} {g['dphi']:6.2f} {g['skill']:+7.3f} {g['corr']:+6.2f} "
          f"{g['slope']:6.2f} {g['imp']:+8.2f} [{g['ci'][0]:+5.2f},{g['ci'][1]:+5.2f}] {g['p']:7.4f} {g['div']:5.2f}")
    print(f"  per-target mean counted: " + " ".join(f"{m:.0f}" for m in g["mu"]))
    print(f"  degeneracy guard: {len(set(np.round(g['mu']).astype(int)))}/10 distinct, "
          f"{sum(g['mu'][i+1]>=g['mu'][i] for i in range(9))}/9 monotone")
    pos = sum(1 for r in per.values() if r["skill"] > 0)
    sig = sum(1 for r in per.values() if r["p"] < 0.05 and r["imp"] > 0)
    print(f"  {pos}/{len(cells)} cells beat the constant predictor; {sig}/{len(cells)} beat raw noise at p<0.05\n")
    json.dump({f"{c}/{s}": {k:(v.tolist() if isinstance(v,np.ndarray) else float(v))
                            for k,v in per[(c,s)].items()} for c,s in cells},
              open(f"{OUT}/summary_bl{int(bl*100)}.json","w"), indent=1)
print("  'div' is across-seed image diversity as a fraction of untouched noise. Counting accuracy")
print("  bought by driving this toward zero is the degenerate solution, not a result.")
