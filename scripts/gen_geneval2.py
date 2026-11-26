"""Does Dphi transfer to colours it never saw, or did it memorise ten cells?

Three cells whose colour appeared nowhere in training or in the pool (orange, purple) against
three it was trained on. Same sampler, same held-out seeds, count absent from the prompt, every
image counted twice. If the seen cells work and the unseen ones fall to the baseline, Dphi
memorised; if both work, it learned a mapping over c_bar.
"""
import json, os, sys, time
import numpy as np, torch
WT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/.claude/worktrees/cs485-gpu-setup-5bb57e"
sys.path.insert(0, WT); os.chdir(WT)
from countsampler.data.prompts import PromptSampler
from countsampler.evaluation import build_hard_counter
from countsampler.evaluation.hard_counters import count_images
from countsampler.generator import FrozenTurboGenerator
from countsampler.novel.shared import build_sampler
from countsampler.utils.config import load_config
from countsampler.utils.misc import resolve_device, seed_everything, to_uint8_images
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_condition import condition
FLAT = 'data.templates=["a flat lay photo of {count} {color} {shape} on a white background"]'
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/general"
T = list(range(10, 30, 2)); SEEDS = 8
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "rich"
EVAL = [(("red","cross"),"seen"), (("green","star"),"seen"), (("blue","diamond"),"seen"),
        (("orange","cross"),"UNSEEN colour"), (("purple","cross"),"UNSEEN colour"),
        (("orange","triangle"),"UNSEEN colour")]
cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"], [FLAT, "data.include_count=false"])
seed_everything(int(cfg.experiment.seed))
dev = resolve_device(cfg.experiment.get("device", "auto"))
gen = FrozenTurboGenerator.from_config(cfg.generator, dev)
owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
col = build_hard_counter("color")
ps = PromptSampler(cfg.data)
c, h, w = gen.latent_shape
ck = torch.load(f"{OUT}/sampler_{VARIANT}.pt", weights_only=False)
cb0 = condition(gen.encode_prompts([ps.prompt(1, *ck["cells"][0])[0]]), VARIANT)[0]
net = build_sampler(ck["scfg"], c, cb0.shape[0]).to(dev); net.load_state_dict(ck["state"]); net.eval()
eps = torch.randn(SEEDS, c, h, w, generator=torch.Generator().manual_seed(90210))
rows, t0 = [], time.perf_counter()
for (color, shape), grp in EVAL:
    p = ps.prompt(1, color, shape)[0]
    cb = condition(gen.encode_prompts([p]), VARIANT)[0]
    for arm in ("baseline", "sampler"):
        pairs = [(s, t) for s in range(SEEDS) for t in T]
        oc, cc = [], []
        for i in range(0, len(pairs), 8):
            pb = pairs[i:i+8]
            e = torch.stack([eps[s] for s, _ in pb]).to(dev)
            if arm == "baseline":
                z = e
            else:
                with torch.no_grad():
                    z = net(e, torch.tensor([t for _, t in pb], device=dev), cb[None].expand(len(pb), -1))
            with torch.no_grad():
                u8 = to_uint8_images(gen.decode(gen.generate_latents(z, gen.encode_prompts([p]*len(pb)))))
            oc += [r.count for r in count_images(owl, list(u8), color, shape)]
            cc += [col.count(u8[k], color, shape).count for k in range(len(u8))]
        for k, (s, t) in enumerate(pairs):
            rows.append(dict(color=color, shape=shape, group=grp, arm=arm, seed=s, target=t,
                             owl=float(oc[k]), col=float(cc[k])))
    print(f"  {color}/{shape} ({grp})  {(time.perf_counter()-t0)/60:.1f} min", flush=True)
json.dump(rows, open(f"{OUT}/eval_{VARIANT}.json","w"), indent=1)
bar = min(np.mean([abs(t-k) for t in T]) for k in range(5, 40))
rng = np.random.default_rng(0)
print(f"\n===== TRANSFER TO UNSEEN COLOURS. count absent from the prompt, held-out seeds =====")
print(f"  best constant predictor MAE {bar:.2f}\n")
print(f"  {'cell':>17} {'group':>14} {'base':>6} {'Dphi':>6} {'skill':>7} {'corr':>6} {'slope':>6} {'improve':>8} {'95% CI':>16}")
def stats(sel):
    b = {(x["color"],x["shape"],x["seed"],x["target"]): abs(x["owl"]-x["target"]) for x in sel if x["arm"]=="baseline"}
    s = {(x["color"],x["shape"],x["seed"],x["target"]): abs(x["owl"]-x["target"]) for x in sel if x["arm"]=="sampler"}
    ks = sorted(b); bb = np.array([b[k] for k in ks]); ss = np.array([s[k] for k in ks])
    boot = np.array([(bb[i]-ss[i]).mean() for i in (rng.integers(0,len(ks),len(ks)) for _ in range(5000))])
    mu = np.array([np.mean([x["owl"] for x in sel if x["arm"]=="sampler" and x["target"]==t]) for t in T])
    cr = np.corrcoef(T, mu)[0,1] if mu.std() > 1e-9 else float("nan")
    return bb.mean(), ss.mean(), cr, np.polyfit(T,mu,1)[0], (bb-ss).mean(), np.percentile(boot,[2.5,97.5]), (boot<=0).mean()
for (color, shape), grp in EVAL:
    sel = [x for x in rows if x["color"]==color and x["shape"]==shape]
    b,s,cr,sl,imp,ci,pv = stats(sel)
    print(f"  {color+'/'+shape:>17} {grp:>14} {b:6.2f} {s:6.2f} {(bar-s)/bar:+7.3f} {cr:+6.2f} {sl:6.2f} {imp:+8.2f} [{ci[0]:+5.2f},{ci[1]:+5.2f}]")
print()
for grp in ("seen", "UNSEEN colour"):
    sel = [x for x in rows if x["group"]==grp]
    b,s,cr,sl,imp,ci,pv = stats(sel)
    print(f"  {grp:>14} pooled: baseline {b:.2f} -> Dphi {s:.2f}  skill {(bar-s)/bar:+.3f}  "
          f"improvement {imp:+.2f} 95% CI [{ci[0]:+.2f},{ci[1]:+.2f}]  p {pv:.4f}")
    mu = [np.mean([x["owl"] for x in sel if x["arm"]=="sampler" and x["target"]==t]) for t in T]
    print(f"  {'':>14}  per-target mean: " + " ".join(f"{m:.0f}" for m in mu) +
          f"   ({len(set(np.round(mu).astype(int)))}/10 distinct, {sum(mu[i+1]>=mu[i] for i in range(9))}/9 monotone)")
