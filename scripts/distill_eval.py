"""The learned sampler, scored. Count absent from the prompt, seeds held out from fitting."""
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
FLAT = 'data.templates=["a flat lay photo of {count} {color} {shape} on a white background"]'
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/distill"
T = list(range(10, 30, 2)); SEEDS = 8
cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"], [FLAT, "data.include_count=false"])
seed_everything(int(cfg.experiment.seed))
dev = resolve_device(cfg.experiment.get("device", "auto"))
gen = FrozenTurboGenerator.from_config(cfg.generator, dev)
owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
col = build_hard_counter("color")
ps = PromptSampler(cfg.data)
c, h, w = gen.latent_shape
ck = torch.load(f"{OUT}/sampler.pt", weights_only=False)
cb0 = gen.encode_prompts([ps.prompt(1, *ck["cells"][0])[0]]).pooled[0].float()
net = build_sampler(ck["scfg"], c, cb0.shape[0]).to(dev); net.load_state_dict(ck["state"]); net.eval()
# seeds never used in distillation (that drew from generator 7)
eps = torch.randn(SEEDS, c, h, w, generator=torch.Generator().manual_seed(90210))
rows, t0 = [], time.perf_counter()
for color, shape in ck["cells"]:
    p = ps.prompt(1, color, shape)[0]
    cb = gen.encode_prompts([p]).pooled[0].float()
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
                    z = net(e, torch.tensor([t for _, t in pb], device=dev),
                            cb[None].expand(len(pb), -1))
            with torch.no_grad():
                u8 = to_uint8_images(gen.decode(gen.generate_latents(z, gen.encode_prompts([p]*len(pb)))))
            oc += [r.count for r in count_images(owl, list(u8), color, shape)]
            cc += [col.count(u8[k], color, shape).count for k in range(len(u8))]
        for k, (s, t) in enumerate(pairs):
            rows.append(dict(color=color, shape=shape, arm=arm, seed=s, target=t,
                             owl=float(oc[k]), col=float(cc[k])))
    print(f"  {color} {shape}  {(time.perf_counter()-t0)/60:.1f} min", flush=True)
json.dump(rows, open(f"{OUT}/eval.json", "w"), indent=1)
bar = min(np.mean([abs(t-k) for t in T]) for k in range(5, 40))
print(f"\n====== LEARNED SAMPLER, COUNT ABSENT FROM THE PROMPT, HELD-OUT SEEDS ======")
print(f"  best constant predictor MAE {bar:.2f}\n")
print(f"  {'arm':>9} {'MAE':>6} {'skill':>7} {'corr':>6} {'slope':>6} {'sd':>5} {'corr(owl,col)':>14}")
for arm in ("baseline", "sampler"):
    r = [x for x in rows if x["arm"] == arm]
    mae = np.mean([abs(x["owl"]-x["target"]) for x in r])
    mu = np.array([np.mean([x["owl"] for x in r if x["target"]==t]) for t in T])
    sd = np.mean([np.std([x["owl"] for x in r if x["target"]==t]) for t in T])
    cr = np.corrcoef(T, mu)[0,1] if mu.std() > 1e-9 else float("nan")
    oc = np.corrcoef([x["owl"] for x in r], [x["col"] for x in r])[0,1]
    print(f"  {arm:>9} {mae:6.2f} {(bar-mae)/bar:+7.3f} {cr:+6.2f} {np.polyfit(T,mu,1)[0]:6.2f} {sd:5.2f} {oc:+14.2f}")
print(f"\n  per-target mean counted")
print(f"  {'arm':>9} " + " ".join(f"{t:>5}" for t in T))
for arm in ("baseline", "sampler"):
    r = [x for x in rows if x["arm"]==arm]
    print(f"  {arm:>9} " + " ".join(f"{np.mean([x['owl'] for x in r if x['target']==t]):5.1f}" for t in T))
# paired bootstrap
rng = np.random.default_rng(0)
key = lambda x: (x["color"], x["shape"], x["seed"], x["target"])
b = {key(x): abs(x["owl"]-x["target"]) for x in rows if x["arm"]=="baseline"}
s = {key(x): abs(x["owl"]-x["target"]) for x in rows if x["arm"]=="sampler"}
ks = sorted(b); bb = np.array([b[k] for k in ks]); ss = np.array([s[k] for k in ks])
boot = np.array([(bb[i]-ss[i]).mean() for i in (rng.integers(0,len(ks),len(ks)) for _ in range(10000))])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"\n  paired over {len(ks)} units: improvement {(bb-ss).mean():+.2f}  "
      f"95% CI [{lo:+.2f}, {hi:+.2f}]  p(no improvement) {(boot<=0).mean():.4f}")
mu = [np.mean([x["owl"] for x in rows if x["arm"]=="sampler" and x["target"]==t]) for t in T]
print(f"  degeneracy guard: {len(set(np.round(mu).astype(int)))}/10 distinct, "
      f"{sum(mu[i+1]>=mu[i] for i in range(9))}/9 monotone")
