"""The method as it actually stands: one Dphi calibrated per cell, scored across four cells.

Transfer to unseen cells was measured and does not happen, and spreading one Dphi over ten cells
diluted every one of them (+0.028 pooled against the three-cell model's +0.272). So the honest
form of this method is a per-cell calibration: for a given colour/shape, count a pool of noises
once, then regress Dphi onto the references. This measures what that procedure delivers, cell by
cell, including its cost.

Each cell is scored on eight seeds that were never used in its training, with the count absent
from the prompt - every target gets the identical text - and every image counted by OWLv2 and by
the independent colour counter.
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
FLAT = 'data.templates=["a flat lay photo of {count} {color} {shape} on a white background"]'
GEN = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/general"
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/percal"
T = list(range(10, 30, 2)); BL = 0.75; SEEDS = 8; STEPS = 4000
CELLS = [("red","cross"), ("blue","cross"), ("green","star"), ("red","diamond")]
os.makedirs(OUT, exist_ok=True)

def slerp_b(a, b, t):
    af, bf = a.flatten(1).float(), b.flatten(1).float()
    an, bn = af/af.norm(dim=1,keepdim=True), bf/bf.norm(dim=1,keepdim=True)
    om = torch.acos((an*bn).sum(1,keepdim=True).clamp(-1+1e-7,1-1e-7)); s = torch.sin(om)
    return ((torch.sin((1-t)*om)/s)*af + (torch.sin(t*om)/s)*bf).view_as(a)

cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"], [FLAT, "data.include_count=false"])
seed_everything(int(cfg.experiment.seed))
dev = resolve_device(cfg.experiment.get("device", "auto"))
gen = FrozenTurboGenerator.from_config(cfg.generator, dev)
owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
col = build_hard_counter("color")
ps = PromptSampler(cfg.data)
c, h, w = gen.latent_shape
pool = torch.load(f"{GEN}/pool.pt", weights_only=False)["pool"]
refs = json.load(open(f"{GEN}/refs.json"))
eps_ev = torch.randn(SEEDS, c, h, w, generator=torch.Generator().manual_seed(90210))
scfg = dict(cfg.sampler); scfg["norm_preserve"] = True
rows, cost = [], {}
for color, shape in CELLS:
    t0 = time.perf_counter()
    p = ps.prompt(1, color, shape)[0]
    cb = gen.encode_prompts([p]).pooled[0].float()
    R = torch.stack([pool[refs[f"{color}_{shape}"][str(t)]] for t in T]).to(dev)
    NNd = torch.tensor(T, device=dev)
    net = build_sampler(scfg, c, cb.shape[0]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    g = torch.Generator().manual_seed(7)
    for step in range(STEPS):
        j = torch.randint(0, len(T), (16,), generator=g)
        e = torch.randn(16, c, h, w, generator=g).to(dev)
        loss = torch.nn.functional.mse_loss(net(e, NNd[j], cb[None].expand(16,-1)), slerp_b(e, R[j], BL))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
    train_min = (time.perf_counter()-t0)/60
    net.eval()
    for arm in ("baseline", "sampler"):
        pairs = [(s, t) for s in range(SEEDS) for t in T]
        oc, cc = [], []
        for i in range(0, len(pairs), 8):
            pb = pairs[i:i+8]
            e = torch.stack([eps_ev[s] for s,_ in pb]).to(dev)
            if arm == "sampler":
                with torch.no_grad():
                    z = net(e, torch.tensor([t for _,t in pb], device=dev), cb[None].expand(len(pb),-1))
            else:
                z = e
            with torch.no_grad():
                u8 = to_uint8_images(gen.decode(gen.generate_latents(z, gen.encode_prompts([p]*len(pb)))))
            oc += [r.count for r in count_images(owl, list(u8), color, shape)]
            cc += [col.count(u8[k], color, shape).count for k in range(len(u8))]
        for k,(s,t) in enumerate(pairs):
            rows.append(dict(color=color, shape=shape, arm=arm, seed=s, target=t,
                             owl=float(oc[k]), col=float(cc[k])))
    cost[f"{color}/{shape}"] = train_min
    print(f"  {color}/{shape}: trained {train_min:.1f} min, evaluated  "
          f"({(time.perf_counter()-t0)/60:.1f} min total)", flush=True)
json.dump(rows, open(f"{OUT}/eval.json","w"), indent=1)
bar = min(np.mean([abs(t-k) for t in T]) for k in range(5,40))
rng = np.random.default_rng(0)
def stats(sel):
    b = {(x["color"],x["shape"],x["seed"],x["target"]): abs(x["owl"]-x["target"]) for x in sel if x["arm"]=="baseline"}
    s = {(x["color"],x["shape"],x["seed"],x["target"]): abs(x["owl"]-x["target"]) for x in sel if x["arm"]=="sampler"}
    ks = sorted(b); bb = np.array([b[k] for k in ks]); ss = np.array([s[k] for k in ks])
    boot = np.array([(bb[i]-ss[i]).mean() for i in (rng.integers(0,len(ks),len(ks)) for _ in range(10000))])
    mu = np.array([np.mean([x["owl"] for x in sel if x["arm"]=="sampler" and x["target"]==t]) for t in T])
    o = np.array([x["owl"] for x in sel if x["arm"]=="sampler"]); cl = np.array([x["col"] for x in sel if x["arm"]=="sampler"])
    return (bb.mean(), ss.mean(), np.corrcoef(T,mu)[0,1], np.polyfit(T,mu,1)[0], (bb-ss).mean(),
            np.percentile(boot,[2.5,97.5]), (boot<=0).mean(), mu, np.corrcoef(o,cl)[0,1])
print(f"\n===== PER-CELL CALIBRATED SAMPLER. count absent from the prompt, held-out seeds =====")
print(f"  best constant predictor MAE {bar:.2f}\n")
print(f"  {'cell':>15} {'base':>6} {'Dphi':>6} {'skill':>7} {'corr':>6} {'slope':>6} {'improve':>8} {'95% CI':>15} {'p':>7} {'o/c':>5}")
for color, shape in CELLS:
    b,s,cr,sl,imp,ci,pv,mu,oc = stats([x for x in rows if x["color"]==color and x["shape"]==shape])
    print(f"  {color+'/'+shape:>15} {b:6.2f} {s:6.2f} {(bar-s)/bar:+7.3f} {cr:+6.2f} {sl:6.2f} {imp:+8.2f} "
          f"[{ci[0]:+5.2f},{ci[1]:+5.2f}] {pv:7.4f} {oc:+5.2f}")
b,s,cr,sl,imp,ci,pv,mu,oc = stats(rows)
print(f"\n  {'POOLED':>15} {b:6.2f} {s:6.2f} {(bar-s)/bar:+7.3f} {cr:+6.2f} {sl:6.2f} {imp:+8.2f} "
      f"[{ci[0]:+5.2f},{ci[1]:+5.2f}] {pv:7.4f} {oc:+5.2f}")
print(f"\n  per-target mean counted, pooled: " + " ".join(f"{m:.0f}" for m in mu))
print(f"  degeneracy guard: {len(set(np.round(mu).astype(int)))}/10 distinct, "
      f"{sum(mu[i+1]>=mu[i] for i in range(9))}/9 monotone")
print(f"\n  cost per cell: {np.mean(list(cost.values())):.1f} min of training plus one 96-noise")
print(f"  counting pass. 'o/c' is OWLv2 against the independent colour counter.")
