"""How much of the reference does a cell need, and what does that cost in diversity?

offset_diagnose.py settled two things. Dphi reproduces its training target almost exactly
(red/triangle slerp 5.21 vs dphi 5.26, blue/diamond 5.28 vs 5.38), so the regression is not the
gap. And the error is not a bias: mean(counted - target) is -0.24 and -0.75 on those cells, near
zero. The per-target means scatter instead of sitting high.

What the error IS: the count of a reference noise does not survive being blended with an unseen
seed, and how well it survives is cell-specific. Blending 0.75 of the way costs red/cross almost
nothing (MAE 0 -> 3.24) and costs red/triangle everything (0 -> 5.21).

So the lever is the blend, and it is not free: at blend 1.0 every seed collapses to the same ten
images. This sweeps the blend per cell and measures BOTH sides - counting error and across-seed
image diversity, as a fraction of what untouched noise has. A blend that buys accuracy by
collapsing diversity is reported as that trade, not as a win.
"""
import json, os, sys, time
import numpy as np, torch
WT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/.claude/worktrees/cs485-gpu-setup-5bb57e"
sys.path.insert(0, WT); os.chdir(WT)
from countsampler.data.prompts import PromptSampler
from countsampler.evaluation import build_hard_counter
from countsampler.evaluation.hard_counters import count_images
from countsampler.generator import FrozenTurboGenerator
from countsampler.utils.config import load_config
from countsampler.utils.misc import resolve_device, seed_everything, to_uint8_images
FLAT = 'data.templates=["a flat lay photo of {count} {color} {shape} on a white background"]'
GEN = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/general"
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/percal"
T = list(range(10, 30, 2)); SEEDS = 8
BLENDS = [0.0, 0.75, 0.85, 0.95]
CELLS = [("red","triangle"), ("blue","diamond"), ("green","star"), ("red","cross")]

def slerp_b(a,b,t):
    if t == 0.0: return a.clone()
    af,bf=a.flatten(1).float(),b.flatten(1).float()
    an,bn=af/af.norm(dim=1,keepdim=True),bf/bf.norm(dim=1,keepdim=True)
    om=torch.acos((an*bn).sum(1,keepdim=True).clamp(-1+1e-7,1-1e-7)); s=torch.sin(om)
    return ((torch.sin((1-t)*om)/s)*af+(torch.sin(t*om)/s)*bf).view_as(a)

cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"], [FLAT, "data.include_count=false"])
seed_everything(int(cfg.experiment.seed)); dev = resolve_device(cfg.experiment.get("device","auto"))
gen = FrozenTurboGenerator.from_config(cfg.generator, dev)
owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
col = build_hard_counter("color"); ps = PromptSampler(cfg.data); c,h,w = gen.latent_shape
pool = torch.load(f"{GEN}/pool.pt", weights_only=False)["pool"]
refs = json.load(open(f"{GEN}/refs.json"))
eps_ev = torch.randn(SEEDS, c, h, w, generator=torch.Generator().manual_seed(90210))
rows = []
for color, shape in CELLS:
    t0 = time.perf_counter()
    p = ps.prompt(1,color,shape)[0]
    R = torch.stack([pool[refs[f"{color}_{shape}"][str(t)]] for t in T]).to(dev)
    for bl in BLENDS:
        imgs = {}
        pairs = [(s,t) for s in range(SEEDS) for t in T]
        oc, cc = [], []
        for i in range(0, len(pairs), 8):
            pb = pairs[i:i+8]
            e = torch.stack([eps_ev[s] for s,_ in pb]).to(dev)
            z = slerp_b(e, R[torch.tensor([T.index(t) for _,t in pb], device=dev)], bl)
            with torch.no_grad():
                u8 = to_uint8_images(gen.decode(gen.generate_latents(z, gen.encode_prompts([p]*len(pb)))))
            oc += [r.count for r in count_images(owl, list(u8), color, shape)]
            cc += [col.count(u8[k], color, shape).count for k in range(len(u8))]
            for k,(s,t) in enumerate(pb):
                imgs[(s,t)] = u8[k].astype(np.float32)
        # across-seed diversity at fixed target: mean sd over pixels, averaged over targets
        div = float(np.mean([np.stack([imgs[(s,t)] for s in range(SEEDS)]).std(axis=0).mean() for t in T]))
        for k,(s,t) in enumerate(pairs):
            rows.append(dict(color=color, shape=shape, blend=bl, seed=s, target=t,
                             owl=float(oc[k]), col=float(cc[k]), div=div))
    print(f"  {color}/{shape} done ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
json.dump(rows, open(f"{OUT}/blend.json","w"), indent=1)
bar = min(np.mean([abs(t-k) for t in T]) for k in range(5,40))
print(f"\n===== BLEND vs ACCURACY vs DIVERSITY.  constant-predictor MAE {bar:.2f} =====\n")
print(f"  {'cell':>15} {'blend':>6} {'MAE':>6} {'skill':>7} {'corr':>6} {'bias':>7} {'diversity':>10} {'vs raw':>7}")
for color, shape in CELLS:
    base = [x for x in rows if x["color"]==color and x["shape"]==shape and x["blend"]==0.0][0]["div"]
    for bl in BLENDS:
        sel=[x for x in rows if x["color"]==color and x["shape"]==shape and x["blend"]==bl]
        e=np.array([abs(x["owl"]-x["target"]) for x in sel])
        mu=np.array([np.mean([x["owl"] for x in sel if x["target"]==t]) for t in T])
        d=sel[0]["div"]
        print(f"  {color+'/'+shape:>15} {bl:6.2f} {e.mean():6.2f} {(bar-e.mean())/bar:+7.3f} "
              f"{np.corrcoef(T,mu)[0,1]:+6.2f} {np.mean([x['owl']-x['target'] for x in sel]):+7.2f} "
              f"{d:10.2f} {d/base:7.2f}")
    print()
print("  'diversity' is the across-seed pixel sd at a fixed target, averaged over targets;")
print("  'vs raw' is that as a fraction of untouched noise (blend 0.00). A blend whose accuracy")
print("  comes with 'vs raw' near zero has bought it by making every seed render the same image.")
