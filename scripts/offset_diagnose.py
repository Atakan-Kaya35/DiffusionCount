"""Where does the offset on the negative cells enter?

The references are not the problem. gen_genpool.py fitted them to within a count on every cell -
red/triangle's are exactly [10..28] - so the pool contains the right noises and the selection
found them. Yet red/triangle tracks the request at r +0.91 and still scores -0.053: it moves the
right way and lands too high.

Three places the offset can live, measured here as three arms on the same held-out seeds:

  ref      z = r_n itself              what the reference is worth at full strength
  slerp    z = slerp(eps, r_n, 0.75)   what the TRAINING TARGET is worth on an unseen seed
  dphi     z = Dphi(eps, n, c_bar)     what the trained sampler delivers

If slerp already carries the offset, blending only 0.75 of the way leaves a pull toward the seed's
own count and the fix is the blend. If slerp is clean and dphi is not, the regression is the gap.
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
T = list(range(10, 30, 2)); BL = 0.75; SEEDS = 8
CELLS = [("red","triangle"), ("blue","diamond"), ("green","star"), ("red","cross")]

def slerp_b(a,b,t):
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
scfg = dict(cfg.sampler); scfg["norm_preserve"] = True
rows = []
for color, shape in CELLS:
    t0 = time.perf_counter()
    p = ps.prompt(1,color,shape)[0]; cb = gen.encode_prompts([p]).pooled[0].float()
    R = torch.stack([pool[refs[f"{color}_{shape}"][str(t)]] for t in T]).to(dev)
    ck = f"{OUT}/sampler_{color}_{shape}.pt"
    net = None
    if os.path.exists(ck):
        net = build_sampler(scfg, c, cb.shape[0]).to(dev)
        net.load_state_dict(torch.load(ck, map_location=dev)); net.eval()
    arms = ["ref", "slerp"] + (["dphi"] if net is not None else [])
    for arm in arms:
        pairs = [(s,t) for s in range(SEEDS) for t in T]
        oc, cc = [], []
        for i in range(0, len(pairs), 8):
            pb = pairs[i:i+8]
            e = torch.stack([eps_ev[s] for s,_ in pb]).to(dev)
            ridx = torch.tensor([T.index(t) for _,t in pb], device=dev)
            if arm == "ref":     z = R[ridx]
            elif arm == "slerp": z = slerp_b(e, R[ridx], BL)
            else:
                with torch.no_grad():
                    z = net(e, torch.tensor([t for _,t in pb], device=dev), cb[None].expand(len(pb),-1))
            with torch.no_grad():
                u8 = to_uint8_images(gen.decode(gen.generate_latents(z, gen.encode_prompts([p]*len(pb)))))
            oc += [r.count for r in count_images(owl, list(u8), color, shape)]
            cc += [col.count(u8[k], color, shape).count for k in range(len(u8))]
        for k,(s,t) in enumerate(pairs):
            rows.append(dict(color=color, shape=shape, arm=arm, seed=s, target=t,
                             owl=float(oc[k]), col=float(cc[k])))
    print(f"  {color}/{shape} done ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
json.dump(rows, open(f"{OUT}/offset.json","w"), indent=1)
bar = min(np.mean([abs(t-k) for t in T]) for k in range(5,40))
print(f"\n===== WHERE THE OFFSET ENTERS.  constant-predictor MAE {bar:.2f} =====\n")
print(f"  {'cell':>15} {'arm':>6} {'MAE':>6} {'skill':>7} {'corr':>6} {'slope':>6} {'bias':>7}   per-target mean counted")
for color, shape in CELLS:
    for arm in ("ref","slerp","dphi"):
        sel=[x for x in rows if x["color"]==color and x["shape"]==shape and x["arm"]==arm]
        if not sel: continue
        e=np.array([abs(x["owl"]-x["target"]) for x in sel])
        bias=np.mean([x["owl"]-x["target"] for x in sel])
        mu=np.array([np.mean([x["owl"] for x in sel if x["target"]==t]) for t in T])
        print(f"  {color+'/'+shape:>15} {arm:>6} {e.mean():6.2f} {(bar-e.mean())/bar:+7.3f} "
              f"{np.corrcoef(T,mu)[0,1]:+6.2f} {np.polyfit(T,mu,1)[0]:6.2f} {bias:+7.2f}   "
              + " ".join(f"{m:.0f}" for m in mu))
    print()
print("  'bias' is mean(counted - target): a positive number means the arm lands high.")
print("  ref is the ceiling this mechanism has on that cell; slerp is what Dphi was trained to copy.")
