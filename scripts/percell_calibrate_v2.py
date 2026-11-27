"""Per-cell calibration at blend 0.85, on all ten cells, scored on a fresh seed bank.

blend_sweep.py showed the blend is the lever: the count of a reference noise survives meeting an
unseen seed only partially, and raising the blend from 0.75 to 0.85 turns all three negative cells
positive. It does so at a cost - across-seed pixel diversity falls to about 0.40 of untouched
noise - and at 0.95 the cost is 0.16, which is close to rendering the same ten images for every
seed. 0.85 is taken as the operating point and used for EVERY cell; no per-cell blend is fitted,
so nothing here is selected cell by cell.

One discipline point. The sweep that chose 0.85 ran on seed bank 90210, which is the bank the
ten-cell table is scored on. Choosing a hyper-parameter there and then reporting on it would be
selection on the test set, so this run scores on a fresh bank (24680) that played no part in the
choice. The 0.75 numbers are re-measured on the same fresh bank, so the comparison is like for
like rather than against the previously published figures.

Diversity is reported beside skill, because a method that buys counting accuracy by collapsing
the image distribution has not solved the problem it claims to.
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
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/percal2"
T = list(range(10, 30, 2)); SEEDS = 8; STEPS = 4000; EVAL_SEED = 24680
BLENDS = [0.75, 0.85]
CELLS = [("red","cross"), ("blue","cross"), ("green","cross"), ("yellow","cross"),
         ("red","star"), ("blue","star"), ("green","star"),
         ("red","diamond"), ("blue","diamond"), ("red","triangle")]
os.makedirs(OUT, exist_ok=True)

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
eps_ev = torch.randn(SEEDS, c, h, w, generator=torch.Generator().manual_seed(EVAL_SEED))
scfg = dict(cfg.sampler); scfg["norm_preserve"] = True
rows = []

def run_arm(net, cb, p, color, shape, tag, bl):
    imgs, oc, cc = {}, [], []
    pairs = [(s,t) for s in range(SEEDS) for t in T]
    for i in range(0, len(pairs), 8):
        pb = pairs[i:i+8]
        e = torch.stack([eps_ev[s] for s,_ in pb]).to(dev)
        if net is None: z = e
        else:
            with torch.no_grad():
                z = net(e, torch.tensor([t for _,t in pb], device=dev), cb[None].expand(len(pb),-1))
        with torch.no_grad():
            u8 = to_uint8_images(gen.decode(gen.generate_latents(z, gen.encode_prompts([p]*len(pb)))))
        oc += [r.count for r in count_images(owl, list(u8), color, shape)]
        cc += [col.count(u8[k], color, shape).count for k in range(len(u8))]
        for k,(s,t) in enumerate(pb): imgs[(s,t)] = u8[k].astype(np.float32)
    div = float(np.mean([np.stack([imgs[(s,t)] for s in range(SEEDS)]).std(axis=0).mean() for t in T]))
    for k,(s,t) in enumerate(pairs):
        rows.append(dict(color=color, shape=shape, arm=tag, blend=bl, seed=s, target=t,
                         owl=float(oc[k]), col=float(cc[k]), div=div))

for color, shape in CELLS:
    t0 = time.perf_counter()
    p = ps.prompt(1,color,shape)[0]; cb = gen.encode_prompts([p]).pooled[0].float()
    R = torch.stack([pool[refs[f"{color}_{shape}"][str(t)]] for t in T]).to(dev)
    NNd = torch.tensor(T, device=dev)
    run_arm(None, cb, p, color, shape, "baseline", 0.0)
    for bl in BLENDS:
        net = build_sampler(scfg, c, cb.shape[0]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS); g = torch.Generator().manual_seed(7)
        for step in range(STEPS):
            j = torch.randint(0,len(T),(16,),generator=g); e = torch.randn(16,c,h,w,generator=g).to(dev)
            loss = torch.nn.functional.mse_loss(net(e,NNd[j],cb[None].expand(16,-1)), slerp_b(e,R[j],bl))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        net.eval(); torch.save(net.state_dict(), f"{OUT}/sampler_{color}_{shape}_bl{int(bl*100)}.pt")
        run_arm(net, cb, p, color, shape, "sampler", bl)
    print(f"  {color}/{shape} done ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
json.dump(rows, open(f"{OUT}/eval.json","w"), indent=1)
print(f"\n  wrote {OUT}/eval.json  -> run scripts/percell_v2_summary.py")
