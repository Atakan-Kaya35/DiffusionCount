"""Calibrate Dphi on the six remaining cells that the 96-noise pool already covers.

percell_calibrate.py established the procedure on four cells. The pool in runs/general holds
counted references for ten, so the other six cost only training and evaluation - no new counting
pass - and they are what turns a four-cell claim into a ten-cell one.

Nothing about the method changes: count absent from the prompt (identical text for every target),
eight seeds never used in training, every image counted by OWLv2 and by the independent colour
counter. Writes eval_more.json; percell_calibrate_summary.py merges it with the original four.
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
CELLS = [("green","cross"), ("yellow","cross"), ("red","star"),
         ("blue","star"), ("blue","diamond"), ("red","triangle")]
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
    torch.save(net.state_dict(), f"{OUT}/sampler_{color}_{shape}.pt")
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
json.dump(rows, open(f"{OUT}/eval_more.json","w"), indent=1)
json.dump(cost, open(f"{OUT}/cost_more.json","w"), indent=1)
print(f"\n  wrote {OUT}/eval_more.json  -> run scripts/percell_calibrate_summary.py")
