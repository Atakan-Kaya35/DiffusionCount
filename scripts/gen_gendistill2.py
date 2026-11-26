"""Train Dphi on ten cells, then ask it for colours it has never seen.

The question. The +0.272 result generalises across seeds but not across cells, because its
references were fitted per cell. Dphi takes c_bar, so it is permitted to learn the MAPPING from
(count, cell) to a region of the shell rather than memorise ten lookup entries. If it has learned
the mapping, a colour whose c_bar never appeared in training should still work. If it has
memorised, held-out colours will fall to the baseline.

Orange and purple appear nowhere in the pool or the training set. They are asked for at test time
on three cells: orange/cross, purple/cross, orange/triangle.

Training is supervised regression onto slerp(eps, r_{n,c}, 0.75) with norm_preserve on, the
projection without which no admissible w reaches a displacement of 1.112.
"""
import json, os, sys, time
import numpy as np, torch
WT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/.claude/worktrees/cs485-gpu-setup-5bb57e"
sys.path.insert(0, WT); os.chdir(WT)
from countsampler.data.prompts import PromptSampler
from countsampler.generator import FrozenTurboGenerator
from countsampler.novel.shared import build_sampler
from countsampler.utils.config import load_config
from countsampler.utils.misc import resolve_device, seed_everything
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_condition import condition
FLAT = 'data.templates=["a flat lay photo of {count} {color} {shape} on a white background"]'
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/general"
T = list(range(10, 30, 2)); BL = 0.75
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "rich"

def slerp_b(a, b, t):
    af, bf = a.flatten(1).float(), b.flatten(1).float()
    an, bn = af / af.norm(dim=1, keepdim=True), bf / bf.norm(dim=1, keepdim=True)
    om = torch.acos((an * bn).sum(1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)); s = torch.sin(om)
    return ((torch.sin((1 - t) * om) / s) * af + (torch.sin(t * om) / s) * bf).view_as(a)

cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"], [FLAT, "data.include_count=false"])
seed_everything(0)
dev = resolve_device(cfg.experiment.get("device", "auto"))
gen = FrozenTurboGenerator.from_config(cfg.generator, dev)
ps = PromptSampler(cfg.data)
c, h, w = gen.latent_shape
pool = torch.load(f"{OUT}/pool.pt", weights_only=False)["pool"]
refs = json.load(open(f"{OUT}/refs.json"))
CELLS = [tuple(k.split("_")) for k in refs]
R = torch.stack([pool[refs[f"{cl}_{sh}"][str(t)]] for cl, sh in CELLS for t in T])
CB = torch.stack([condition(gen.encode_prompts([ps.prompt(1, cl, sh)[0]]), VARIANT)[0].cpu()
                  for cl, sh in CELLS for t in T])
NN = torch.tensor([t for cl, sh in CELLS for t in T])
print(f"  {len(CELLS)} cells x {len(T)} targets = {len(NN)} (cell, target) pairs")
print(f"  condition variant: {VARIANT}, dim {CB.shape[1]}")
print(f"  training colours: {sorted({cl for cl, _ in CELLS})}")
scfg = dict(cfg.sampler); scfg["norm_preserve"] = True
net = build_sampler(scfg, c, CB.shape[1]).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=0.0)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
B, g, t0 = 16, torch.Generator().manual_seed(7), time.perf_counter()
Rd, CBd, NNd = R.to(dev), CB.to(dev), NN.to(dev)
for step in range(STEPS):
    j = torch.randint(0, len(NN), (B,), generator=g)
    eps = torch.randn(B, c, h, w, generator=g).to(dev)
    tgt = slerp_b(eps, Rd[j], BL)
    z = net(eps, NNd[j], CBd[j])
    loss = torch.nn.functional.mse_loss(z, tgt)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
    if step % 1000 == 0 or step == STEPS - 1:
        with torch.no_grad():
            cos = torch.nn.functional.cosine_similarity((z-eps).flatten(1), (tgt-eps).flatten(1)).mean().item()
            d = ((z - eps).flatten(1).norm(dim=1) / (c*h*w)**0.5).mean().item()
        print(f"  step {step:5d}  mse {loss.item():.4f}  displacement {d:.3f}  cos {cos:+.3f}  "
              f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
torch.save({"state": net.state_dict(), "scfg": scfg, "cells": CELLS, "targets": T, "variant": VARIANT},
           f"{OUT}/sampler_{VARIANT}.pt")
print(f"\n  wrote {OUT}/sampler.pt")
print("  cos here is on TRAINING cells and says only that the regression fitted. Whether Dphi")
print("  learned a mapping or memorised ten entries is decided by the held-out colours.")
