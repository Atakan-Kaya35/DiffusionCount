"""Distil the per-cell reference mapping into Dphi - a learned sampler, not a lookup.

What makes this possible. The per-cell pool established that for each (cell, target) there EXISTS
a noise the frozen model renders as that count, verified by two independent counters, and that
blending toward it at 0.75 beats the constant predictor on held-out seeds. So the mapping is no
longer something to discover from a reward - six policy-gradient runs could not get a usable
gradient from one - it is a supervised target: Dphi must reproduce slerp(eps, r_{n,c}, 0.75).

What makes it representable. That target sits at a normalised displacement of 1.112 from eps,
and learned ADDITIVE directions were measured to collapse by 0.08. The sampler is therefore built
with norm_preserve, which rescales z back to ||eps|| - the property that lets slerp travel to
1.42 without ever leaving the shell the U-Net was trained on. Without it no admissible w reaches
the target and the regression is asking for something the architecture cannot express.

The generator is not in the loop: both the input and the target are noises, so this is an
ordinary regression and runs in seconds. The test is afterwards, on held-out seeds, and it is the
only thing that counts - whether Dphi's noise renders the requested number.
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
FLAT = 'data.templates=["a flat lay photo of {count} {color} {shape} on a white background"]'
PC = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/percell"
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/distill"
T = list(range(10, 30, 2)); BL = 0.75
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
os.makedirs(OUT, exist_ok=True)

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
pool = torch.randn(128, c, h, w, generator=torch.Generator().manual_seed(4242))
refs = json.load(open(f"{PC}/refs.json"))
CELLS = [tuple(k.split("_")) for k in refs]
cbar = {}
for color, shape in CELLS:
    cbar[(color, shape)] = gen.encode_prompts([ps.prompt(1, color, shape)[0]]).pooled[0].float().cpu()
R = torch.stack([pool[refs[f"{cl}_{sh}"][str(t)]] for cl, sh in CELLS for t in T])  # [cells*T, C,H,W]
IDX = {(cl, sh, t): i for i, (cl, sh, t) in enumerate([(cl, sh, t) for cl, sh in CELLS for t in T])}
CB = torch.stack([cbar[(cl, sh)] for cl, sh in CELLS for t in T])
NN = torch.tensor([t for cl, sh in CELLS for t in T])

scfg = dict(cfg.sampler); scfg["norm_preserve"] = True
net = build_sampler(scfg, c, CB.shape[1]).to(dev)
print(f"  sampler {sum(p.numel() for p in net.parameters())/1e6:.2f}M params, norm_preserve=True")
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
    if step % 500 == 0 or step == STEPS - 1:
        with torch.no_grad():
            d = ((z - eps).flatten(1).norm(dim=1) / (c*h*w)**0.5).mean().item()
            dt = ((tgt - eps).flatten(1).norm(dim=1) / (c*h*w)**0.5).mean().item()
            cos = torch.nn.functional.cosine_similarity((z-eps).flatten(1), (tgt-eps).flatten(1)).mean().item()
        print(f"  step {step:5d}  mse {loss.item():.4f}  displacement {d:.3f} (target {dt:.3f})  "
              f"direction cos {cos:+.3f}  {(time.perf_counter()-t0)/60:.1f} min", flush=True)
torch.save({"state": net.state_dict(), "scfg": scfg, "cells": CELLS, "targets": T}, f"{OUT}/sampler.pt")
print(f"\n  wrote {OUT}/sampler.pt")
print("  mse is against a target at displacement 1.112; direction cos is what actually matters,")
print("  since reaching the right REGION of the shell is the mechanism, not hitting one point.")
