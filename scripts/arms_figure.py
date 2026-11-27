"""All four arms on one cell, one seed: what each actually draws.

Rows, top to bottom:
  prompt      the count IS in the text - what you get by just asking
  count-free  count absent, noise untouched - flat by construction
  dphi@0.85   the calibrated sampler at the operating point
  RL          the policy-gradient checkpoint, shown because it failed and the failure is visible
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

WT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/.claude/worktrees/cs485-gpu-setup-5bb57e"
sys.path.insert(0, WT)
os.chdir(WT)
from countsampler.data.prompts import PromptSampler
from countsampler.evaluation import build_hard_counter
from countsampler.evaluation.hard_counters import count_images
from countsampler.generator import FrozenTurboGenerator
from countsampler.novel.shared import build_sampler
from countsampler.utils.config import load_config
from countsampler.utils.misc import resolve_device, seed_everything, to_uint8_images

RUNS = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs"
OUT = f"{RUNS}/baselines"
T = list(range(10, 30, 2))
SEED_BANK = 24680
color, shape = (sys.argv[1] if len(sys.argv) > 1 else "red/cross").split("/")
FLAT = 'data.templates=["a flat lay photo of {count} {color} {shape} on a white background"]'

cfg_cf = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"],
                     [FLAT, "data.include_count=false", "data.n_max=28"])
cfg_p = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"],
                    [FLAT, "data.include_count=true", "data.n_max=28"])
seed_everything(int(cfg_cf.experiment.seed))
dev = resolve_device(cfg_cf.experiment.get("device", "auto"))
gen = FrozenTurboGenerator.from_config(cfg_cf.generator, dev)
owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
ps_cf, ps_p = PromptSampler(cfg_cf.data), PromptSampler(cfg_p.data)
c, h, w = gen.latent_shape
eps = torch.randn(8, c, h, w, generator=torch.Generator().manual_seed(SEED_BANK))
e0 = torch.stack([eps[0]] * len(T)).to(dev)
p_cf = ps_cf.prompt(1, color, shape)[0]
cb = gen.encode_prompts([p_cf]).pooled[0].float()
scfg = dict(cfg_cf.sampler)
scfg["norm_preserve"] = True
NNd = torch.tensor(T, device=dev)
master = json.load(open(f"{RUNS}/baselines/master.json"))[f"{color}/{shape}"]


def load(path):
    net = build_sampler(scfg, c, cb.shape[0]).to(dev)
    sd = torch.load(path, map_location=dev, weights_only=False)
    net.load_state_dict(sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd)
    net.eval()
    return net


arms = []
arms.append(("prompt: count in the text", f"MAE {master['prompt']:.2f}",
             [ps_p.prompt(t, color, shape)[0] for t in T], None))
arms.append(("count-free, untouched noise", f"MAE {master['countfree']:.2f}",
             [p_cf] * len(T), None))
arms.append(("Dphi calibrated, blend 0.85", f"MAE {master['dphi85']:.2f}",
             [p_cf] * len(T), f"{RUNS}/percal2/sampler_{color}_{shape}_bl85.pt"))
rl = f"{RUNS}/rl_shell2/sampler_last.pt"
if os.path.exists(rl):
    arms.append(("RL policy gradient (failed)", "MAE 19.00", [p_cf] * len(T), rl))

S, PAD, LAB, HDR = 180, 6, 235, 42
img = Image.new("RGB", (LAB + len(T) * (S + PAD), HDR + len(arms) * (S + PAD + 20) + 10), "white")
d = ImageDraw.Draw(img)
d.text((LAB, 6), f"{color} {shape}, one held-out seed. Every arm on the same noise, the same grid, "
                 "counted by the same detector.", fill="black")
d.text((LAB, 23), "Only the top row has the number in its prompt; the rest use one identical "
                  "count-free string for all ten columns.", fill=(90, 90, 90))
for row, (name, note, prompts, ckpt) in enumerate(arms):
    z = e0
    if ckpt:
        with torch.no_grad():
            z = load(ckpt)(e0, NNd, cb[None].expand(len(T), -1))
    with torch.no_grad():
        u8 = to_uint8_images(gen.decode(gen.generate_latents(z, gen.encode_prompts(prompts))))
    cs = [q.count for q in count_images(owl, list(u8), color, shape)]
    y = HDR + row * (S + PAD + 20)
    d.text((8, y + S // 2 - 14), name[:32], fill="black")
    d.text((8, y + S // 2 + 2), note, fill=(60, 60, 60))
    for k, t in enumerate(T):
        x = LAB + k * (S + PAD)
        img.paste(Image.fromarray(u8[k]).resize((S, S), Image.LANCZOS), (x, y))
        if row == 0:
            d.text((x + S // 2 - 22, 28), f"ask {t}", fill="black")
        d.text((x + 4, y + S + 3), f"counted {cs[k]}",
               fill=(0, 130, 0) if abs(cs[k] - t) <= 3 else
               ((200, 120, 0) if abs(cs[k] - t) <= 6 else (190, 0, 0)))
    print(f"  {name}: {cs}", flush=True)
img.save(f"{OUT}/arms.png")
print(f"  wrote {OUT}/arms.png")
