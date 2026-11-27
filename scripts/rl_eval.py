"""Score an RL checkpoint on the stored grid, as one more arm beside the baselines.

Single cell, since the run trains on one: same seed bank, same counts, both counters, and the
baselines read from runs/baselines/store.json rather than regenerated.
"""
import json
import os
import sys

import numpy as np
import torch

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
CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{RUNS}/rl_shell2/sampler_last.pt"
CELL = (sys.argv[2] if len(sys.argv) > 2 else "red/cross").split("/")
T = list(range(10, 30, 2))
SEEDS = list(range(8))
SEED_BANK = 24680
FLAT = 'data.templates=["a flat lay photo of {count} {color} {shape} on a white background"]'

color, shape = CELL
cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"],
                  [FLAT, "data.include_count=false", "data.n_max=28"])
seed_everything(int(cfg.experiment.seed))
dev = resolve_device(cfg.experiment.get("device", "auto"))
gen = FrozenTurboGenerator.from_config(cfg.generator, dev)
owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
col = build_hard_counter("color")
ps = PromptSampler(cfg.data)
c, h, w = gen.latent_shape
scfg = dict(cfg.sampler)
scfg["norm_preserve"] = True
cb = gen.encode_prompts([ps.prompt(1, color, shape)[0]]).pooled[0].float()
net = build_sampler(scfg, c, cb.shape[0]).to(dev)
sd = torch.load(CKPT, map_location=dev, weights_only=False)
# the trainer writes a wrapper carrying the config beside the weights; percal writes bare weights
net.load_state_dict(sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd)
net.eval()
p = ps.prompt(1, color, shape)[0]
eps = torch.randn(len(SEEDS), c, h, w, generator=torch.Generator().manual_seed(SEED_BANK))

pairs = [(s, t) for s in SEEDS for t in T]
oc, cc, bgs = [], [], []
for i in range(0, len(pairs), 8):
    pb = pairs[i:i + 8]
    e = torch.stack([eps[s] for s, _ in pb]).to(dev)
    with torch.no_grad():
        z = net(e, torch.tensor([t for _, t in pb], device=dev), cb[None].expand(len(pb), -1))
        u8 = to_uint8_images(gen.decode(gen.generate_latents(z, gen.encode_prompts([p] * len(pb)))))
    oc += [r.count for r in count_images(owl, list(u8), color, shape)]
    cc += [col.count(u8[k], color, shape).count for k in range(len(u8))]
    bgs += list(((u8.min(axis=3) > 170) & ((u8.max(axis=3) - u8.min(axis=3)) < 40)).mean(axis=(1, 2)))

oc = np.array(oc, float)
tg = np.array([t for _, t in pairs], float)
bar = min(np.mean([abs(t - k) for t in T]) for k in range(5, 40))
mu = np.array([oc[tg == t].mean() for t in T])
st = json.load(open(f"{RUNS}/baselines/store.json"))
ref = {}
for mode in ("prompt", "countfree"):
    sel = [r for r in st["rows"] if r["color"] == color and r["shape"] == shape and r["mode"] == mode]
    ref[mode] = float(np.mean([abs(r["owl"] - r["count"]) for r in sel]))
master = json.load(open(f"{RUNS}/baselines/master.json"))[f"{color}/{shape}"]

print(f"\n===== RL CHECKPOINT ON THE STORED GRID.  {color}/{shape} =====")
print(f"  {os.path.relpath(CKPT, RUNS)}, seed bank {SEED_BANK}, counts {T[0]}-{T[-1]}\n")
print(f"  {'arm':>22} {'MAE':>6} {'skill':>7}")
for name, v in (("prompt (count in text)", ref["prompt"]), ("count-free, raw noise", ref["countfree"]),
                ("best constant", bar), ("dphi calibrated 0.85", master["dphi85"]),
                ("RL policy gradient", float(np.abs(oc - tg).mean()))):
    print(f"  {name:>22} {v:6.2f} {(bar - v) / bar:+7.3f}")
print(f"\n  RL per-count mean counted: " + " ".join(f"{v:.0f}" for v in mu))
print(f"  requested:                 " + " ".join(f"{t:.0f}" for t in T))
print(f"  corr {np.corrcoef(T, mu)[0, 1]:+.2f}  slope {np.polyfit(T, mu, 1)[0]:.2f}  "
      f"bias {np.mean(oc - tg):+.2f}  backdrop {np.mean(bgs):.2f}  "
      f"OWLv2 vs colour counter {np.corrcoef(oc, cc)[0, 1]:+.2f}")
