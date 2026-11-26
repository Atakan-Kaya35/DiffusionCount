"""Reference pool over ten cells, with two colours held out of it entirely.

The standing limitation of the +0.272 result is that it generalises across seeds but not across
cells: a reference noise fitted on one colour/shape does not transfer, because one fixed noise
spans 28.3 objects across six cells while the cell explains 55.9 of the variance against 21.9 for
the requested target. Dphi takes c_bar, so it is at least ALLOWED to interpolate across cells
rather than memorise ten of them. This builds the training set for that test.

Cells were chosen by the two-counter screen, never by anything involving a requested count.
Circles, squares, hexagons and pentagons failed it outright - blue/circle reads 4.3 where the
colour counter finds 24.0 - so they are absent. Orange and purple appear nowhere here: they are
the held-out colours, and Dphi will be asked for them having never seen their c_bar.
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
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/general"
T = list(range(10, 30, 2)); N = 96
TRAIN_CELLS = [("red","cross"),("blue","cross"),("green","cross"),("yellow","cross"),
               ("red","star"),("blue","star"),("green","star"),
               ("red","diamond"),("blue","diamond"),("red","triangle")]
os.makedirs(OUT, exist_ok=True)
cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"], [FLAT, "data.include_count=false"])
seed_everything(0)
gen = FrozenTurboGenerator.from_config(cfg.generator, resolve_device(cfg.experiment.get("device", "auto")))
owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
col = build_hard_counter("color")
ps = PromptSampler(cfg.data)
c, h, w = gen.latent_shape
pool = torch.randn(N, c, h, w, generator=torch.Generator().manual_seed(31415))
torch.save({"pool": pool}, f"{OUT}/pool.pt")
refs, t0 = {}, time.perf_counter()
print(f"  {len(TRAIN_CELLS)} training cells x {N} noises; orange and purple held out entirely\n")
print(f"  {'cell':>16} {'range':>9} {'agree':>7} {'fitted reference counts':>44}")
for color, shape in TRAIN_CELLS:
    p = ps.prompt(1, color, shape)[0]
    o, cc = [], []
    for i in range(0, N, 8):
        z = pool[i:i+8].to(gen.device)
        with torch.no_grad():
            u8 = to_uint8_images(gen.decode(gen.generate_latents(z, gen.encode_prompts([p]*z.shape[0]))))
        o += [r.count for r in count_images(owl, list(u8), color, shape)]
        cc += [col.count(u8[k], color, shape).count for k in range(len(u8))]
    o, cc = np.array(o, float), np.array(cc, float)
    agree = np.abs(o - cc) <= np.maximum(6.0, 0.5 * np.maximum(o, cc))
    cand = np.where(agree)[0]
    r = {t: int(cand[np.argmin(np.abs(o[cand] - t))]) for t in T}
    refs[f"{color}_{shape}"] = r
    print(f"  {color+'/'+shape:>16} {int(o.min()):4d}-{int(o.max()):<4d} {agree.sum():3d}/{N} "
          f"{str([int(o[r[t]]) for t in T]):>44}  ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
json.dump({k: {str(a): b for a, b in v.items()} for k, v in refs.items()}, open(f"{OUT}/refs.json","w"), indent=1)
err = np.mean([abs(t - t) for t in T])
print(f"\n  wrote {OUT}/refs.json  ({(time.perf_counter()-t0)/60:.1f} min)")
print("  a reference whose count is far from its target means the pool does not contain that")
print("  count for that cell, and no sampler trained on it can be asked to reproduce one.")
