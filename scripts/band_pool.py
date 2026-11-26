"""Count control inside the band the prompt renders AND the counter can measure.

Two measurements force this framing. The flat-lay count-free prompt renders 10 to 29 objects over
a noise pool and never fewer, so targets 1-9 do not exist at any noise. And a search that
minimises OWLv2 selects for its blind spot - on the plain-plural prompt it chose noises the
detector reads as 0 while an independent counter finds 92. Flat-lay renders well-separated
objects, which is where the hand-count check of 2026-11-25 found OWLv2 accurate (11 vs ~12,
17 vs ~16-18), so inside 10-29 the instrument can be trusted.

So the question becomes the one that is actually answerable: with the count ABSENT from the
prompt, can the choice of noise alone place the object count on a requested value within that
band? Every reference is cross-checked against the colour counter, which has a different failure
mode, and any noise where the two disagree badly is rejected rather than used.
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
CELLS = [("blue", "pentagon"), ("red", "cross"), ("yellow", "diamond")]
TARGETS = list(range(10, 30, 2))          # 10,12,...,28 - inside the renderable, measurable band
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/band"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128

os.makedirs(OUT, exist_ok=True)
cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"], [FLAT, "data.include_count=false"])
seed_everything(0)
gen = FrozenTurboGenerator.from_config(cfg.generator, resolve_device(cfg.experiment.get("device", "auto")))
owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
col = build_hard_counter("color")
ps = PromptSampler(cfg.data)
c, h, w = gen.latent_shape
g = torch.Generator().manual_seed(515)
pool = torch.randn(N, c, h, w, generator=g)
O, C, t0 = [], [], time.perf_counter()
for color, shape in CELLS:
    enc8 = gen.encode_prompts([ps.prompt(1, color, shape)[0]] * 8)
    oc, cc = [], []
    for i in range(0, N, 8):
        z = pool[i:i+8].to(gen.device)
        e = enc8 if z.shape[0] == 8 else gen.encode_prompts([ps.prompt(1, color, shape)[0]] * z.shape[0])
        with torch.no_grad():
            u8 = to_uint8_images(gen.decode(gen.generate_latents(z, e)))
        oc += [r.count for r in count_images(owl, list(u8), color, shape)]
        cc += [col.count(u8[k], color, shape).count for k in range(len(u8))]
    O.append(oc); C.append(cc)
    print(f"  {color} {shape} done  {(time.perf_counter()-t0)/60:.1f} min", flush=True)
O, C = np.array(O, float), np.array(C, float)
mo, mc = O.mean(axis=0), C.mean(axis=0)
torch.save({"pool": pool, "owl": O, "col": C}, f"{OUT}/pool.pt")
print(f"\n  {N} noises x {len(CELLS)} cells, flat-lay count-free prompt")
print(f"  OWLv2  mean {mo.mean():5.1f}  range {mo.min():.0f}-{mo.max():.0f}")
print(f"  colour mean {mc.mean():5.1f}  range {mc.min():.0f}-{mc.max():.0f}")
print(f"  corr(OWLv2, colour) over the pool {np.corrcoef(mo, mc)[0,1]:+.3f}")
# reject noises where the two counters disagree badly - that is where a search would cheat
ok = np.abs(mo - mc) <= np.maximum(6.0, 0.5 * np.maximum(mo, mc))
print(f"  noises where the two counters agree within tolerance: {ok.sum()}/{N}")
refs, rows = {}, []
print(f"\n  {'target':>7} {'noise':>6} {'OWLv2':>7} {'colour':>7}")
for t in TARGETS:
    cand = np.where(ok)[0]
    i = int(cand[np.argmin(np.abs(mo[cand] - t))])
    refs[t] = i
    rows.append((t, i, mo[i], mc[i]))
    print(f"  {t:7d} {i:6d} {mo[i]:7.1f} {mc[i]:7.1f}")
json.dump({str(k): int(v) for k, v in refs.items()}, open(f"{OUT}/refs.json", "w"), indent=1)
orc = np.mean([abs(mo[refs[t]] - t) for t in TARGETS])
bar = min(np.mean([abs(t - k) for t in TARGETS]) for k in range(5, 40))
print(f"\n  oracle MAE {orc:.2f}   best-constant bar {bar:.2f}   skill {(bar-orc)/bar:+.3f}")
print(f"  wrote {OUT}")
