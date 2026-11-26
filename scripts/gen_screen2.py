"""Which (colour, shape) cells can the instrument actually measure?

Pentagons turned out to be invisible to OWLv2 in these renders - 64 of 240 images read as 0 to 2
objects while the colour counter found a mean of 22. That was caught only because every image is
counted twice. Assuming the other seven shapes are fine would be making the same mistake again,
so this screens all of them before any cell is used for anything.

The criterion is agreement between OWLv2 and the colour counter, which knows nothing about any
requested count, so selecting cells on it cannot flatter a later result. Also reported is the
count RANGE each cell spans over the pool, because a cell whose range does not contain the
targets cannot be controlled no matter how good the sampler is.
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
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/screen"
N = 32
os.makedirs(OUT, exist_ok=True)
cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"], [FLAT, "data.include_count=false"])
seed_everything(0)
gen = FrozenTurboGenerator.from_config(cfg.generator, resolve_device(cfg.experiment.get("device", "auto")))
owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
col = build_hard_counter("color")
ps = PromptSampler(cfg.data)
c, h, w = gen.latent_shape
pool = torch.randn(N, c, h, w, generator=torch.Generator().manual_seed(99))
# round and many-sided shapes failed the first screen outright: blue/circle read 4.3 against the
# colour counter's 24.0, blue/square 5.6 against 29.3, hexagons disagreed by 19-23 in both
# colours. Only these four survived, so only these are screened across the remaining colours.
SHAPES = ["cross", "diamond", "star", "triangle"]
COLORS = ["green", "yellow", "orange", "purple"]
rows, t0 = [], time.perf_counter()
print(f"  {len(COLORS)}x{len(SHAPES)} cells, {N} noises each, count-free flat-lay prompt\n")
print(f"  {'cell':>16} {'OWLv2':>7} {'colour':>7} {'disagree':>9} {'corr':>6} {'range':>9}  verdict")
for shape in SHAPES:
    for color in COLORS:
        p = ps.prompt(1, color, shape)[0]
        o, cc = [], []
        for i in range(0, N, 8):
            z = pool[i:i+8].to(gen.device)
            with torch.no_grad():
                u8 = to_uint8_images(gen.decode(gen.generate_latents(z, gen.encode_prompts([p]*z.shape[0]))))
            o += [r.count for r in count_images(owl, list(u8), color, shape)]
            cc += [col.count(u8[k], color, shape).count for k in range(len(u8))]
        o, cc = np.array(o, float), np.array(cc, float)
        dis = abs(o.mean() - cc.mean())
        cr = np.corrcoef(o, cc)[0, 1] if o.std() > 0 and cc.std() > 0 else float("nan")
        # per-image correlation matters as much as agreement of the means: blue/diamond had the
        # means within 2.7 but a per-image corr of -0.08, which is two instruments that happen to
        # average alike while disagreeing on every individual picture
        usable = dis <= 5.0 and o.min() <= 12 and o.max() >= 26 and (cr == cr and cr >= 0.35)
        rows.append(dict(color=color, shape=shape, owl=float(o.mean()), col=float(cc.mean()),
                         disagree=float(dis), corr=float(cr) if cr == cr else None,
                         lo=float(o.min()), hi=float(o.max()), usable=bool(usable)))
        print(f"  {color+'/'+shape:>16} {o.mean():7.1f} {cc.mean():7.1f} {dis:9.1f} {cr:+6.2f} "
              f"{int(o.min()):4d}-{int(o.max()):<4d}  {'USABLE' if usable else ''}", flush=True)
json.dump(rows, open(f"{OUT}/screen2.json", "w"), indent=1)
ok = sorted({r["shape"] for r in rows if r["usable"]})
print(f"\n  shapes usable in both colours: {[s for s in ok if sum(r['usable'] for r in rows if r['shape']==s)==len(COLORS)]}")
print(f"  usable anywhere: {ok}")
print(f"  criterion: means within 5 objects, per-image corr >= 0.35, pool range covers 12 to 26")
print(f"  {(time.perf_counter()-t0)/60:.1f} min")
