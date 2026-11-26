"""Does the NOISE place the count on a requested value, inside the band where both the prompt
renders and the detector can be trusted?

The prompt is count-free: every target gets the identical string, so the text carries no number
and any skill above the constant predictor is attributable to the noise alone. For a target t,

    z = slerp(eps_seed, r_t, blend)

where r_t is a reference noise the frozen model renders as about t objects under that same
count-free prompt. slerp holds ||z|| fixed, so z stays as Gaussian as its endpoints and cannot
collapse - which is what separated this from the learned directions that died by d = 0.08.

blend is swept because blend = 1 is the reference noise itself: maximum control, zero diversity
(every seed gives one image). Lower blend keeps the seed's own content. The trade-off is the
result, so it is reported rather than tuned away.

Every image is counted twice, by OWLv2 and by the independent colour counter. A search that
optimises one detector will find that detector's blind spot - that is how the previous attempt
was invalidated - so the two must agree for the number to mean anything.
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
from countsampler.utils.misc import ensure_dir, resolve_device, save_image_grid, seed_everything, to_uint8_images

FLAT = 'data.templates=["a flat lay photo of {count} {color} {shape} on a white background"]'
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/band"
TARGETS = list(range(10, 30, 2))
BLENDS = [0.0, 0.5, 0.75, 1.0]      # 0.0 is the untouched baseline: same noise whatever the target
SEEDS = 4

def slerp(a, b, t):
    if t == 0.0: return a.clone()
    af, bf = a.flatten().float(), b.flatten().float()
    an, bn = af / af.norm(), bf / bf.norm()
    om = torch.acos((an * bn).sum().clamp(-1 + 1e-7, 1 - 1e-7)); s = torch.sin(om)
    return ((torch.sin((1 - t) * om) / s) * af + (torch.sin(t * om) / s) * bf).view_as(a)

cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"], [FLAT, "data.include_count=false"])
seed_everything(int(cfg.experiment.seed))
gen = FrozenTurboGenerator.from_config(cfg.generator, resolve_device(cfg.experiment.get("device", "auto")))
owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
col = build_hard_counter("color")
ps = PromptSampler(cfg.data)
c, h, w = gen.latent_shape
d = torch.load(f"{OUT}/pool.pt", weights_only=False); pool = d["pool"]
refs = {int(k): v for k, v in json.load(open(f"{OUT}/refs.json")).items()}
# held-out seeds: not the pool the references were chosen from
eps = torch.randn(SEEDS, c, h, w, generator=torch.Generator().manual_seed(90210))
cells = [(cl, sh) for cl in list(cfg.data.colors)[:3] for sh in list(cfg.data.heldout_shapes)[:2]]
gd = ensure_dir(f"{OUT}/grids")
rows, t0 = [], time.perf_counter()
print(f"  {len(cells)} cells x {len(TARGETS)} targets x {SEEDS} seeds x {len(BLENDS)} blends", flush=True)
for color, shape in cells:
    prompt = ps.prompt(1, color, shape)[0]          # count-free: identical for every target
    for bl in BLENDS:
        zs = torch.stack([slerp(eps[s], pool[refs[t]], bl) for s in range(SEEDS) for t in TARGETS])
        # chunk of 8: the pool run held 8 and ran at 0.9 min / 128 images, while 40 at once
        # pushed the 8 GB card to 7.5 GB and thrashed to a near halt
        imgs, oc, cc = [], [], []
        for i in range(0, zs.shape[0], 8):
            zb = zs[i:i + 8].to(gen.device)
            with torch.no_grad():
                ib = gen.decode(gen.generate_latents(zb, gen.encode_prompts([prompt] * zb.shape[0])))
            ub = to_uint8_images(ib)
            imgs.append(ib.cpu())
            oc += [r.count for r in count_images(owl, list(ub), color, shape)]
            cc += [col.count(ub[k], color, shape).count for k in range(len(ub))]
        img = torch.cat(imgs)
        for k, (s, t) in enumerate([(s, t) for s in range(SEEDS) for t in TARGETS]):
            rows.append(dict(color=color, shape=shape, blend=bl, seed=s, target=t,
                             owl=float(oc[k]), col=float(cc[k])))
        if bl == 1.0:
            save_image_grid(torch.nn.functional.interpolate(img, scale_factor=0.5, mode="bilinear", antialias=True),
                            f"{gd}/{color}_{shape}.png", nrow=len(TARGETS))
    print(f"    {color} {shape}  {(time.perf_counter()-t0)/60:.1f} min", flush=True)
json.dump(rows, open(f"{OUT}/eval.json", "w"), indent=1)

bar = min(np.mean([abs(t - k) for t in TARGETS]) for k in range(5, 40))
print(f"\n================ COUNT CONTROL FROM NOISE ALONE, BAND 10-28 ================")
print(f"  the count is absent from the prompt; the best constant predictor scores MAE {bar:.2f}\n")
print(f"  {'blend':>6} {'MAE(OWLv2)':>11} {'skill':>7} {'MAE(colour)':>12} {'skill':>7} {'corr':>6} {'distinct':>9}")
for bl in BLENDS:
    r = [x for x in rows if x["blend"] == bl]
    mo = np.mean([abs(x["owl"] - x["target"]) for x in r])
    mc = np.mean([abs(x["col"] - x["target"]) for x in r])
    cr = np.corrcoef([x["owl"] for x in r], [x["col"] for x in r])[0, 1]
    # distinct: how many of the 10 targets get a distinguishable mean count
    mu = [np.mean([x["owl"] for x in r if x["target"] == t]) for t in TARGETS]
    dis = len(set(np.round(mu).astype(int)))
    print(f"  {bl:6.2f} {mo:11.2f} {(bar-mo)/bar:+7.3f} {mc:12.2f} {(bar-mc)/bar:+7.3f} {cr:+6.2f} {dis:6d}/10")
print(f"\n  per-target mean counted (OWLv2), by blend")
print(f"  {'blend':>6} " + " ".join(f"{t:>5}" for t in TARGETS))
for bl in BLENDS:
    r = [x for x in rows if x["blend"] == bl]
    print(f"  {bl:6.2f} " + " ".join(f"{np.mean([x['owl'] for x in r if x['target']==t]):5.1f}" for t in TARGETS))
print(f"\n  blend 0.00 is the untouched baseline - the same noise for every target, so its row")
print(f"  should be flat. A rising row at higher blend is the noise carrying the count.")
print(f"  corr is OWLv2 against the independent colour counter: low corr means the numbers")
print(f"  disagree and neither can be trusted.")
print(f"  wrote {OUT}/eval.json and {gd}")
