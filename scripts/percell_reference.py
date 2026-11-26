"""If a reference noise references only the cell it was fitted on, fit one per cell.

That is not a workaround, it is what the measurement implies: the colour/shape cell accounted
for 55.9 of the variance in the counted number against 21.9 for the requested target. Dphi takes
c_bar, so a per-cell mapping from target to noise is inside the hypothesis class the project is
actually proposing - this measures whether that mapping exists before anything tries to learn it.

The honest test. At blend 1.0 the evaluated image IS the image that was counted when the
reference was chosen, so any skill there is fitting on test and means nothing. So the references
are chosen on one set of noises, and scored after blending with HELD-OUT seeds, where the
resulting image was never counted. Blend 1.0 is still reported, clearly labelled, as the
fit-on-test ceiling.

Cells are the three where OWLv2 and the colour counter agree within 2 objects; pentagons are
excluded because the detector cannot see them.
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
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/percell"
CELLS = [("blue", "diamond"), ("green", "diamond"), ("red", "diamond")]
T = list(range(10, 30, 2)); N = 128; SEEDS = 4
BLENDS = [0.0, 0.5, 0.75, 1.0]
os.makedirs(OUT, exist_ok=True)

def slerp(a, b, t):
    if t == 0.0: return a.clone()
    af, bf = a.flatten().float(), b.flatten().float()
    an, bn = af / af.norm(), bf / bf.norm()
    om = torch.acos((an * bn).sum().clamp(-1 + 1e-7, 1 - 1e-7)); s = torch.sin(om)
    return ((torch.sin((1 - t) * om) / s) * af + (torch.sin(t * om) / s) * bf).view_as(a)

cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"], [FLAT, "data.include_count=false"])
seed_everything(0)
gen = FrozenTurboGenerator.from_config(cfg.generator, resolve_device(cfg.experiment.get("device", "auto")))
owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
col = build_hard_counter("color")
ps = PromptSampler(cfg.data)
c, h, w = gen.latent_shape
pool = torch.randn(N, c, h, w, generator=torch.Generator().manual_seed(4242))
eps = torch.randn(SEEDS, c, h, w, generator=torch.Generator().manual_seed(90210))

def run(zs, color, shape):
    p = ps.prompt(1, color, shape)[0]
    o, cc = [], []
    for i in range(0, zs.shape[0], 8):
        zb = zs[i:i+8].to(gen.device)
        with torch.no_grad():
            u8 = to_uint8_images(gen.decode(gen.generate_latents(zb, gen.encode_prompts([p]*zb.shape[0]))))
        o += [r.count for r in count_images(owl, list(u8), color, shape)]
        cc += [col.count(u8[k], color, shape).count for k in range(len(u8))]
    return np.array(o, float), np.array(cc, float)

t0, rows, refs_all = time.perf_counter(), [], {}
for color, shape in CELLS:
    o, cc = run(pool, color, shape)
    agree = np.abs(o - cc) <= np.maximum(6.0, 0.5 * np.maximum(o, cc))
    cand = np.where(agree)[0]
    refs = {t: int(cand[np.argmin(np.abs(o[cand] - t))]) for t in T}
    refs_all[f"{color}_{shape}"] = refs
    print(f"  {color} {shape}: pool range {o.min():.0f}-{o.max():.0f}, "
          f"{agree.sum()}/{N} counters agree, fitted refs {[int(o[refs[t]]) for t in T]}  "
          f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    for bl in BLENDS:
        zs = torch.stack([slerp(eps[s], pool[refs[t]], bl) for s in range(SEEDS) for t in T])
        eo, ec = run(zs, color, shape)
        for k, (s, t) in enumerate([(s, t) for s in range(SEEDS) for t in T]):
            rows.append(dict(color=color, shape=shape, blend=bl, seed=s, target=t,
                             owl=float(eo[k]), col=float(ec[k])))
    print(f"    evaluated ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
json.dump(rows, open(f"{OUT}/eval.json", "w"), indent=1)
json.dump({k: {str(a): b for a, b in v.items()} for k, v in refs_all.items()}, open(f"{OUT}/refs.json", "w"), indent=1)
bar = min(np.mean([abs(t - k) for t in T]) for k in range(5, 40))
print(f"\n========== PER-CELL REFERENCE NOISE, COUNT ABSENT FROM THE PROMPT ==========")
print(f"  best constant predictor MAE {bar:.2f}\n")
print(f"  {'blend':>6} {'MAE':>6} {'skill':>7} {'corr':>6} {'slope':>6} {'sd':>5} {'corr(o,c)':>10}   note")
for bl in BLENDS:
    r = [x for x in rows if x["blend"] == bl]
    mae = np.mean([abs(x["owl"] - x["target"]) for x in r])
    mu = np.array([np.mean([x["owl"] for x in r if x["target"] == t]) for t in T])
    sd = np.mean([np.std([x["owl"] for x in r if x["target"] == t]) for t in T])
    cr = np.corrcoef(T, mu)[0, 1] if mu.std() > 0 else float("nan")
    sl = np.polyfit(T, mu, 1)[0]
    oc = np.corrcoef([x["owl"] for x in r], [x["col"] for x in r])[0, 1]
    note = "fit on test, ceiling only" if bl == 1.0 else ("untouched baseline" if bl == 0.0 else "held-out seeds")
    print(f"  {bl:6.2f} {mae:6.2f} {(bar-mae)/bar:+7.3f} {cr:+6.2f} {sl:6.2f} {sd:5.2f} {oc:+10.2f}   {note}")
print(f"\n  per-target mean counted")
print(f"  {'blend':>6} " + " ".join(f"{t:>5}" for t in T))
for bl in BLENDS:
    r = [x for x in rows if x["blend"] == bl]
    print(f"  {bl:6.2f} " + " ".join(f"{np.mean([x['owl'] for x in r if x['target']==t]):5.1f}" for t in T))
print(f"\n  wrote {OUT}")
