"""A persistent, structured store of baseline counts. Built once, read many times.

Why this exists. Every comparison so far has been against UNTOUCHED NOISE under a count-free
prompt. That baseline cannot know n - the prompt is identical for every target - so its response
is flat by construction and beating it is a low bar. It answers "better than ignoring the
request", which is worth knowing but is not the comparison a reader wants.

The comparison a reader wants is against the obvious thing: ASK THE MODEL FOR THE NUMBER. Put the
count in the prompt, "a flat lay photo of 17 red crosses on a white background", and count what
comes out. That is what a practitioner would do instead of training anything, and the sampler has
to be measured against it.

So this builds a store keyed by (colour, shape, count, seed, mode) and keeps it on disk:

  mode "prompt"     the count IS in the text. The real baseline.
  mode "countfree"  the count is absent and the noise untouched. The flat reference.

It is idempotent: a key already present is not regenerated. Rows measured by earlier runs are
imported rather than re-rendered. Both counters are recorded for every image, always.

  --import-percal2   pull the count-free baseline rows already measured into the store
  --build prompt     render and count whatever "prompt" rows are missing
  --report           print the structured table
"""
import argparse
import json
import os
import sys
import time

import numpy as np

RUNS = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs"
STORE = f"{RUNS}/baselines/store.json"
TARGETS = list(range(10, 30, 2))
SEEDS = list(range(8))
SEED_BANK = 24680
CELLS = [("red", "cross"), ("blue", "cross"), ("green", "cross"), ("yellow", "cross"),
         ("red", "star"), ("blue", "star"), ("green", "star"),
         ("red", "diamond"), ("blue", "diamond"), ("red", "triangle")]


def key(r):
    return (r["color"], r["shape"], int(r["count"]), int(r["seed"]), r["mode"])


def load():
    if os.path.exists(STORE):
        return json.load(open(STORE))
    return {"schema": 1,
            "grid": {"targets": TARGETS, "seeds": SEEDS, "seed_bank": SEED_BANK,
                     "cells": [f"{c}/{s}" for c, s in CELLS]},
            "rows": []}


def save(st):
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    json.dump(st, open(STORE, "w"), indent=1)
    print(f"  store now holds {len(st['rows'])} rows -> {STORE}")


def add(st, rows):
    have = {key(r) for r in st["rows"]}
    new = [r for r in rows if key(r) not in have]
    st["rows"] += new
    print(f"  added {len(new)} rows, skipped {len(rows) - len(new)} already present")
    return st


def build_prompt_rows(st):
    import torch
    WT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/.claude/worktrees/cs485-gpu-setup-5bb57e"
    sys.path.insert(0, WT)
    os.chdir(WT)
    from countsampler.data.prompts import PromptSampler
    from countsampler.evaluation import build_hard_counter
    from countsampler.evaluation.hard_counters import count_images
    from countsampler.generator import FrozenTurboGenerator
    from countsampler.utils.config import load_config
    from countsampler.utils.misc import resolve_device, seed_everything, to_uint8_images

    # include_count=true: the number goes INTO the text. That is the whole point of this arm.
    flat = 'data.templates=["a flat lay photo of {count} {color} {shape} on a white background"]'
    cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"],
                      [flat, "data.include_count=true", "data.n_max=28"])
    seed_everything(int(cfg.experiment.seed))
    dev = resolve_device(cfg.experiment.get("device", "auto"))
    gen = FrozenTurboGenerator.from_config(cfg.generator, dev)
    owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
    col = build_hard_counter("color")
    ps = PromptSampler(cfg.data)
    c, h, w = gen.latent_shape
    eps = torch.randn(len(SEEDS), c, h, w, generator=torch.Generator().manual_seed(SEED_BANK))

    # The template spells the number out ("fourteen"), so checking for the digit is wrong. What
    # must hold is that the text actually VARIES with the count - that is what makes this the
    # prompt-conditioned arm rather than another count-free one.
    s14, s22 = ps.prompt(14, "red", "cross")[0], ps.prompt(22, "red", "cross")[0]
    print(f"  prompt check: {s14!r}")
    print(f"                {s22!r}")
    if s14 == s22:
        raise SystemExit("  the prompt does not vary with the count - refusing to build a mislabelled store")

    have = {key(r) for r in st["rows"]}
    rows, t0 = [], time.perf_counter()
    for color, shape in CELLS:
        todo = [(s, t) for s in SEEDS for t in TARGETS
                if (color, shape, t, s, "prompt") not in have]
        if not todo:
            print(f"  {color}/{shape}: already complete", flush=True)
            continue
        for i in range(0, len(todo), 8):
            pb = todo[i:i + 8]
            z = torch.stack([eps[s] for s, _ in pb]).to(dev)
            prompts = [ps.prompt(t, color, shape)[0] for _, t in pb]
            with torch.no_grad():
                u8 = to_uint8_images(gen.decode(gen.generate_latents(z, gen.encode_prompts(prompts))))
            oc = [r.count for r in count_images(owl, list(u8), color, shape)]
            for k, (s, t) in enumerate(pb):
                rows.append(dict(color=color, shape=shape, count=t, seed=s, mode="prompt",
                                 owl=float(oc[k]),
                                 col=float(col.count(u8[k], color, shape).count),
                                 source="baseline_store.py"))
        print(f"  {color}/{shape}: {len(todo)} rendered ({(time.perf_counter() - t0) / 60:.1f} min)",
              flush=True)
    return rows


def report(st):
    rows = st["rows"]
    if not rows:
        print("  store is empty")
        return
    bar = min(np.mean([abs(t - k) for t in TARGETS]) for k in range(5, 40))
    print(f"\n===== BASELINE STORE.  best constant predictor MAE {bar:.2f} =====")
    print(f"  seed bank {st['grid']['seed_bank']}, targets {TARGETS[0]}-{TARGETS[-1]} step 2, "
          f"{len(st['grid']['seeds'])} seeds")
    print("  'skill' here is against the constant predictor, same scale as every other table.\n")
    for mode in sorted({r["mode"] for r in rows}):
        m = [r for r in rows if r["mode"] == mode]
        print(f"  --- mode '{mode}'  ({len(m)} rows) ---")
        print(f"  {'cell':>15} {'MAE':>6} {'skill':>7} {'corr':>6} {'slope':>6} {'bias':>7} {'o/c':>5}"
              "   per-count mean counted")
        for color, shape in CELLS:
            sel = [r for r in m if r["color"] == color and r["shape"] == shape]
            if not sel:
                continue
            e = np.array([abs(r["owl"] - r["count"]) for r in sel])
            mu = np.array([np.mean([r["owl"] for r in sel if r["count"] == t]) if
                           any(r["count"] == t for r in sel) else np.nan for t in TARGETS])
            ok = ~np.isnan(mu)
            tt = np.array(TARGETS)[ok]
            cr = np.corrcoef(tt, mu[ok])[0, 1] if ok.sum() > 2 and mu[ok].std() > 0 else float("nan")
            sl = np.polyfit(tt, mu[ok], 1)[0] if ok.sum() > 2 else float("nan")
            oc = np.corrcoef([r["owl"] for r in sel], [r["col"] for r in sel])[0, 1]
            print(f"  {color + '/' + shape:>15} {e.mean():6.2f} {(bar - e.mean()) / bar:+7.3f} "
                  f"{cr:+6.2f} {sl:6.2f} {np.mean([r['owl'] - r['count'] for r in sel]):+7.2f} {oc:+5.2f}   "
                  + " ".join(f"{v:.0f}" if not np.isnan(v) else "  ." for v in mu))
        e = np.array([abs(r["owl"] - r["count"]) for r in m])
        mu = np.array([np.mean([r["owl"] for r in m if r["count"] == t]) for t in TARGETS])
        oc = np.corrcoef([r["owl"] for r in m], [r["col"] for r in m])[0, 1]
        print(f"  {'POOLED':>15} {e.mean():6.2f} {(bar - e.mean()) / bar:+7.3f} "
              f"{np.corrcoef(TARGETS, mu)[0, 1]:+6.2f} {np.polyfit(TARGETS, mu, 1)[0]:6.2f} "
              f"{np.mean([r['owl'] - r['count'] for r in m]):+7.2f} {oc:+5.2f}   "
              + " ".join(f"{v:.0f}" for v in mu) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--import-percal2", action="store_true")
    ap.add_argument("--build", choices=["prompt"], default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    st = load()

    if args.import_percal2:
        src = f"{RUNS}/percal2/eval.json"
        rows = [dict(color=x["color"], shape=x["shape"], count=int(x["target"]), seed=int(x["seed"]),
                     mode="countfree", owl=x["owl"], col=x["col"], source="percal2/eval.json")
                for x in json.load(open(src)) if x["arm"] == "baseline"]
        print(f"  {len(rows)} count-free baseline rows from percal2 (seed bank {SEED_BANK})")
        save(add(st, rows))

    if args.build == "prompt":
        save(add(st, build_prompt_rows(st)))

    if args.report:
        report(st)


if __name__ == "__main__":
    main()
