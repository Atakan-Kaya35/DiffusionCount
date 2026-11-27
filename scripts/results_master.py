"""One table, every arm, scored from the stored baselines rather than regenerated.

Each script used to carry its own baseline, which meant the comparison moved whenever a run was
repeated and the weakest baseline - untouched noise under a count-free prompt - was the one
usually quoted. The baseline store fixes the reference set: it is keyed by (colour, shape, count,
seed, mode), built once, and read here. Nothing in this file renders an image.

Arms, all on the same grid and the same seed bank:

  prompt        the count IS in the text. What you get by just asking the model.
  prompt+cal    that arm with each cell's own response curve inverted. Not a usable method - it
                is the "ask for a different number" tactic - but it is the strongest honest
                version of the prompt baseline, so the sampler should be measured against it.
  countfree     count absent, noise untouched. Flat by construction; the weak reference.
  constant      the best single number for the whole band. MAE 5.00 here.
  dphi@0.75     the calibrated sampler at the original blend.
  dphi@0.85     the calibrated sampler at the operating point.

The paired bootstrap compares dphi@0.85 against the PROMPT baseline, unit (cell, seed, count),
because that is the comparison a reader cares about.
"""
import json

import numpy as np

RUNS = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs"
STORE = f"{RUNS}/baselines/store.json"
PERCAL2 = f"{RUNS}/percal2/eval.json"
T = list(range(10, 30, 2))


def mae(rows, kt="count"):
    return float(np.mean([abs(r["owl"] - r[kt]) for r in rows]))


def main():
    st = json.load(open(STORE))
    cells = [tuple(c.split("/")) for c in st["grid"]["cells"]]
    base = st["rows"]
    samp = json.load(open(PERCAL2))
    bar = min(np.mean([abs(t - k) for t in T]) for k in range(5, 40))
    rng = np.random.default_rng(0)

    def arm(color, shape, which):
        if which in ("prompt", "countfree"):
            return {(r["seed"], r["count"]): r["owl"] for r in base
                    if r["color"] == color and r["shape"] == shape and r["mode"] == which}
        bl = float(which)
        return {(r["seed"], r["target"]): r["owl"] for r in samp
                if r["color"] == color and r["shape"] == shape
                and r["arm"] == "sampler" and r["blend"] == bl}

    def calibrated(d):
        """Invert this cell's own fitted response: the best 'just ask' could ever do here."""
        x = np.array([k[1] for k in d], float)
        y = np.array([d[k] for k in d], float)
        a, b = np.polyfit(x, y, 1)
        if abs(a) < 1e-6:
            return {k: float("nan") for k in d}
        return {k: (d[k] - b) / a for k in d}

    print(f"\n===== MASTER RESULTS.  MAE in objects, lower is better =====")
    print(f"  grid: {len(cells)} cells x counts {T[0]}-{T[-1]} step 2 x {len(st['grid']['seeds'])} "
          f"seeds (bank {st['grid']['seed_bank']})")
    print(f"  baselines read from the store, not regenerated.  constant predictor = {bar:.2f}\n")
    head = f"  {'cell':>15} {'prompt':>7} {'+cal':>7} {'countfree':>10} {'constant':>9} {'dphi.75':>8} {'dphi.85':>8}"
    print(head)
    rowsout = {}
    for color, shape in cells:
        p = arm(color, shape, "prompt")
        cf = arm(color, shape, "countfree")
        d75 = arm(color, shape, "0.75")
        d85 = arm(color, shape, "0.85")
        pc = calibrated(p)
        f = lambda d: float(np.mean([abs(v - k[1]) for k, v in d.items()])) if d else float("nan")
        rowsout[f"{color}/{shape}"] = dict(prompt=f(p), prompt_cal=f(pc), countfree=f(cf),
                                           constant=bar, dphi75=f(d75), dphi85=f(d85))
        print(f"  {color + '/' + shape:>15} {f(p):7.2f} {f(pc):7.2f} {f(cf):10.2f} {bar:9.2f} "
              f"{f(d75):8.2f} {f(d85):8.2f}")
    allp = {(c, s) + k: v for c, s in cells for k, v in arm(c, s, "prompt").items()}
    alld = {(c, s) + k: v for c, s in cells for k, v in arm(c, s, "0.85").items()}
    allcf = {(c, s) + k: v for c, s in cells for k, v in arm(c, s, "countfree").items()}
    allpc = {(c, s) + k: v for c, s in cells for k, v in calibrated(arm(c, s, "prompt")).items()}
    g = lambda d: float(np.mean([abs(v - k[3]) for k, v in d.items()]))
    print(f"\n  {'POOLED':>15} {g(allp):7.2f} {g(allpc):7.2f} {g(allcf):10.2f} {bar:9.2f} "
          f"{g({(c, s) + k: v for c, s in cells for k, v in arm(c, s, '0.75').items()}):8.2f} "
          f"{g(alld):8.2f}")

    ks = sorted(set(allp) & set(alld))
    bb = np.array([abs(allp[k] - k[3]) for k in ks])
    ss = np.array([abs(alld[k] - k[3]) for k in ks])
    boot = np.array([(bb[i] - ss[i]).mean() for i in
                     (rng.integers(0, len(ks), len(ks)) for _ in range(10000))])
    ci = np.percentile(boot, [2.5, 97.5])
    print(f"\n  dphi@0.85 against the PROMPT baseline, paired over (cell, seed, count), n = {len(ks)}:")
    print(f"    improvement {bb.mean() - ss.mean():+.2f} objects  95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  "
          f"p = {(boot <= 0).mean():.4f}")
    pb = np.mean([allp[k] - k[3] for k in ks])
    print(f"    the prompt baseline's error is mostly bias: mean(counted - requested) = {pb:+.2f}")
    mu_p = [np.mean([allp[k] for k in ks if k[3] == t]) for t in T]
    mu_d = [np.mean([alld[k] for k in ks if k[3] == t]) for t in T]
    print(f"\n  per-count mean counted, requested  " + " ".join(f"{t:4d}" for t in T))
    print(f"  {'prompt':>33}  " + " ".join(f"{v:4.0f}" for v in mu_p))
    print(f"  {'dphi@0.85':>33}  " + " ".join(f"{v:4.0f}" for v in mu_d))
    json.dump(rowsout, open(f"{RUNS}/baselines/master.json", "w"), indent=1)
    print(f"\n  wrote {RUNS}/baselines/master.json")


if __name__ == "__main__":
    main()
