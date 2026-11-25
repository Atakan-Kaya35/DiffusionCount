"""Count calibration by prompt remapping: fit on some seeds, emit an evaluation on the others.

The frozen generator hears the count and draws about twice too many of them - fitted on held-out
seeds, `counted = 1.514 n + 0.069`, with a near-zero offset. So to obtain m objects, do not ask
for m: ask for the n whose output is closest to m. That is the whole method. It needs no training
and no sampler.

It is a GENERATION-side method - it changes the prompt that is issued, so the images really do
contain closer to the requested number - as distinct from a measurement-side correction, which
rescales the counted value afterwards and is only ever a ceiling.

The map is chosen to minimise expected |counted - target| on the FIT seeds and scored on the
others, so it cannot see the rows it is judged on. Granularity matters: `--by shape` (30 free
choices) beats `--by global` (10), while `--by color_shape` (170 free choices against ~680 fit
rows) overfits and does worse than both. That is measured, not assumed, and `--by` lets it be
re-measured.

The output is an ordinary evaluation directory - records.csv in the usual schema plus metrics.json
- so the normal tools read it unchanged:

    python scripts/remap_prompts.py runs/eval_baseline_white_s8 --out runs/remap_white
    python scripts/compare_paired.py runs/eval_baseline_white_s8 runs/remap_white
    python scripts/results_table.py runs/eval_baseline_white_s8 runs/remap_white --shapes

In that comparison the two sides share the fit/score split: only the scored seeds appear in the
remapped run, so only those pair.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401


def load(d: Path) -> list[dict]:
    with open(d / "records.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def group_key(r: dict, by: str) -> str:
    return {"global": "", "shape": r["shape"], "color_shape": f"{r['color']} {r['shape']}"}[by]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="baseline evaluation directory to calibrate from")
    ap.add_argument("--out", required=True, help="where to write the remapped evaluation")
    ap.add_argument("--by", default="shape", choices=["global", "shape", "color_shape"],
                    help="granularity of the map (default: shape, measured best)")
    ap.add_argument("--fit-seeds", default="even", choices=["even", "odd"],
                    help="which seeds fit the map; the others are scored")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    rows = load(src)
    for r in rows:
        r["seed"] = int(r["seed"])
        r["n"] = int(float(r["n"]))
        r["counted"] = float(r["counted"])
    counts = sorted({r["n"] for r in rows})
    fit_even = args.fit_seeds == "even"
    fit = [r for r in rows if (r["seed"] % 2 == 0) == fit_even]
    score = [r for r in rows if (r["seed"] % 2 == 0) != fit_even]
    if not fit or not score:
        raise SystemExit("need both even and odd seeds in the source evaluation")

    # ---- fit: for each group and target, the prompt minimising expected |counted - target|
    by_group: dict[str, dict[int, list[float]]] = {}
    for r in fit:
        by_group.setdefault(group_key(r, args.by), {}).setdefault(r["n"], []).append(r["counted"])
    table: dict[str, dict[int, int]] = {}
    for g, per_n in by_group.items():
        table[g] = {}
        for m in counts:
            table[g][m] = min(per_n, key=lambda n: float(np.mean(np.abs(np.array(per_n[n]) - m))))

    # ---- score: issuing prompt table[g][m] for target m gives that row's counted value
    lookup = {(r["color"], r["shape"], r["seed"], r["n"]): r for r in score}
    recs = []
    for r in score:
        g = group_key(r, args.by)
        target = r["n"]
        prompt_n = table.get(g, table.get("", {})).get(target)
        if prompt_n is None:
            continue
        src_row = lookup.get((r["color"], r["shape"], r["seed"], prompt_n))
        if src_row is None:
            continue
        recs.append({"color": r["color"], "shape": r["shape"],
                     "heldout_shape": r.get("heldout_shape", "True"), "seed": r["seed"],
                     "n": target, "counted": int(src_row["counted"]),
                     "regularity": src_row.get("regularity", ""), "clip": src_row.get("clip", "nan"),
                     "prompt_n": prompt_n})
    if not recs:
        raise SystemExit("no records produced")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "records.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    with open(out / "prompt_map.json", "w", encoding="utf-8") as fh:
        json.dump({"by": args.by, "fit_seeds": args.fit_seeds,
                   "map": {g: {str(m): n for m, n in t.items()} for g, t in table.items()}}, fh, indent=2)

    p = np.array([r["counted"] for r in recs], float)
    t = np.array([r["n"] for r in recs], float)
    ks = sorted(set(int(x) for x in t))
    bar = min(np.abs(t - k).mean() for k in range(min(ks), max(ks) + 1))
    mae = float(np.abs(p - t).mean())
    m = {"overall": {"n": len(p), "exact_acc": float((p == t).mean()),
                     "within1_acc": float((np.abs(p - t) <= 1).mean()), "mae": mae,
                     "rel_err": float((np.abs(p - t) / np.maximum(t, 1)).mean())},
         "note": f"prompt remapping by {args.by}, fitted on {args.fit_seeds} seeds of "
                 f"{src.name} and scored on the others"}
    for k in ks:
        m["overall"][f"exact_acc@{k}"] = float((p[t == k] == k).mean())
    m["heldout_shapes"] = dict(m["overall"])
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2)

    # baseline on the very same scored rows, for a like-for-like line
    bp = np.array([r["counted"] for r in score], float)
    bt = np.array([r["n"] for r in score], float)
    bmae = float(np.abs(bp - bt).mean())
    hi = t >= 4
    print(f"  source            {src}")
    print(f"  granularity       {args.by}  ({sum(len(v) for v in table.values())} free choices, "
          f"{len(fit)} fit rows)")
    print(f"  map               " + "  ".join(
        f"{g or 'all'}: " + "".join(str(table[g][m]) if table[g][m] < 10 else "X" for m in counts)
        for g in sorted(table)))
    print(f"\n  baseline, same scored rows   MAE {bmae:6.3f}   skill {(bar - bmae) / bar:+.3f}")
    print(f"  remapped                     MAE {mae:6.3f}   skill {(bar - mae) / bar:+.3f}   "
          f"exact {(p == t).mean():.3f}")
    print(f"  slope(n>=4) {np.polyfit(t[hi], p[hi], 1)[0]:+.3f}   "
          f"spread {np.mean([p[t == k].std() for k in ks]):.2f}   "
          f"counted mean {p.mean():.2f}   zero {100 * (p == 0).mean():.1f}%")
    print(f"\n  wrote {out}/records.csv, metrics.json, prompt_map.json")


if __name__ == "__main__":
    main()
