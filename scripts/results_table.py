"""The results table for this project, with the columns that decide the question.

`scripts/compare.py` leads with MAE, which on this task is the most misleading number available:
the frozen generator over-counts, so a sampler that emits almost nothing scores well on MAE while
carrying no count information. Method C did exactly that — MAE 8.166 to 5.311 against the
baseline it was measured on, significant on a paired test, while producing 0.19 objects per image.

This leads with the quantities that separate counting from not counting:

    spread    sd of the counted value at fixed n, averaged over n. The ceiling analysis shows
              centring alone scores about -1.0, and skill first turns positive when the spread
              roughly halves - so this, not MAE, is what a method has to move.
    slope     response of counted to requested over n >= 4, PER SHAPE. The aggregate hides that
              cross is +1.91, diamond +0.52 and pentagon -0.65 on the frozen baseline.
    mean/zero what the run actually emitted. A degenerate run is obvious here and nowhere else.

    python scripts/results_table.py runs/eval_baseline_nms runs/d_white/eval ...
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401


def load(d: Path):
    rows = []
    with open(d / "records.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((int(float(r["counted"])), int(float(r["n"])), r["shape"], r["color"]))
    p = np.array([r[0] for r in rows], float)
    t = np.array([r[1] for r in rows])
    sh = np.array([r[2] for r in rows])
    return p, t, sh


def stats(p, t):
    ks = sorted(set(int(x) for x in t))
    bar = min(np.abs(t - k).mean() for k in range(min(ks), max(ks) + 1))
    mae = np.abs(p - t).mean()
    hi = t >= 4
    slope = np.polyfit(t[hi], p[hi], 1)[0] if hi.sum() > 2 and len(set(t[hi])) > 1 else float("nan")
    return dict(mae=mae, skill=(bar - mae) / bar, exact=(p == t).mean(),
                spread=float(np.mean([p[t == k].std() for k in ks])),
                slope=slope, mean=p.mean(), zero=100 * (p == 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--shapes", action="store_true", help="also break every run down by shape")
    args = ap.parse_args()

    runs = []
    for d in args.dirs:
        d = Path(d)
        try:
            runs.append((d.name if d.name != "eval" else d.parent.name, *load(d)))
        except OSError:
            print(f"(skipping {d}: no records.csv)")

    print("\n| run | MAE | skill | spread | slope n>=4 | counted mean | zero % | exact |")
    print("|---|---|---|---|---|---|---|---|")
    for name, p, t, _ in runs:
        s = stats(p, t)
        flag = ""
        if s["zero"] > 40 or s["mean"] < 0.5 * t.mean() or s["slope"] < 0.2:
            flag = " **⚠ degenerate**"
        print(f"| `{name}`{flag} | {s['mae']:.3f} | {s['skill']:+.3f} | {s['spread']:.2f} | "
              f"{s['slope']:+.3f} | {s['mean']:.2f} | {s['zero']:.1f} | {s['exact']:.3f} |")

    if args.shapes:
        for shape in sorted({x for _, _, _, sh in runs for x in sh}):
            print(f"\n**{shape}**\n")
            print("| run | MAE | skill | spread | slope n>=4 | counted mean |")
            print("|---|---|---|---|---|---|")
            for name, p, t, sh in runs:
                m = sh == shape
                if m.sum() < 10:
                    continue
                s = stats(p[m], t[m])
                print(f"| `{name}` | {s['mae']:.3f} | {s['skill']:+.3f} | {s['spread']:.2f} | "
                      f"{s['slope']:+.3f} | {s['mean']:.2f} |")

    print("\n_A lower MAE is not a better model. Read spread and slope first; a run marked "
          "degenerate emits too little to be counting at all._")


if __name__ == "__main__":
    main()
