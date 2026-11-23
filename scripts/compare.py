"""Collect metrics.json from several evaluation directories into one comparison table (spec §8).

    python scripts/compare.py runs/eval_baseline runs/method_a/eval runs/method_b/eval runs/method_c/eval
"""
import argparse
import json
from pathlib import Path

COLUMNS = [
    ("overall", "exact_acc"), ("overall", "within1_acc"), ("overall", "mae"),
    ("heldout_shapes", "exact_acc"), ("train_shapes", "exact_acc"),
    (None, "seed_spearman_mean"), (None, "latency_ms_per_image"), (None, "diversity"),
    (None, "nn_regularity_cv"), (None, "clip_score"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    header = ["run"] + [f"{g}.{k}" if g else k for g, k in COLUMNS]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for d in args.dirs:
        m = json.loads((Path(d) / "metrics.json").read_text())
        cells = [d]
        for g, k in COLUMNS:
            v = (m.get(g, {}) if g else m).get(k)
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "–")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
