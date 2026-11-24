"""Collect metrics.json from several evaluation directories into comparison tables (spec §8).
Synthetic-prompt evaluations and each benchmark get their own table.

    python scripts/compare.py runs/eval_baseline runs/method_a/eval runs/method_b/eval runs/method_c/eval
    python scripts/compare.py runs/eval_baseline_cococount runs/method_a/eval_cococount
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

SYNTHETIC_COLUMNS = [
    ("overall", "exact_acc"), ("overall", "within1_acc"), ("overall", "mae"),
    ("heldout_shapes", "exact_acc"), ("train_shapes", "exact_acc"),
    (None, "seed_spearman_mean"), (None, "latency_ms_per_image"), (None, "diversity"),
    (None, "nn_regularity_cv"), (None, "clip_score"),
]
BENCHMARK_COLUMNS = [
    (None, "image_exact_acc_mean_over_seeds"), (None, "image_exact_acc_std_over_seeds"),
    ("object", "exact_acc"), ("object", "mae"), ("in_train_range", "image_exact_acc"),
    ("out_of_train_range", "image_exact_acc"), ("multi_object", "image_exact_acc"),
    (None, "latency_ms_per_image"), (None, "clip_score"),
]


def table(title: str, rows: list[tuple[str, dict]], columns) -> None:
    header = ["run"] + [f"{g}.{k}" if g else k for g, k in columns]
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for name, m in rows:
        cells = [name]
        for g, k in columns:
            v = (m.get(g, {}) if g else m).get(k)
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "–")
        print("| " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    groups = defaultdict(list)
    for d in args.dirs:
        m = json.loads((Path(d) / "metrics.json").read_text(encoding="utf-8"))
        groups[m.get("benchmark", "synthetic prompts")].append((d, m))
    for kind, rows in groups.items():
        table(kind, rows, SYNTHETIC_COLUMNS if kind == "synthetic prompts" else BENCHMARK_COLUMNS)


if __name__ == "__main__":
    main()
