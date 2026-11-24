"""Counting benchmarks for spec §8: "each on CoCoCount and the D2D benchmarks".

  CoCoCount (Make It Count, Binyamin et al., CVPR 2025): 200 prompts "A photo of {number} {objects}
      [{scene}]" over 20 COCO classes, counts 2, 3, 4, 5, 7, 10. The JSON ships in
      github.com/Litalby1/make-it-count (dataset/CoCoCount.json). That repo states no licence, so
      the file is downloaded by scripts/prepare_benchmarks.py rather than committed here.
  D2D-Small / D2D-Multi / D2D-Large (D2D, Yoo et al., arXiv 2510.19278, §4.1): the authors' files
      are not public yet, so `build_d2d_style` RECONSTRUCTS them from the paper's description:
        small: 400 prompts, 40 countable COCO classes x counts 1-10,
               "A realistic photo of a scene with [count] [object]."
        multi: 400 prompts with two different classes, N1, N2 < 10 (we use 1-9),
               "A realistic photo of a scene with [count] [object] and [count] [object]."
        large: "similarly constructed with counts 11-20" (we read this as D2D-Small with counts 11-20).
      The exact 40-class list is not published. D2D_STYLE_CLASSES is our choice (the 20 CoCoCount
      classes plus 20 other countable COCO classes). Report these as "D2D-style", not as the
      official benchmark numbers.
  jsonl: any custom set, one JSON object per line (see `load_jsonl`).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .prompts import count_phrase, pluralize

COCOCOUNT_URL = "https://raw.githubusercontent.com/Litalby1/make-it-count/main/dataset/CoCoCount.json"
DEFAULT_PATHS = {
    "cococount": "data_cache/benchmarks/CoCoCount.json",
    "d2d_small": "benchmarks/d2d_style_small.jsonl",
    "d2d_multi": "benchmarks/d2d_style_multi.jsonl",
    "d2d_large": "benchmarks/d2d_style_large.jsonl",
}
BENCHMARKS = ("cococount", "d2d_small", "d2d_multi", "d2d_large", "jsonl")

# The 20 CoCoCount classes (COCO names) + 20 further countable COCO classes (our choice, see module doc).
COCOCOUNT_CLASSES = ["car", "airplane", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                     "backpack", "tie", "sports ball", "baseball glove", "cup", "bowl", "apple", "donut",
                     "cell phone", "clock"]
D2D_STYLE_CLASSES = COCOCOUNT_CLASSES + ["bicycle", "motorcycle", "bus", "truck", "boat", "traffic light",
                                         "bench", "zebra", "giraffe", "umbrella", "suitcase", "kite", "bottle",
                                         "wine glass", "banana", "orange", "chair", "laptop", "book", "teddy bear"]


@dataclass
class ObjectCount:
    name: str    # class name, also the detector query (COCO name for YOLO)
    count: int


@dataclass
class BenchmarkItem:
    prompt: str
    objects: list[ObjectCount]
    seed: int | None = None            # per-record seed (CoCoCount ships one)
    meta: dict = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        return sum(o.count for o in self.objects)

    def to_json(self) -> dict:
        d = {"prompt": self.prompt, "objects": [{"name": o.name, "count": o.count} for o in self.objects]}
        if self.seed is not None:
            d["seed"] = self.seed
        if self.meta:
            d["meta"] = self.meta
        return d


def load_cococount(path: str | Path, dedupe: bool = False) -> list[BenchmarkItem]:
    """Make It Count's CoCoCount.json: a list of {prompt, object, object_plural, object_id, scene,
    number, int_number, seed}. `object` is the COCO class name. With `dedupe`, prompts that differ only
    by seed are kept once (161 distinct prompts in the release, per later work)."""
    items, seen = [], set()
    for r in json.loads(Path(path).read_text(encoding="utf-8")):
        if dedupe and r["prompt"] in seen:
            continue
        seen.add(r["prompt"])
        items.append(BenchmarkItem(prompt=r["prompt"], objects=[ObjectCount(r["object"], int(r["int_number"]))],
                                   seed=int(r["seed"]) if r.get("seed") is not None else None,
                                   meta={"scene": r.get("scene", ""), "object_id": r.get("object_id")}))
    return items


def load_jsonl(path: str | Path) -> list[BenchmarkItem]:
    """One JSON object per line, either
         {"prompt": ..., "objects": [{"name": "dog", "count": 3}, ...], "seed": 7}
      or the single-object shorthand {"prompt": ..., "object": "dog", "count": 3}. `seed` is optional."""
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        objs = r.get("objects") or [{"name": r["object"], "count": r["count"]}]
        items.append(BenchmarkItem(prompt=r["prompt"], objects=[ObjectCount(o["name"], int(o["count"])) for o in objs],
                                   seed=r.get("seed"), meta=r.get("meta", {})))
    return items


def save_jsonl(items: list[BenchmarkItem], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(it.to_json()) + "\n" for it in items), encoding="utf-8")


def _phrase(n: int, name: str, count_format: str) -> str:
    return f"{count_phrase(n, count_format)} {name if n == 1 else pluralize(name)}"


def build_d2d_style(kind: str, classes: list[str] | None = None, count_format: str = "word",
                    seed: int = 0) -> list[BenchmarkItem]:
    """Reconstruction of D2D-Small / -Multi / -Large from the paper's description (see module doc)."""
    classes = list(classes or D2D_STYLE_CLASSES)
    if kind in ("small", "large"):
        counts = range(1, 11) if kind == "small" else range(11, 21)
        return [BenchmarkItem(f"A realistic photo of a scene with {_phrase(n, c, count_format)}.",
                              [ObjectCount(c, n)], meta={"benchmark": f"d2d_style_{kind}"})
                for c in classes for n in counts]
    if kind == "multi":
        rng = random.Random(seed)
        items = []
        for _ in range(400):
            a, b = rng.sample(classes, 2)
            na, nb = rng.randint(1, 9), rng.randint(1, 9)
            items.append(BenchmarkItem(
                f"A realistic photo of a scene with {_phrase(na, a, count_format)} and {_phrase(nb, b, count_format)}.",
                [ObjectCount(a, na), ObjectCount(b, nb)], meta={"benchmark": "d2d_style_multi"}))
        return items
    raise ValueError("kind must be 'small', 'multi' or 'large'")


def load_benchmark(name: str, path: str | Path | None = None, dedupe: bool = False) -> list[BenchmarkItem]:
    """Load a benchmark by name. D2D-style sets fall back to building in memory if their file is missing."""
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark '{name}', choose from {BENCHMARKS}")
    path = Path(path) if path else (Path(DEFAULT_PATHS[name]) if name in DEFAULT_PATHS else None)
    if name == "jsonl":
        if path is None:
            raise ValueError("the jsonl benchmark needs a path")
        return load_jsonl(path)
    if name == "cococount":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python scripts/prepare_benchmarks.py")
        return load_cococount(path, dedupe=dedupe)
    if path.exists():
        return load_jsonl(path)
    return build_d2d_style(name.split("_", 1)[1])
