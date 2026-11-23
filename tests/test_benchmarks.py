"""Benchmark loaders (CoCoCount, D2D-style, JSONL), scoring logic, and the comparison script."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import tiny_cfg

from countsampler.data.benchmarks import (
    D2D_STYLE_CLASSES,
    BenchmarkItem,
    ObjectCount,
    build_d2d_style,
    load_benchmark,
    load_cococount,
    load_jsonl,
    save_jsonl,
)
from countsampler.data.prompts import pluralize
from countsampler.evaluation import ColorComponentsCounter, CountResult, detector_query, run_benchmark_evaluation
from countsampler.novel.shared import build_sampler

ROOT = Path(__file__).resolve().parents[1]

# First records of Make It Count's dataset/CoCoCount.json (verbatim format), plus a seed-only duplicate.
COCOCOUNT_SAMPLE = [
    {"prompt": "A photo of three ties on the ground", "object": "tie", "object_plural": "ties", "object_id": 27,
     "scene": "on the ground", "number": "three", "int_number": 3, "seed": 591907},
    {"prompt": "A photo of ten airplanes", "object": "airplane", "object_plural": "airplanes", "object_id": 4,
     "scene": "", "number": "ten", "int_number": 10, "seed": 669912},
    {"prompt": "A photo of seven birds", "object": "bird", "object_plural": "birds", "object_id": 14,
     "scene": "", "number": "seven", "int_number": 7, "seed": 847010},
    {"prompt": "A photo of ten airplanes", "object": "airplane", "object_plural": "airplanes", "object_id": 4,
     "scene": "", "number": "ten", "int_number": 10, "seed": 12},
]


class FixedCounter:
    """Fake detector that always reports the same count."""

    def __init__(self, n):
        self.n = n
        self.queries = []

    def count(self, image, color, shape):
        self.queries.append((color, shape))
        return CountResult(self.n)


def test_pluralize_irregular_and_multiword():
    assert [pluralize(w) for w in ("sheep", "wine glass", "bus", "teddy bear", "knife", "circle", "cross")] == \
        ["sheep", "wine glasses", "buses", "teddy bears", "knives", "circles", "crosses"]


def test_load_cococount(tmp_path):
    p = tmp_path / "CoCoCount.json"
    p.write_text(json.dumps(COCOCOUNT_SAMPLE))
    items = load_cococount(p)
    assert len(items) == 4
    assert items[0].prompt == "A photo of three ties on the ground"
    assert items[0].objects == [ObjectCount("tie", 3)] and items[0].seed == 591907
    assert items[0].meta["scene"] == "on the ground"
    assert len(load_cococount(p, dedupe=True)) == 3


def test_jsonl_roundtrip_and_shorthand(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text('{"prompt": "three dogs", "object": "dog", "count": 3, "seed": 5}\n\n'
                 '{"prompt": "two cats and a dog", "objects": [{"name": "cat", "count": 2}, {"name": "dog", "count": 1}]}\n')
    items = load_jsonl(p)
    assert items[0].objects == [ObjectCount("dog", 3)] and items[0].seed == 5
    assert items[1].total_count == 3 and items[1].seed is None
    save_jsonl(items, tmp_path / "c.jsonl")
    assert [it.to_json() for it in load_jsonl(tmp_path / "c.jsonl")] == [it.to_json() for it in items]


def test_d2d_style_sets_follow_the_paper_description():
    small, multi, large = (build_d2d_style(k) for k in ("small", "multi", "large"))
    assert len(D2D_STYLE_CLASSES) == len(set(D2D_STYLE_CLASSES)) == 40
    assert len(small) == len(multi) == len(large) == 400
    assert {it.objects[0].count for it in small} == set(range(1, 11))
    assert {it.objects[0].count for it in large} == set(range(11, 21))
    assert all(len(it.objects) == 2 and it.objects[0].name != it.objects[1].name for it in multi)
    assert all(1 <= o.count <= 9 for it in multi for o in it.objects)
    assert small[0].prompt == "A realistic photo of a scene with one car."
    assert [it.to_json() for it in build_d2d_style("multi")] == [it.to_json() for it in multi]  # deterministic


def test_committed_benchmark_files_match_builder():
    for kind in ("small", "multi", "large"):
        committed = load_jsonl(ROOT / "benchmarks" / f"d2d_style_{kind}.jsonl")
        assert [it.to_json() for it in committed] == [it.to_json() for it in build_d2d_style(kind)]


def test_load_benchmark_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no files here: D2D-style sets are built in memory, CoCoCount must be downloaded
    assert len(load_benchmark("d2d_small")) == 400
    with pytest.raises(FileNotFoundError, match="prepare_benchmarks"):
        load_benchmark("cococount")
    with pytest.raises(KeyError):
        load_benchmark("nope")


def test_counter_queries():
    assert detector_query(None, "sports ball") == "a photo of a sports ball"
    assert detector_query("red", "circle") == "a photo of a red circle"
    with pytest.raises(ValueError, match="needs a colour"):
        ColorComponentsCounter().count(__import__("numpy").zeros((8, 8, 3), dtype="uint8"), None, "dog")


def _items():
    return [
        BenchmarkItem("two dogs", [ObjectCount("dog", 2)]),                               # correct
        BenchmarkItem("three cats", [ObjectCount("cat", 3)]),                             # wrong
        BenchmarkItem("two dogs and two cats", [ObjectCount("dog", 2), ObjectCount("cat", 2)]),  # correct
        BenchmarkItem("two dogs and three cats", [ObjectCount("dog", 2), ObjectCount("cat", 3)]),  # wrong
        BenchmarkItem("twelve cars", [ObjectCount("car", 12)]),                           # wrong, out of range
    ]


def test_benchmark_scoring(tiny_gen, tmp_path):
    cfg = tiny_cfg("method_a.yaml", tmp_path)
    sampler = build_sampler(cfg.sampler, tiny_gen.latent_channels, tiny_gen.pooled_dim).to(tiny_gen.device)
    counter = FixedCounter(2)
    m = run_benchmark_evaluation(cfg, tiny_gen, sampler, counter, _items(), tmp_path / "b", benchmark_name="toy",
                                 seeds=2, save_images=True)
    assert m["num_images"] == 10 and m["image_exact_acc"] == pytest.approx(0.4)
    assert m["image_exact_acc_per_seed"] == {"0": 0.4, "1": 0.4} and m["image_exact_acc_std_over_seeds"] == 0.0
    assert m["object"]["exact_acc"] == pytest.approx(4 / 7)
    assert m["out_of_train_range"] == {"image_exact_acc": 0.0, "num_images": 2}
    assert m["multi_object"] == {"image_exact_acc": 0.5, "num_images": 4}
    assert all(color is None for color, _ in counter.queries)
    manifest = (tmp_path / "b" / "manifest.jsonl").read_text().splitlines()
    assert len(manifest) == 10 and (tmp_path / "b" / json.loads(manifest[0])["file"]).exists()


def test_benchmark_policies_and_record_seeds(tiny_gen, tmp_path):
    cfg = tiny_cfg("method_a.yaml", tmp_path)
    m = run_benchmark_evaluation(cfg, tiny_gen, None, FixedCounter(2), _items(), tmp_path / "skip",
                                 seeds=1, multi_object_count="skip")
    assert m["num_prompts"] == 3 and "multi_object" not in m
    seeded = [BenchmarkItem("two dogs", [ObjectCount("dog", 2)], seed=123)]
    m = run_benchmark_evaluation(cfg, tiny_gen, None, FixedCounter(2), seeded, tmp_path / "rec", seeds="record")
    assert m["image_exact_acc_per_seed"] == {"record": 1.0}
    with pytest.raises(ValueError, match="no seed"):
        run_benchmark_evaluation(cfg, tiny_gen, None, FixedCounter(2), _items(), tmp_path / "x", seeds="record")


def test_compare_groups_synthetic_and_benchmark_runs(tmp_path):
    (tmp_path / "syn").mkdir()
    (tmp_path / "syn" / "metrics.json").write_text(json.dumps({"overall": {"exact_acc": 0.25}}))
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "metrics.json").write_text(json.dumps(
        {"benchmark": "cococount", "image_exact_acc_mean_over_seeds": 0.5, "object": {"exact_acc": 0.5}}))
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare.py"), str(tmp_path / "syn"),
                          str(tmp_path / "bench")], capture_output=True, text=True, check=True).stdout
    assert "### synthetic prompts" in out and "### cococount" in out and "0.500" in out and "0.250" in out
