import random

import torch

from countsampler.data import (
    CounterSceneSampler,
    ShapeRenderer,
    build_prompt,
    counter_categories,
    density_target,
    heldout_combos,
)
from countsampler.evaluation import ColorComponentsCounter
from countsampler.utils.config import load_config


def test_prompts():
    assert build_prompt("{count} {color} {shape}", 3, "red", "circle") == ("three red circles", "circles")
    assert build_prompt("{count} {color} {shape}", 1, "red", "cross") == ("one red cross", "cross")
    assert build_prompt("a photo of {count} {color} {shape}", 4, "red", "cross", include_count=False) == \
        ("a photo of red crosses", "crosses")
    assert build_prompt("{count} {color} {shape}", 12, "blue", "star", count_format="digit")[0] == "12 blue stars"


def test_render_and_hard_count_agree():
    renderer = ShapeRenderer(size=256)
    counter = ColorComponentsCounter()
    for n in (1, 4, 7, 10):
        scene = renderer.render(random.Random(n), [("red", "circle", n), ("blue", "square", 2)])
        assert counter.count(scene.image, "red").count == n
        assert counter.count(scene.image, "blue").count == 2


def test_density_target_sums_to_counts():
    renderer = ShapeRenderer(size=128)
    cats = counter_categories(["red", "blue"], ["circle", "star"])
    idx = {c: i for i, c in enumerate(cats)}
    scene = renderer.render(random.Random(0), [("red", "star", 3), ("blue", "circle", 5)], touching=0.5)
    d = density_target(scene.objects, idx, (32, 32), sigma=1.5)
    # categories: [red circle, red star, blue circle, blue star]; each Gaussian sums to exactly 1
    assert torch.allclose(d.sum((1, 2)), torch.tensor([0.0, 3.0, 5.0, 0.0]), atol=1e-4)


def test_heldout_combos_never_in_train():
    cats = counter_categories(["red", "blue"], ["circle", "star"])
    ho = heldout_combos(cats, 10, 0.3, seed=0)
    smp = CounterSceneSampler(cats, n_max=10, heldout=ho, split="train", empty_prob=0.0)
    rng = random.Random(0)
    for _ in range(300):
        spec, _, _ = smp.sample(rng)
        for c, s, n in spec:
            assert (f"{c} {s}", n) not in ho


def test_config_inheritance_and_overrides(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = load_config([root / "configs/method_a.yaml", root / "configs/gpu_8gb.yaml"],
                      ["train.steps=7", "method.spectral.K=32"])
    assert cfg.method.name == "A" and cfg.generator.model_id == "stabilityai/sd-turbo"
    assert cfg.train.steps == 7 and cfg.train.eps_per_batch == 1 and cfg.method.spectral.K == 32
