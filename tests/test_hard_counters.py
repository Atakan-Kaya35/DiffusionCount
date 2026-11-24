"""Batched counting must be a throughput change only, never a results change.

Evaluation spends most of its time in the hard counter, one forward per image. `count_images`
routes to a counter's batched path when it has one. If batching ever altered a count, every
comparison against an evaluation computed the slow way would be invalid, so the equivalence is
pinned here rather than assumed.
"""
import numpy as np
import pytest

from countsampler.evaluation import ColorComponentsCounter, CountResult, count_images


def _scene(circles, size=96, color=(0, 0, 220)):
    """White canvas with filled discs at the given (x, y, r), in BGR-ish uint8 RGB."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    for cx, cy, r in circles:
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = color
    return img


SCENES = [
    _scene([(24, 24, 10)]),
    _scene([(24, 24, 10), (70, 70, 10)]),
    _scene([(20, 20, 9), (70, 20, 9), (45, 70, 9)]),
    _scene([]),
]


def test_count_images_matches_the_per_image_loop():
    counter = ColorComponentsCounter()
    batched = count_images(counter, SCENES, "blue", "circle")
    looped = [counter.count(s, "blue", "circle") for s in SCENES]
    assert [r.count for r in batched] == [r.count for r in looped]
    for b, l in zip(batched, looped):
        assert len(b.centers) == len(l.centers)
        for cb, cl in zip(b.centers, l.centers):
            assert cb == pytest.approx(cl, abs=1e-9)


def test_count_images_preserves_order_and_uses_the_batched_path():
    calls = {"batch": 0, "single": 0}

    class Stub:
        def count(self, image, color, shape=None):
            calls["single"] += 1
            return CountResult(int(image[0, 0, 0]))

        def count_batch(self, images, color, shape=None):
            calls["batch"] += 1
            return [CountResult(int(im[0, 0, 0])) for im in images]

    imgs = [np.full((2, 2, 3), i, dtype=np.uint8) for i in (3, 1, 4, 1, 5)]
    out = count_images(Stub(), imgs, "red", "circle")
    assert [r.count for r in out] == [3, 1, 4, 1, 5]      # order preserved
    assert calls == {"batch": 1, "single": 0}             # batched path taken, once


def test_count_images_falls_back_when_there_is_no_batched_path():
    class OnlySingle:
        def count(self, image, color, shape=None):
            return CountResult(int(image[0, 0, 0]))

    imgs = [np.full((2, 2, 3), i, dtype=np.uint8) for i in (7, 8)]
    assert [r.count for r in count_images(OnlySingle(), imgs, "red", "circle")] == [7, 8]


def test_count_images_handles_an_empty_batch():
    assert count_images(ColorComponentsCounter(), [], "blue", "circle") == []


def test_count_images_halves_the_batch_on_out_of_memory():
    """An OOM mid-evaluation must degrade to smaller batches, not lose the run."""
    import torch

    seen = []

    class Flaky:
        def count_batch(self, images, color, shape=None):
            seen.append(len(images))
            if len(images) > 2:
                raise torch.cuda.OutOfMemoryError("simulated")
            return [CountResult(int(im[0, 0, 0])) for im in images]

    imgs = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(8)]
    out = count_images(Flaky(), imgs, "red", "circle")
    assert [r.count for r in out] == list(range(8))   # order and values survive the retries
    assert 8 in seen and 2 in seen                    # tried big, settled smaller
