"""Datasets for training the density counter C_ψ (spec §4.1).

Two input spaces:
  * latent: synthetic images are VAE-encoded once into a cache (preferred by the spec, because
            the counter then reads x̂0 directly without decoding);
  * image:  scenes are rendered on the fly at the counter's input resolution.
Generator samples ("harvested" x̂0 latents with hard-counter labels) can be mixed into latent mode.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset

from ..utils.misc import gaussian_blur
from .shapes import ShapeRenderer, density_target, pil_to_tensor


def counter_categories(colors, shapes) -> list[str]:
    return [f"{c} {s}" for c in colors for s in shapes]


def heldout_combos(categories: list[str], n_max: int, fraction: float, seed: int) -> set[tuple[str, int]]:
    """Held-out (category, count) combinations, never seen while training the counter."""
    rng = random.Random(seed)
    combos = [(c, n) for c in categories for n in range(1, n_max + 1)]
    return set(rng.sample(combos, int(round(fraction * len(combos)))))


class CounterSceneSampler:
    """Chooses which categories / counts appear in each synthetic counter-training scene."""

    def __init__(self, categories: list[str], n_max: int, max_categories: int = 3, max_total: int = 20,
                 empty_prob: float = 0.05, touching_prob: float = 0.15, identical_prob: float = 0.2,
                 heldout: set | None = None, split: str = "train", heldout_share: float = 0.5):
        self.categories = list(categories)
        self.n_max = n_max
        self.max_categories = max_categories
        self.max_total = max_total
        self.empty_prob = empty_prob
        self.touching_prob = touching_prob
        self.identical_prob = identical_prob
        self.heldout = set(heldout or ())
        self.split = split
        self.heldout_share = heldout_share

    def sample(self, rng: random.Random):
        """Returns (spec [(color, shape, n)], touching probability, identical flag)."""
        touching = self.touching_prob if rng.random() < 0.5 else 0.0
        identical = rng.random() < self.identical_prob
        if rng.random() < self.empty_prob:
            return [], touching, identical
        n_cats = rng.randint(1, min(self.max_categories, len(self.categories)))
        cats = rng.sample(self.categories, n_cats)
        spec: list[tuple[str, int]] = []
        if self.split != "train" and self.heldout and rng.random() < self.heldout_share:
            cat, n = rng.choice(sorted(self.heldout))
            spec.append((cat, n))
            cats = [c for c in cats if c != cat][: n_cats - 1]
        for cat in cats:
            for _ in range(50):
                n = rng.randint(1, self.n_max)
                if self.split == "train" and (cat, n) in self.heldout:
                    continue
                spec.append((cat, n))
                break
        while len(spec) > 1 and sum(n for _, n in spec) > self.max_total:
            spec.pop()
        return [(c.split(" ", 1)[0], c.split(" ", 1)[1], n) for c, n in spec], touching, identical


def _object_counts(objects, category_index: dict[str, int]) -> torch.Tensor:
    counts = torch.zeros(len(category_index))
    for obj in objects:
        cat = obj.category if hasattr(obj, "category") else obj[0]
        if cat in category_index:
            counts[category_index[cat]] += 1
    return counts


class ImageCountingDataset(Dataset):
    """Scenes rendered on the fly (deterministic per index) at the counter input resolution."""

    def __init__(self, scene_sampler: CounterSceneSampler, renderer: ShapeRenderer, categories: list[str],
                 length: int, seed: int, density_sigma: float):
        self.sampler = scene_sampler
        self.renderer = renderer
        self.index = {c: i for i, c in enumerate(categories)}
        self.length = length
        self.seed = seed
        self.sigma = density_sigma

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        rng = random.Random(self.seed * 1_000_003 + i)
        spec, touching, identical = self.sampler.sample(rng)
        scene = self.renderer.render(rng, spec, touching=touching, identical=identical)
        img = pil_to_tensor(scene.image)
        dens = density_target(scene.objects, self.index, tuple(img.shape[-2:]), self.sigma)
        return img, dens, _object_counts(scene.objects, self.index)


@torch.no_grad()
def build_latent_cache(path: str | Path, scene_sampler: CounterSceneSampler, renderer: ShapeRenderer,
                       categories: list[str], num: int, seed: int, encode_fn, batch_size: int = 16,
                       image_aug: dict | None = None) -> None:
    """Render `num` scenes, (optionally) augment in image space, VAE-encode, save latents + objects."""
    from tqdm import tqdm

    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    latents, objects = [], []
    for start in tqdm(range(0, num, batch_size), desc=f"latent cache {Path(path).name}"):
        imgs = []
        for _ in range(min(batch_size, num - start)):
            spec, touching, identical = scene_sampler.sample(rng)
            scene = renderer.render(rng, spec, touching=touching, identical=identical)
            imgs.append(pil_to_tensor(scene.image))
            objects.append([(o.category, float(o.x), float(o.y)) for o in scene.objects])
        x = torch.stack(imgs)
        if image_aug:
            x = augment_batch(x, image_aug, gen).clamp(-1, 1)
        latents.append(encode_fn(x).half().cpu())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": torch.cat(latents), "objects": objects, "categories": list(categories)}, path)


class LatentCountingDataset(Dataset):
    """Cached latents (synthetic and/or harvested generator x̂0) with on-the-fly density targets."""

    def __init__(self, cache_path: str | Path, categories: list[str], density_sigma: float, flip: bool = True):
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        self.latents = data["latents"]
        self.objects = data["objects"]
        self.index = {c: i for i, c in enumerate(categories)}
        self.sigma = density_sigma
        self.flip = flip

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, i: int):
        lat = self.latents[i].float()
        objs = [tuple(o) for o in self.objects[i]]
        if self.flip and torch.rand(()) < 0.5:
            lat = lat.flip(-1)
            objs = [(c, 1.0 - x, y) for c, x, y in objs]
        if self.flip and torch.rand(()) < 0.5:
            lat = lat.flip(-2)
            objs = [(c, x, 1.0 - y) for c, x, y in objs]
        dens = density_target(objs, self.index, tuple(lat.shape[-2:]), self.sigma)
        return lat, dens, _object_counts(objs, self.index)


def concat_latent_datasets(paths, categories, density_sigma, flip=True) -> Dataset:
    return ConcatDataset([LatentCountingDataset(p, categories, density_sigma, flip) for p in paths])


def augment_batch(x: torch.Tensor, aug: dict, generator: torch.Generator | None = None) -> torch.Tensor:
    """Heavy augmentation (§4.1): contrast, Gaussian blur, Gaussian noise; per sample with prob `prob`.

    Applied to images (in [-1, 1]) or to latents. Blur and noise mimic the blurrier x̂0.
    """
    b = x.shape[0]
    dev = x.device

    def rand(*shape):
        return torch.rand(*shape, generator=generator).to(dev)

    apply = rand(b) < float(aug.get("prob", 0.8))
    lo, hi = aug.get("contrast", [0.8, 1.2])
    c = (lo + (hi - lo) * rand(b))[:, None, None, None]
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    y = (x - mean) * c + mean
    blur_max = float(aug.get("blur_sigma_max", 0.0))
    if blur_max > 0:
        levels = torch.round(rand(b) * 4) / 4 * blur_max  # 5 discrete blur levels
        for lvl in levels.unique():
            if lvl > 0:
                m = levels == lvl
                y[m] = gaussian_blur(y[m], float(lvl))
    noise_std = float(aug.get("noise_std", 0.0))
    if noise_std > 0:
        y = y + torch.randn(y.shape, generator=generator).to(dev) * (rand(b) * noise_std)[:, None, None, None]
    return torch.where(apply[:, None, None, None], y, x)
