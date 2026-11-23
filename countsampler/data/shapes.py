"""Synthetic basic-shape scenes with known counts (spec §7, §4.1).

Used for training and validating the counter C_ψ, for the offline pre-checks (§8, including
identical and touching shapes), and as ground truth for the evaluation counters.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw

COLOR_RGB = {
    "red": (215, 35, 35),
    "green": (35, 155, 60),
    "blue": (35, 75, 210),
    "yellow": (240, 205, 25),
    "orange": (245, 130, 25),
    "purple": (135, 50, 175),
    "black": (25, 25, 25),
}

SHAPES = ("circle", "square", "triangle", "star", "hexagon", "pentagon", "diamond", "cross")


def _regular_polygon(cx, cy, r, k, rot):
    return [(cx + r * math.cos(rot + 2 * math.pi * i / k), cy + r * math.sin(rot + 2 * math.pi * i / k))
            for i in range(k)]


def shape_outline(shape: str, cx: float, cy: float, r: float, rot: float):
    """Polygon vertices for `shape` with circumradius r, or None for a circle."""
    if shape == "circle":
        return None
    if shape == "square":
        return _regular_polygon(cx, cy, r, 4, rot + math.pi / 4)
    if shape == "diamond":
        return [(cx, cy - r), (cx + 0.65 * r, cy), (cx, cy + r), (cx - 0.65 * r, cy)]
    if shape == "triangle":
        return _regular_polygon(cx, cy, r, 3, rot - math.pi / 2)
    if shape == "pentagon":
        return _regular_polygon(cx, cy, r, 5, rot - math.pi / 2)
    if shape == "hexagon":
        return _regular_polygon(cx, cy, r, 6, rot)
    if shape == "star":
        pts = []
        for i in range(10):
            rr = r if i % 2 == 0 else 0.45 * r
            a = rot - math.pi / 2 + math.pi * i / 5
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        return pts
    if shape == "cross":
        t = 0.33 * r
        base = [(-t, -r), (t, -r), (t, -t), (r, -t), (r, t), (t, t), (t, r), (-t, r), (-t, t), (-r, t),
                (-r, -t), (-t, -t)]
        c, s = math.cos(rot), math.sin(rot)
        return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in base]
    raise ValueError(f"unknown shape '{shape}'")


@dataclass
class SceneObject:
    color: str
    shape: str
    x: float  # centre, normalised to [0, 1] (column)
    y: float  # centre, normalised to [0, 1] (row)
    r: float  # circumradius, normalised to image size

    @property
    def category(self) -> str:
        return f"{self.color} {self.shape}"


@dataclass
class Scene:
    image: Image.Image
    objects: list[SceneObject]


class ShapeRenderer:
    def __init__(self, size: int = 512, radius_range=(0.045, 0.09), min_gap: float = 0.01,
                 background: str = "white", color_jitter: int = 20, supersample: int = 2,
                 max_tries: int = 400):
        self.size = size
        self.radius_range = radius_range
        self.min_gap = min_gap
        self.background = background
        self.color_jitter = color_jitter
        self.supersample = supersample
        self.max_tries = max_tries

    def _bg(self, rng: random.Random):
        if self.background == "white":
            return (255, 255, 255)
        if self.background == "random_light":
            return tuple(rng.randint(215, 255) for _ in range(3))
        return tuple(self.background)

    def _place(self, rng: random.Random, radii: list[float], touching: float):
        """Non-overlapping centres (normalised). `touching` = prob. a shape is placed against another."""
        placed: list[tuple[float, float, float]] = []
        for r in radii:
            ok = False
            for _ in range(self.max_tries):
                if placed and rng.random() < touching:
                    px, py, pr = rng.choice(placed)
                    ang = rng.uniform(0, 2 * math.pi)
                    dist = pr + r + rng.uniform(0.0, 0.004)
                    x, y = px + dist * math.cos(ang), py + dist * math.sin(ang)
                    gap = -0.002
                else:
                    x, y = rng.uniform(r, 1 - r), rng.uniform(r, 1 - r)
                    gap = self.min_gap
                if not (r <= x <= 1 - r and r <= y <= 1 - r):
                    continue
                if all(math.hypot(x - qx, y - qy) >= r + qr + gap for qx, qy, qr in placed):
                    placed.append((x, y, r))
                    ok = True
                    break
            if not ok:
                return None
        return placed

    def render(self, rng: random.Random, spec: list[tuple[str, str, int]], touching: float = 0.0,
               identical: bool = False) -> Scene:
        """spec = [(color, shape, count), ...]. `identical`: one size/rotation per category."""
        lo, hi = self.radius_range
        scale = 1.0
        for _attempt in range(30):
            radii, meta = [], []
            for color, shape, count in spec:
                r_cat, rot_cat = rng.uniform(lo, hi) * scale, rng.uniform(0, 2 * math.pi)
                for _ in range(count):
                    r = r_cat if identical else rng.uniform(lo, hi) * scale
                    rot = rot_cat if identical else rng.uniform(0, 2 * math.pi)
                    radii.append(r)
                    meta.append((color, shape, rot))
            order = list(range(len(radii)))
            rng.shuffle(order)
            placed = self._place(rng, [radii[i] for i in order], touching)
            if placed is not None:
                break
            scale *= 0.9
        else:
            raise RuntimeError(f"could not place scene {spec}")

        s = self.size * self.supersample
        img = Image.new("RGB", (s, s), self._bg(rng))
        draw = ImageDraw.Draw(img)
        objects = []
        for k, i in enumerate(order):
            color, shape, rot = meta[i]
            x, y, r = placed[k]
            base = COLOR_RGB[color]
            j = self.color_jitter
            fill = tuple(int(min(255, max(0, c + rng.randint(-j, j)))) for c in base)
            outline = shape_outline(shape, x * s, y * s, r * s, rot)
            if outline is None:
                draw.ellipse([x * s - r * s, y * s - r * s, x * s + r * s, y * s + r * s], fill=fill)
            else:
                draw.polygon(outline, fill=fill)
            objects.append(SceneObject(color=color, shape=shape, x=x, y=y, r=r))
        if self.supersample > 1:
            img = img.resize((self.size, self.size), Image.LANCZOS)
        return Scene(image=img, objects=objects)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor [3, H, W] in [-1, 1]."""
    arr = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float()
    return arr / 127.5 - 1.0


def density_target(objects, category_index: dict[str, int], out_hw: tuple[int, int],
                   sigma: float) -> torch.Tensor:
    """[K, h, w] density: one 2-D Gaussian per object, each normalised to sum to exactly 1 (§4.1)."""
    h, w = out_hw
    k = len(category_index)
    target = torch.zeros(k, h, w)
    ys = torch.arange(h, dtype=torch.float32)[:, None] + 0.5
    xs = torch.arange(w, dtype=torch.float32)[None, :] + 0.5
    for obj in objects:
        cat = obj.category if hasattr(obj, "category") else obj[0]
        if cat not in category_index:
            continue
        ox, oy = (obj.x, obj.y) if hasattr(obj, "x") else (obj[1], obj[2])
        g = torch.exp(-((xs - ox * w) ** 2 + (ys - oy * h) ** 2) / (2 * sigma ** 2))
        target[category_index[cat]] += g / g.sum()
    return target
