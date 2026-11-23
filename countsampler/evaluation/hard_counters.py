"""Independent hard counters for evaluation (spec §4.2 Remark 3, §8).

They have zero or undefined gradient and are never used as a training signal.
  * ColorComponentsCounter: HSV colour mask → morphological opening → connected components.
    Built in and fast. Suited to the synthetic "n <colour> <shape>s" prompts. Touching objects
    merge into one component and are under-counted (a known limitation).
  * OwlV2Counter: open-vocabulary detector (transformers OWLv2) + NMS. Optional.
  * CountGD: not pip-installable; wrap its predictor in a class with the same `count` signature.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

HUE_RANGES = {  # degrees
    "red": [(0, 14), (340, 360)],
    "orange": [(14, 42)],
    "yellow": [(42, 72)],
    "green": [(75, 170)],
    "blue": [(190, 260)],
    "purple": [(260, 330)],
}


@dataclass
class CountResult:
    count: int
    centers: list[tuple[float, float]] = field(default_factory=list)  # normalised (x, y)
    areas: list[float] = field(default_factory=list)                  # fraction of image


def _to_uint8(image) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        x = image.detach().float().cpu()
        if x.ndim == 3 and x.shape[0] == 3:
            x = x.permute(1, 2, 0)
        return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
    return np.asarray(image, dtype=np.uint8)


def rgb_to_hsv(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = img.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = (60 * ((g - b)[rm] / d[rm]) + 360) % 360
    h[gm] = 60 * ((b - r)[gm] / d[gm]) + 120
    h[bm] = 60 * ((r - g)[bm] / d[bm]) + 240
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return h, s, mx


class ColorComponentsCounter:
    name = "color_components"

    def __init__(self, sat_min: float = 0.35, val_min: float = 0.25, min_area_frac: float = 5e-4,
                 open_iterations: int = 1):
        self.sat_min = sat_min
        self.val_min = val_min
        self.min_area_frac = min_area_frac
        self.open_iterations = open_iterations

    def mask(self, image, color: str) -> np.ndarray:
        img = _to_uint8(image)
        h, s, v = rgb_to_hsv(img)
        if color == "black":
            return v < 0.2
        if color not in HUE_RANGES:
            raise ValueError(f"no hue range for colour '{color}'")
        hue = np.zeros_like(h, dtype=bool)
        for lo, hi in HUE_RANGES[color]:
            hue |= (h >= lo) & (h < hi)
        return hue & (s >= self.sat_min) & (v >= self.val_min)

    def count(self, image, color: str, shape: str | None = None) -> CountResult:
        from scipy import ndimage

        m = self.mask(image, color)
        if self.open_iterations:
            m = ndimage.binary_opening(m, iterations=self.open_iterations)
        m = ndimage.binary_fill_holes(m)
        labels, num = ndimage.label(m)
        if num == 0:
            return CountResult(0)
        hgt, wid = m.shape
        areas = ndimage.sum(m, labels, index=np.arange(1, num + 1)) / (hgt * wid)
        coms = ndimage.center_of_mass(m, labels, index=np.arange(1, num + 1))
        keep = [i for i in range(num) if areas[i] >= self.min_area_frac]
        centers = [(float(coms[i][1] / wid), float(coms[i][0] / hgt)) for i in keep]
        return CountResult(len(keep), centers, [float(areas[i]) for i in keep])


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou: float) -> list[int]:
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        x1 = torch.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = torch.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = torch.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = torch.minimum(boxes[i, 3], boxes[rest, 3])
        inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        ious = inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-9)
        order = rest[ious <= iou]
    return keep


class OwlV2Counter:
    name = "owlv2"

    def __init__(self, model_id: str = "google/owlv2-base-patch16-ensemble", threshold: float = 0.3,
                 nms_iou: float = 0.5, device="cuda"):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()
        self.threshold = threshold
        self.nms_iou = nms_iou
        self.device = device

    @torch.no_grad()
    def count(self, image, color: str, shape: str | None = None) -> CountResult:
        from PIL import Image

        img = Image.fromarray(_to_uint8(image))
        inputs = self.processor(text=[[f"a photo of a {color} {shape}"]], images=img, return_tensors="pt")
        outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
        size = max(img.size)  # OWLv2 pads to a square
        res = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=torch.tensor([[size, size]], device=self.device))[0]
        boxes, scores = res["boxes"].cpu(), res["scores"].cpu()
        keep = _nms(boxes, scores, self.nms_iou) if len(scores) else []
        w, h = img.size
        centers = [(float((boxes[i, 0] + boxes[i, 2]) / 2 / w), float((boxes[i, 1] + boxes[i, 3]) / 2 / h))
                   for i in keep]
        return CountResult(len(keep), centers)


def build_hard_counter(name: str, **kwargs):
    if name in ("color", "color_components"):
        return ColorComponentsCounter(**kwargs)
    if name == "owlv2":
        return OwlV2Counter(**kwargs)
    raise KeyError(f"unknown hard counter '{name}' (color | owlv2)")
