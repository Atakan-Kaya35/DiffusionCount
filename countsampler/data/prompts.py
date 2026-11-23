"""Prompt templates p_n for counts n (spec §1, §7).

    build_prompt("{count} {color} {shape}", 3, "red", "circle")  -> ("three red circles", "circles")
    include_count=False (ablation §7)                             -> ("red circles", "circles")
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                "nineteen", "twenty"]

_IRREGULAR = {"cross": "crosses", "box": "boxes"}


def pluralize(noun: str) -> str:
    if noun in _IRREGULAR:
        return _IRREGULAR[noun]
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def count_phrase(n: int, fmt: str = "word") -> str:
    if fmt == "word" and 0 <= n < len(NUMBER_WORDS):
        return NUMBER_WORDS[n]
    return str(n)


def build_prompt(template: str, n: int, color: str, shape: str, count_format: str = "word",
                 include_count: bool = True) -> tuple[str, str]:
    """Returns (prompt, object_word). The object word is the token whose cross-attention is read."""
    noun = shape if (include_count and n == 1) else pluralize(shape)
    count = count_phrase(n, count_format) if include_count else ""
    text = template.format(count=count, color=color, shape=noun)
    text = re.sub(r"\s+", " ", text).strip()
    return text, noun


@dataclass
class PromptSet:
    """One base noise ε gets one (color, shape, template) and a set N of J counts."""

    color: str
    shape: str
    template: str
    label_count: int
    counts: list[int]


class PromptSampler:
    """Samples (color, shape, template) and the J-count set N for each base noise (spec §7 steps 1-2)."""

    def __init__(self, data_cfg, split: str = "train", seed: int = 0):
        self.cfg = data_cfg
        self.n_max = int(data_cfg.n_max)
        self.colors = list(data_cfg.colors)
        self.shapes = list(data_cfg.train_shapes if split == "train" else data_cfg.heldout_shapes)
        self.templates = list(data_cfg.templates)
        self.count_format = data_cfg.get("count_format", "word")
        self.include_count = bool(data_cfg.get("include_count", True))
        self.rng = random.Random(seed)

    def sample(self, j: int) -> PromptSet:
        if j > self.n_max:
            raise ValueError(f"counts_per_eps J={j} exceeds n_max={self.n_max}")
        label = self.rng.randint(1, self.n_max)
        others = self.rng.sample([n for n in range(1, self.n_max + 1) if n != label], j - 1)
        return PromptSet(color=self.rng.choice(self.colors), shape=self.rng.choice(self.shapes),
                         template=self.rng.choice(self.templates), label_count=label,
                         counts=[label] + others)

    def prompt(self, n: int, color: str, shape: str, template: str | None = None) -> tuple[str, str]:
        return build_prompt(template or self.templates[0], n, color, shape, self.count_format,
                            self.include_count)
