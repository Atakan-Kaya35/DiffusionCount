"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  ★ NOVEL ★  OBJECTIVE ASSEMBLY (spec Eq. 15, 17, 22) and stage schedules         ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  L_method = Σ_{active terms} λ_term · L_term                                     ║
║  Method terms are averaged over the unrolled steps (§9 "each iteration").        ║
║  Shared terms (L_anc, L_pri, optionally L_ord) are computed across the J noises  ║
║  of each ε (§7 step 5). Inactive terms are still computed and logged, but they   ║
║  get no gradient weight.                                                         ║
╚══════════════════════════════════════════════════════════════════════════════════╝

To add a new method: subclass `CountMethod`, set `name` and `TERMS`, implement
`method_terms` (and `attention_request` if it reads attention), then register it in
novel/methods/__init__.py.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from ...generator.attention import AttentionRequest
from ...generator.turbo import FrozenTurboGenerator, StepOutput
from ..shared import GeometryLosses


@dataclass
class Stage:
    start: int
    terms: tuple[str, ...]


@dataclass
class FlatBatch:
    """One entry per (ε, n) pair; flat index = eps_index * J + j."""

    prompts: list[str]
    object_words: list[str]
    colors: list[str]
    shapes: list[str]
    counts: torch.Tensor      # [B] long
    eps_index: torch.Tensor   # [B] long, which base noise


def parse_stages(raw) -> list[Stage]:
    stages = sorted((Stage(int(s["start"]), tuple(s["terms"])) for s in raw), key=lambda s: s.start)
    if not stages or stages[0].start != 0:
        raise ValueError("method.stages must contain a stage starting at step 0")
    return stages


class CountMethod:
    name: str = "base"
    TERMS: tuple[str, ...] = ()                  # method-specific term names
    # "x0a" is the output anchor: a hinge on how far x̂0 moved, available to every method.
    # geometry.delta constrains ||z - eps||, which was measured not to control whether the decode
    # is a scene - a random displacement at the full 0.3 budget leaves the image essentially
    # unchanged while a learned one destroys it at 0.08. Methods C, D and B have each collapsed
    # the generator while geom/d_anchor_* sat comfortably inside budget.
    SHARED_TERMS: tuple[str, ...] = ("anc", "pri", "ord", "x0a")

    def __init__(self, cfg, generator: FrozenTurboGenerator):
        self.cfg = cfg
        self.generator = generator
        mcfg = cfg.method
        self.weights = {k: float(v) for k, v in mcfg.weights.items()}
        self.stages = parse_stages(mcfg.stages)
        known = set(self.TERMS) | set(self.SHARED_TERMS)
        for st in self.stages:
            unknown = set(st.terms) - known
            if unknown:
                raise ValueError(f"method {self.name}: unknown terms {unknown} in stage {st}")
            missing_w = set(st.terms) - set(self.weights)
            if missing_w:
                raise ValueError(f"method {self.name}: no weight for terms {missing_w}")
        g = cfg.geometry
        self.geometry = GeometryLosses(delta=g.delta, g=g.g, n_max=cfg.data.n_max,
                                       ord_form=g.get("ord_form", "regression"),
                                       rank_margin=g.get("rank_margin", 0.05),
                                       delta_x=float(g.get("delta_x", 0.15)),
                                       g_power=float(g.get("g_power", 0.7)))
        self.loss_steps = mcfg.get("loss_steps", "all")

    # ------------------------------------------------------------------ to override
    def attention_request(self) -> AttentionRequest | None:
        return None

    @property
    def needs_attention(self) -> bool:
        req = self.attention_request()
        return req is not None and not req.is_empty()

    def method_terms(self, out: StepOutput, batch: FlatBatch, step: int
                     ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        raise NotImplementedError

    # ------------------------------------------------------------------ shared logic
    def active_terms(self, step: int) -> tuple[str, ...]:
        active = self.stages[0].terms
        for st in self.stages:
            if step >= st.start:
                active = st.terms
        return active

    def _selected_steps(self, outs: list[StepOutput]) -> list[StepOutput]:
        if self.loss_steps == "all":
            return outs
        if self.loss_steps == "last":
            return outs[-1:]
        return [outs[i] for i in self.loss_steps]

    def compute(self, step: int, *, eps: torch.Tensor, z_geom: torch.Tensor, counts_geom: torch.Tensor,
                batch: FlatBatch, outs: list[StepOutput],
                x0_ref: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        """Total loss and logs.

        eps: [Be, C, H, W]. z_geom: [Be, J', C, H, W] with counts_geom [Be, J'] (J' = J, or N_max
        when geometry.over = all_counts). outs: generator unroll for the flat batch.
        """
        active = self.active_terms(step)
        terms: dict[str, torch.Tensor] = {}
        logs: dict[str, float] = {}

        acc: dict[str, list[torch.Tensor]] = defaultdict(list)
        selected = self._selected_steps(outs)
        for out in selected:
            vals, step_logs = self.method_terms(out, batch, step)
            for k, v in vals.items():
                acc[k].append(v)
            suffix = f"@t{out.t}" if len(selected) > 1 else ""
            logs.update({f"{k}{suffix}": v for k, v in step_logs.items()})
        for k, vs in acc.items():
            terms[k] = torch.stack(vs).mean()

        terms["anc"] = self.geometry.anchor_loss(z_geom, eps)
        terms["pri"] = self.geometry.prior_loss(z_geom)
        terms["ord"] = self.geometry.order_loss(z_geom, counts_geom)
        logs.update(self.geometry.diagnostics(z_geom, eps))
        if x0_ref is not None:
            x0 = outs[-1].x0_hat
            terms["x0a"] = self.geometry.x0_anchor_loss(x0, x0_ref, eps.shape[0])
            logs.update(self.geometry.x0_diagnostics(x0, x0_ref))

        total = sum(self.weights[k] * terms[k] for k in active)
        for k, v in terms.items():
            logs[f"loss/{k}"] = float(v.detach())
        logs["loss/total"] = float(total.detach())
        logs["stage"] = sum(step >= st.start for st in self.stages)
        return total, logs
