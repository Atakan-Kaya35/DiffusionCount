"""★ NOVEL ★ Per-method objectives (loss weighting and stage schedules). Registry below."""
from .base import CountMethod, FlatBatch, Stage, parse_stages
from .method_a import SpectralCountMethod
from .method_b import DensityCountMethod
from .method_c import PeakSeparationMethod
from .method_d import ComponentCountMethod

METHODS = {
    "A": SpectralCountMethod,
    "B": DensityCountMethod,
    "C": PeakSeparationMethod,
    "D": ComponentCountMethod,
}


def build_method(cfg, generator, **kwargs) -> CountMethod:
    name = str(cfg.method.name).upper()
    if name not in METHODS:
        raise KeyError(f"unknown method '{name}', choose from {sorted(METHODS)}")
    return METHODS[name](cfg, generator, **kwargs)
