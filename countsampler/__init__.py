"""Count-Conditioned Noise Sampler (CS485 project), implementation of the
"Count-Conditioned Noise Sampler: Three Methodologies" spec (draft v5).

    countsampler/novel/      ★ research code (sampler, losses, counter, objectives)
    countsampler/generator/  frozen SD-Turbo / SDXL-Turbo wrapper + attention capture
    countsampler/data/       prompts, synthetic shapes, counter datasets
    countsampler/training/   sampler and counter training loops
    countsampler/evaluation/ hard counters, metrics, evaluation, pre-checks / offline scan
"""
__version__ = "0.1.0"
