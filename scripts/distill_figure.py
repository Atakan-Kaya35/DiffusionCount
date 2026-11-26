"""The learned sampler's output, beside the untouched baseline.

Both rows use the SAME held-out seed and the SAME count-free prompt. The only difference is
that the lower row's noise passed through Dphi, so anything that changes between them is
carried by the noise and nothing else.
"""
import os
import sys

import torch
from PIL import Image, ImageDraw

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
os.chdir(WT)

from countsampler.data.prompts import PromptSampler                 # noqa: E402
from countsampler.evaluation import build_hard_counter              # noqa: E402
from countsampler.evaluation.hard_counters import count_images      # noqa: E402
from countsampler.generator import FrozenTurboGenerator             # noqa: E402
from countsampler.novel.shared import build_sampler                 # noqa: E402
from countsampler.utils.config import load_config                   # noqa: E402
from countsampler.utils.misc import resolve_device, seed_everything, to_uint8_images  # noqa: E402

FLAT = 'data.templates=["a flat lay photo of {count} {color} {shape} on a white background"]'
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/distill"
TARGETS = list(range(10, 30, 2))
CELLS = [("green", "diamond"), ("red", "diamond")]


def main():
    cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"],
                      [FLAT, "data.include_count=false"])
    seed_everything(0)
    dev = resolve_device(cfg.experiment.get("device", "auto"))
    gen = FrozenTurboGenerator.from_config(cfg.generator, dev)
    owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
    ps = PromptSampler(cfg.data)
    c, h, w = gen.latent_shape
    ck = torch.load(f"{OUT}/sampler.pt", weights_only=False)
    cb0 = gen.encode_prompts([ps.prompt(1, *ck["cells"][0])[0]]).pooled[0].float()
    net = build_sampler(ck["scfg"], c, cb0.shape[0]).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    eps = torch.randn(8, c, h, w, generator=torch.Generator().manual_seed(90210))

    S, PAD, LAB, HDR = 190, 6, 180, 38
    img = Image.new("RGB", (LAB + len(TARGETS) * (S + PAD), HDR + 2 * len(CELLS) * (S + PAD + 20) + 10), "white")
    d = ImageDraw.Draw(img)
    d.text((LAB, 6), "Learned sampler  z = proj(eps + w*Dphi(eps, n, c_bar)).  The count never appears "
                     "in the prompt - every column uses the identical text.", fill="black")
    d.text((LAB, 21), "Held-out seed, never used in training. Only the noise differs between the two rows.",
           fill=(90, 90, 90))
    row = 0
    for color, shape in CELLS:
        p = ps.prompt(1, color, shape)[0]
        cb = gen.encode_prompts([p]).pooled[0].float()
        for name in ("baseline: untouched noise", "learned sampler Dphi"):
            e = torch.stack([eps[0]] * len(TARGETS)).to(dev)
            if name.startswith("learned"):
                with torch.no_grad():
                    z = net(e, torch.tensor(TARGETS, device=dev), cb[None].expand(len(TARGETS), -1))
            else:
                z = e
            with torch.no_grad():
                u8 = to_uint8_images(gen.decode(gen.generate_latents(z, gen.encode_prompts([p] * len(TARGETS)))))
            cs = [r.count for r in count_images(owl, list(u8), color, shape)]
            y = HDR + row * (S + PAD + 20)
            d.text((8, y + S // 2 - 6), f"{color} {shape}", fill=(60, 60, 60))
            d.text((8, y + S // 2 + 8), name, fill="black")
            for k, t in enumerate(TARGETS):
                x = LAB + k * (S + PAD)
                img.paste(Image.fromarray(u8[k]).resize((S, S), Image.LANCZOS), (x, y))
                if row == 0:
                    d.text((x + S // 2 - 22, 24), f"ask {t}", fill="black")
                d.text((x + 4, y + S + 3), f"counted {cs[k]}",
                       fill=(0, 130, 0) if abs(cs[k] - t) <= 3 else
                            ((200, 120, 0) if abs(cs[k] - t) <= 6 else (190, 0, 0)))
            print(f"  {color} {shape} / {name}: {cs}", flush=True)
            row += 1
    img.save(f"{OUT}/comparison.png")
    print(f"  wrote {OUT}/comparison.png")


if __name__ == "__main__":
    main()
