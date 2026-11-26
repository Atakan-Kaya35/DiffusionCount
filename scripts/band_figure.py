"""Side-by-side: untouched noise against the selected reference noise, same count-free prompt."""
import json, os, sys
import numpy as np, torch
WT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/.claude/worktrees/cs485-gpu-setup-5bb57e"
sys.path.insert(0, WT); os.chdir(WT)
from countsampler.data.prompts import PromptSampler
from countsampler.evaluation import build_hard_counter
from countsampler.evaluation.hard_counters import count_images
from countsampler.generator import FrozenTurboGenerator
from countsampler.utils.config import load_config
from countsampler.utils.misc import resolve_device, seed_everything, to_uint8_images
from PIL import Image, ImageDraw
FLAT = 'data.templates=["a flat lay photo of {count} {color} {shape} on a white background"]'
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/band"
T = list(range(10, 30, 2))
cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"], [FLAT, "data.include_count=false"])
seed_everything(int(cfg.experiment.seed))
gen = FrozenTurboGenerator.from_config(cfg.generator, resolve_device(cfg.experiment.get("device", "auto")))
owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
ps = PromptSampler(cfg.data)
c, h, w = gen.latent_shape
pool = torch.load(f"{OUT}/pool.pt", weights_only=False)["pool"]
refs = {int(k): v for k, v in json.load(open(f"{OUT}/refs.json")).items()}
eps = torch.randn(1, c, h, w, generator=torch.Generator().manual_seed(90210))[0]
color, shape = "red", "diamond"        # a cell where both counters agree
prompt = ps.prompt(1, color, shape)[0]
rowsz = {"baseline (same noise every time)": [eps] * len(T),
         "selected reference noise": [pool[refs[t]] for t in T]}
S, PAD, LAB = 200, 6, 150
img = Image.new("RGB", (LAB + len(T) * (S + PAD), 34 + 2 * (S + PAD + 22)), "white")
d = ImageDraw.Draw(img)
d.text((LAB, 8), f'count-free prompt, identical for every column:  "{prompt}"', fill="black")
for ri, (name, zl) in enumerate(rowsz.items()):
    z = torch.stack(zl).to(gen.device)
    with torch.no_grad():
        u8 = to_uint8_images(gen.decode(gen.generate_latents(z, gen.encode_prompts([prompt] * len(T)))))
    cs = [r.count for r in count_images(owl, list(u8), color, shape)]
    y = 34 + ri * (S + PAD + 22)
    d.text((8, y + S // 2), name, fill="black")
    for k, t in enumerate(T):
        x = LAB + k * (S + PAD)
        img.paste(Image.fromarray(u8[k]).resize((S, S), Image.LANCZOS), (x, y))
        if ri == 0:
            d.text((x + S // 2 - 24, 20), f"ask {t}", fill="black")
        d.text((x + 4, y + S + 4), f"counted {cs[k]}", fill=(180, 0, 0) if abs(cs[k] - t) > 4 else (0, 120, 0))
    print(f"  {name}: {cs}", flush=True)
img.save(f"{OUT}/comparison.png")
print(f"  wrote {OUT}/comparison.png")
