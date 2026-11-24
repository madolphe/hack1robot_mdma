"""Sana-Sprint sur Mac (Apple Silicon, MPS) avec diffusers : un prompt -> une image, en 1 à 4 pas.

    python txt2img_sana.py "A surreal dreamlike oil painting, a visual metaphor of silence"
    python txt2img_sana.py "..." --small           # transformer 0.6B au lieu de 1.6B
    python txt2img_sana.py "..." --dtype fp16      # fp16 natif sur M1 (bf16 y est émulé)

Composants : encodeur de texte Gemma-2-2B (~5 Go), transformer 1.6B (~3 Go) ou 0.6B (~1,2 Go),
autoencodeur DC-AE (compression x32, d'où très peu de tokens par image).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# le client Xet de huggingface_hub reste bloqué à 0 octet sur cette machine : HTTP classique
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from diffusers import SanaSprintPipeline, SanaTransformer2DModel

REPOS = {
    "1.6B": "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    "0.6B": "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("-o", "--out", default=None, help="fichier PNG (défaut : out/sana-<horodatage>.png)")
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--steps", type=int, default=2, help="1 à 4 (Sprint est distillé pour 2)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--small", action="store_true", help="transformer 0.6B")
    ap.add_argument("--dtype", choices=DTYPES, default="bf16")
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = DTYPES[a.dtype]
    size = "0.6B" if a.small else "1.6B"
    print(f"device={device} dtype={a.dtype} transformer={size}")

    t0 = time.time()
    kw = {}
    if a.small:
        # encodeur de texte et VAE identiques dans les deux dépôts : seul le transformer 0.6B est téléchargé
        kw["transformer"] = SanaTransformer2DModel.from_pretrained(REPOS["0.6B"], subfolder="transformer", torch_dtype=dtype)
    pipe = SanaSprintPipeline.from_pretrained(REPOS["1.6B"], torch_dtype=dtype, **kw).to(device)
    t1 = time.time()
    print(f"modèle chargé en {t1 - t0:.0f} s")

    def sync(_pipe, _step, _t, kw):
        # MPS est asynchrone : sans synchronisation la barre finit tout de suite et le calcul arrive après
        if device == "mps":
            torch.mps.synchronize()
        return kw

    img = pipe(
        prompt=a.prompt, width=a.width, height=a.height, num_inference_steps=a.steps,
        generator=torch.Generator("cpu").manual_seed(a.seed),
        callback_on_step_end=sync,
    ).images[0]
    t2 = time.time()

    out = Path(a.out) if a.out else Path(__file__).parent / "out" / f"sana-{time.strftime('%Y%m%d-%H%M%S')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"image générée en {t2 - t1:.0f} s -> {out}")
    if device == "mps":
        print(f"mémoire MPS max allouée par le driver : {torch.mps.driver_allocated_memory() / 1e9:.1f} Go")


if __name__ == "__main__":
    main()
