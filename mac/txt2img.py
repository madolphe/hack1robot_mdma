"""Z-Image-Turbo sur Mac (Apple Silicon, MPS) avec diffusers : un prompt -> une image.

    python txt2img.py "A surreal dreamlike oil painting, a visual metaphor of silence"
    python txt2img.py "..." --gguf            # transformer quantifié Q8 (~7 Go au lieu de ~12)

Deux phases pour limiter le pic mémoire (mémoire unifiée) :
  1. l'encodeur de texte (Qwen3-4B, ~8 Go) encode le prompt, puis il est libéré ;
  2. le transformer (6B) + le VAE débruitent et décodent.
"""
from __future__ import annotations

import argparse
import gc
import os
import subprocess
import time
from pathlib import Path

# le client Xet de huggingface_hub reste bloqué à 0 octet sur cette machine : HTTP classique
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from diffusers import ZImagePipeline

REPO = "Tongyi-MAI/Z-Image-Turbo"
GGUF_REPO = "unsloth/Z-Image-Turbo-GGUF"


def free():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("-o", "--out", default=None, help="fichier PNG (défaut : out/<horodatage>.png)")
    ap.add_argument("--width", type=int, default=1344)   # comme configs/default.yaml
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gguf", nargs="?", const="z-image-turbo-Q8_0.gguf", default=None,
                    help=f"fichier GGUF du transformer dans {GGUF_REPO} (auto si RAM <= 24 Go)")
    ap.add_argument("--bf16", action="store_true", help="forcer le transformer bf16 complet (~12 Go)")
    a = ap.parse_args()
    ram_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30
    if a.gguf is None and not a.bf16 and ram_gb <= 24:
        a.gguf = "z-image-turbo-Q8_0.gguf"
        print(f"{ram_gb:.0f} Go de RAM : transformer GGUF {a.gguf} (--bf16 pour forcer le modèle complet)")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"device={device} torch={torch.__version__}")
    t0 = time.time()

    # 1. encodage du prompt, sans charger transformer ni VAE
    enc = ZImagePipeline.from_pretrained(REPO, transformer=None, vae=None, torch_dtype=dtype).to(device)
    with torch.no_grad():
        embeds, _ = enc.encode_prompt(a.prompt, device=device, do_classifier_free_guidance=False)
    embeds = [e.detach().clone() for e in embeds]
    del enc
    free()
    t1 = time.time()
    print(f"prompt encodé en {t1 - t0:.0f} s")

    # 2. débruitage + décodage, sans encodeur de texte
    kw = {}
    if a.gguf:
        from diffusers import GGUFQuantizationConfig, ZImageTransformer2DModel

        kw["transformer"] = ZImageTransformer2DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/blob/main/{a.gguf}",
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            config=REPO, subfolder="transformer", torch_dtype=dtype,
        )
    pipe = ZImagePipeline.from_pretrained(REPO, text_encoder=None, tokenizer=None, torch_dtype=dtype, **kw).to(device)
    pipe.set_progress_bar_config(disable=False)
    t2 = time.time()
    print(f"transformer chargé en {t2 - t1:.0f} s")

    def sync(_pipe, _step, _t, kw):
        # MPS est asynchrone : sans synchronisation la barre finit en 1 s et le calcul arrive après
        if device == "mps":
            torch.mps.synchronize()
        return kw

    print("débruitage (chaque pas prend environ une minute sur un M1)...")
    img = pipe(
        prompt_embeds=embeds, width=a.width, height=a.height,
        num_inference_steps=a.steps, guidance_scale=0.0,
        generator=torch.Generator("cpu").manual_seed(a.seed),
        callback_on_step_end=sync,
    ).images[0]
    t3 = time.time()

    out = Path(a.out) if a.out else Path(__file__).parent / "out" / f"{time.strftime('%Y%m%d-%H%M%S')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"image générée en {t3 - t2:.0f} s -> {out}")
    if device == "mps":
        print(f"mémoire MPS max allouée par le driver : {torch.mps.driver_allocated_memory() / 1e9:.1f} Go")


if __name__ == "__main__":
    main()
