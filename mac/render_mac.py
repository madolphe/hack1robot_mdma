"""Passe 2a sur Mac : une image par tranche avec Sana-Sprint (MPS), même logique que introspect/render.py
(t2i puis chaîne img2img, t2i pur toutes les image.reset_every tranches). Écrit frames/NNNN.png et frames.json,
que assemble.py lit tels quels.

    python mac/render_mac.py --run runs/q1_rave            # transformer 0.6B (rapide)
    python mac/render_mac.py --run runs/q1_rave --big      # transformer 1.6B
    python mac/render_mac.py --run runs/q1_rave --backend openrouter   # Recraft via OpenRouter (texte -> image seul)
    python mac/render_mac.py --run runs/q1_rave --backend openrouter --per-slice 3   # 3 images par tranche
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from introspect.common import Log, run_config

REPO = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
REPO_SMALL = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
I2I_STEPS = 4  # à 2 pas, strength 0.7 arrondit à une régénération complète


def render_openrouter(run_dir: Path, wins: list, cfg: dict, log, model: str | None, workers: int, per_slice: int = 1):
    """`per_slice` images par tranche via OpenRouter, en parallèle, toutes avec le prompt de la tranche entière
    (tous ses tokens). Pas d'img2img : la continuité vient du montage (assemble.py, deforum_mac.py).
    Fichiers : NNNN.png (1re image, celle que lit assemble.py), puis NNNN_01.png, NNNN_02.png…"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from PIL import Image
    from introspect import openrouter as orr

    model = model or orr.DEFAULT_IMAGE_MODEL
    W, H = cfg["image"]["size"]
    frames = run_dir / "frames"
    frames.mkdir(exist_ok=True)
    log(f"OpenRouter {model}, {len(wins)} tranches x {per_slice} images, {workers} requêtes en parallèle")
    name = lambda k, j: f"{k:04d}.png" if j == 0 else f"{k:04d}_{j:02d}.png"

    def one(k, j, w):
        t0 = time.time()
        for attempt in range(4):
            try:
                tmp = orr.image(w["prompt"], frames / f"{k:04d}_{j:02d}_raw", model=model, aspect_ratio="16:9")
                break
            except orr.OpenRouterError as e:
                if attempt == 3 or "429" not in str(e):
                    raise
                time.sleep(10 * (attempt + 1))
        Image.open(tmp).convert("RGB").resize((W, H), Image.LANCZOS).save(frames / name(k, j))
        tmp.unlink()
        return k, j, time.time() - t0

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, k, j, w) for k, w in enumerate(wins) for j in range(per_slice)]
        for f in as_completed(futs):
            k, j, dt = f.result()
            log(f"tranche {k + 1}/{len(wins)} image {j + 1}/{per_slice} ({dt:.0f} s) : {wins[k]['prompt'][55:110]}")
    manifest = [{"window": k, "file": name(k, j), "intermediate": False, "sub": j}
                for k in range(len(wins)) for j in range(per_slice)]
    (run_dir / "frames.json").write_text(json.dumps(manifest, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--big", action="store_true", help="transformer 1.6B au lieu de 0.6B")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", choices=["sana", "openrouter"], default="sana")
    ap.add_argument("--model", default=None, help="slug OpenRouter (backend openrouter)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-slice", type=int, default=1, help="images par tranche (backend openrouter)")
    a = ap.parse_args()

    run_dir = Path(a.run)
    cfg = run_config(run_dir)
    ic = cfg["image"]
    log = Log(run_dir, "render")
    wins = json.loads((run_dir / "windows.json").read_text(encoding="utf-8"))[: a.limit]
    if a.backend == "openrouter":
        render_openrouter(run_dir, wins, cfg, log, a.model, a.workers, a.per_slice)
        log.done()
        return
    render_sana(run_dir, wins, cfg, ic, log, a.big)
    log.done()


def render_sana(run_dir, wins, cfg, ic, log, big):
    import torch
    from diffusers import SanaSprintImg2ImgPipeline, SanaSprintPipeline, SanaTransformer2DModel

    torch.set_grad_enabled(False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16  # fp16 : image noire en 1.6B, pas plus rapide en 0.6B (mesuré)

    kw = {} if big else {"transformer": SanaTransformer2DModel.from_pretrained(REPO_SMALL, subfolder="transformer", torch_dtype=dtype)}
    t2i = SanaSprintPipeline.from_pretrained(REPO, torch_dtype=dtype, **kw).to(device)
    i2i = SanaSprintImg2ImgPipeline(**t2i.components)
    for p in (t2i, i2i):
        p.set_progress_bar_config(disable=True)
    W, H = ic["size"]
    seed, reset = ic.get("seed", 1), ic.get("reset_every", 0) or 0
    log(f"Sana-Sprint {'1.6B' if big else '0.6B'} sur {device}, {len(wins)} tranches {W}x{H}, "
        f"strength {ic['strength']}, reset_every {reset}")

    frames = run_dir / "frames"
    frames.mkdir(exist_ok=True)
    prev, manifest = None, []
    for k, w in enumerate(wins):
        t0 = time.time()
        gen = torch.Generator("cpu").manual_seed(seed + k)
        if prev is None or (reset and k % reset == 0):
            img = t2i(prompt=w["prompt"], width=W, height=H, num_inference_steps=2, generator=gen).images[0]
            kind = "t2i"
        else:
            img = i2i(prompt=w["prompt"], image=prev, strength=ic["strength"], width=W, height=H,
                      num_inference_steps=I2I_STEPS, intermediate_timesteps=None, generator=gen).images[0]
            kind = "i2i"
        img.save(frames / f"{k:04d}.png")
        manifest.append({"window": k, "file": f"{k:04d}.png", "intermediate": False})
        prev = img
        log(f"image {k + 1}/{len(wins)} ({kind}, {time.time() - t0:.0f} s) : {w['prompt'][55:120]}")
    (run_dir / "frames.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
