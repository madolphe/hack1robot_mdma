"""Passe 2a : une image par tranche (t2i puis chaîne img2img).

Moteurs (`image.backend`) : `zimage` (Z-Image-Turbo, défaut), `sana_sprint` (SANA-Sprint, 1 à 4 étapes),
`flux2_klein` (FLUX.2 [klein] : l'image précédente sert de référence au lieu d'être rebruitée)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import Log, run_config, load_config, deep_merge
from .windows import make_prompt


def backend_of(ic: dict) -> str:
    if ic.get("backend"):
        return ic["backend"]
    name = ic["model"].lower()
    return "sana_sprint" if "sana" in name else "flux2_klein" if "flux.2" in name else "zimage"


def load_pipes(cfg):
    """Renvoie deux fonctions : t2i(prompt, gen) et i2i(prompt, image, strength, gen) -> PIL.Image."""
    ic = cfg["image"]
    W, H = ic["size"]
    steps, backend = ic["steps"], backend_of(ic)
    if backend == "zimage":
        from diffusers import ZImageImg2ImgPipeline as I2I, ZImagePipeline as T2I
        kw = dict(guidance_scale=0.0)
    elif backend == "sana_sprint":
        from diffusers import SanaSprintImg2ImgPipeline as I2I, SanaSprintPipeline as T2I
        kw = dict(guidance_scale=ic.get("guidance_scale", 4.5))
    elif backend == "flux2_klein":
        from diffusers import Flux2KleinPipeline as T2I
        I2I = None  # même pipeline : l'image précédente passe en référence (`image=`)
        kw = dict(guidance_scale=ic.get("guidance_scale", 1.0))
    else:
        raise ValueError(f"moteur d'image inconnu : {backend}")
    t2i = T2I.from_pretrained(ic["model"], torch_dtype=torch.bfloat16)
    if ic.get("cpu_offload"):  # FLUX.2 [klein] 4B (~16 Go en bf16) sur un GPU de 16 Go
        t2i.enable_model_cpu_offload()
    else:
        t2i.to("cuda")
    i2i = I2I(**t2i.components) if I2I else t2i  # composants partagés : un seul chargement
    t2i.set_progress_bar_config(disable=True)
    i2i.set_progress_bar_config(disable=True)
    kw.update(height=H, width=W, num_inference_steps=steps)

    def run_t2i(prompt, gen):
        return t2i(prompt=prompt, generator=gen, **kw).images[0]

    def run_i2i(prompt, image, strength, gen):
        if backend == "flux2_klein":
            # avec une image de référence, FLUX.2 [klein] attend une consigne d'édition : un prompt
            # seulement descriptif lui fait recopier la référence
            prompt = ic.get("edit_prefix", "Repaint this picture, keeping only its colors and brushwork, as: ") + prompt
            return i2i(prompt=prompt, image=[image], generator=gen, **kw).images[0]
        return i2i(prompt=prompt, image=image, strength=strength, generator=gen, **kw).images[0]

    return run_t2i, run_i2i


def interp_prompt(a: dict, b: dict, frac: float, style: str) -> str:
    wa = dict(a["words"]); wb = dict(b["words"])
    mix = {w: (1 - frac) * wa.get(w, 0) + frac * wb.get(w, 0) for w in set(wa) | set(wb)}
    words = [w for w, _ in sorted(mix.items(), key=lambda x: -x[1])[:12]]
    return words


@torch.no_grad()
def render(run_dir: Path, cfg: dict | None = None, only_t2i: bool = False, limit: int | None = None,
           frames_dir: str = "frames"):
    run_dir = Path(run_dir)
    cfg = cfg or run_config(run_dir)
    log = Log(run_dir, "render")
    ic = cfg["image"]
    wins = json.loads((run_dir / "windows.json").read_text(encoding="utf-8"))
    for w in wins:
        w["prompt"] = make_prompt([x for x, _ in w["words"]], cfg)
    if limit:
        wins = wins[:limit]
    frames = run_dir / frames_dir
    frames.mkdir(exist_ok=True)
    t2i, i2i = load_pipes(cfg)
    log(f"moteur {backend_of(ic)} : {ic['model']}, {ic['steps']} étapes")
    seed = ic.get("seed", 1)
    reset = ic.get("reset_every", 0) or 0
    n_mid = ic.get("intermediates", 0) or 0
    prev = None
    manifest = []
    for k, w in enumerate(wins):
        gen = torch.Generator("cuda").manual_seed(seed + k)
        if prev is not None and n_mid:
            for m in range(1, n_mid + 1):
                frac = m / (n_mid + 1)
                p = make_prompt(interp_prompt(wins[k - 1], w, frac, ic["style"]), cfg)
                img = i2i(p, prev, min(0.35, ic["strength"]), gen)
                name = f"{k:04d}_{m:02d}.png"
                img.save(frames / name)
                manifest.append({"window": k, "file": name, "intermediate": True})
                prev = img
        if prev is None or only_t2i or (reset and k % reset == 0):
            img = t2i(w["prompt"], gen)
        else:
            img = i2i(w["prompt"], prev, ic["strength"], gen)
        img.save(frames / f"{k:04d}.png")
        manifest.append({"window": k, "file": f"{k:04d}.png", "intermediate": False})
        prev = img
        if k % 10 == 0:
            log(f"image {k}/{len(wins)} : {w['prompt'][:90]}")
    (run_dir / ("frames.json" if frames_dir == "frames" else frames_dir + ".json")).write_text(json.dumps(manifest, indent=1))
    log(f"{len(wins)} images")
    log.done()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--only-t2i", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--frames-dir", default="frames")
    ap.add_argument("--override", help="YAML fusionné dans la config du run (essais)")
    a = ap.parse_args(argv)
    cfg = run_config(Path(a.run))
    if a.override:
        import yaml
        cfg = deep_merge(cfg, yaml.safe_load(open(a.override)))
    render(Path(a.run), cfg, only_t2i=a.only_t2i, limit=a.limit, frames_dir=a.frames_dir)


if __name__ == "__main__":
    main()
