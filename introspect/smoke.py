"""Essais unitaires des briques : image, voix (+ effets), clip Stable Audio."""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from .common import ROOT, load_config

OUT = ROOT / "runs" / "setup" / "smoke"


def t(msg, t0):
    print(f"{msg} : {time.time() - t0:.1f} s", flush=True)


def image(cfg):
    from .render import backend_of, load_pipes
    from .windows import make_prompt

    ic = cfg["image"]
    tag = backend_of(ic)
    t0 = time.time()
    t2i, i2i = load_pipes(cfg)
    t(f"chargement {ic['model']}", t0)
    prompt = make_prompt(["silence", "voice", "whisper", "language", "echo", "emptiness"], cfg)
    t0 = time.time()
    img = t2i(prompt, torch.Generator("cuda").manual_seed(1))
    t(f"text-to-image {ic['size'][0]}x{ic['size'][1]}, {ic['steps']} étapes", t0)
    img.save(OUT / f"image_t2i_{tag}.png")
    t0 = time.time()
    img2 = i2i(make_prompt(["ocean", "waves", "depth", "salt", "silence", "echo"], cfg), img, ic["strength"],
               torch.Generator("cuda").manual_seed(1))
    t(f"img2img strength {ic['strength']}", t0)
    img2.save(OUT / f"image_i2i_{tag}.png")
    print("VRAM max", round(torch.cuda.max_memory_allocated() / 2**30, 1), "Gio")


def voice(cfg):
    import soundfile as sf
    from kokoro import KPipeline
    from pedalboard import Chorus, Delay, Pedalboard, PitchShift, Reverb

    t0 = time.time()
    fr = KPipeline(lang_code="f", repo_id="hexgrad/Kokoro-82M")
    t("chargement Kokoro fr", t0)
    t0 = time.time()
    chunks = [a for _, _, a in fr("silence … mémoire … océan", voice="ff_siwis", speed=0.9)]
    wav = np.concatenate([c.numpy() if hasattr(c, "numpy") else c for c in chunks]).astype(np.float32)
    t("synthèse", t0)
    sf.write(OUT / "voice_raw.wav", wav, 24000)
    board = Pedalboard([PitchShift(semitones=-3), Chorus(), Delay(delay_seconds=0.3, feedback=0.4, mix=0.3),
                        Reverb(room_size=0.9, wet_level=0.5)])
    fx = board(wav[None, :], 24000)[0]
    sf.write(OUT / "voice_fx.wav", fx, 24000)
    print("durée", round(len(wav) / 24000, 2), "s")


def audio(cfg):
    import soundfile as sf

    la = cfg["sound"]["sono"]["layer_a"]
    if la.get("engine") == "stable_audio_3":
        from .sound import stable_audio_3

        t0 = time.time()
        clips, sr = stable_audio_3(la, [{"prompt": "silence, echo, whisper, " + la["suffix"], "duration": 10.0, "seed": 1}],
                                   OUT, lambda *m: print(*m, flush=True))
        t(f"Stable Audio 3 {la['model']}, chargement compris", t0)
        sf.write(OUT / "sono_clip_sa3.wav", clips[0], sr)
        return
    from diffusers import StableAudioPipeline

    t0 = time.time()
    from .common import cached_snapshot

    pipe = StableAudioPipeline.from_pretrained(cached_snapshot(cfg["sound"]["sono"]["layer_a"]["model"]),
                                               torch_dtype=torch.float16).to("cuda")
    t("chargement Stable Audio", t0)
    t0 = time.time()
    a = pipe("silence, echo, whisper, ambient texture, drone, field recording", negative_prompt="low quality",
             num_inference_steps=100, audio_end_in_s=10.0, generator=torch.Generator("cuda").manual_seed(1)).audios
    t("clip 10 s, 100 étapes", t0)
    sf.write(OUT / "sono_clip.wav", a[0].T.float().cpu().numpy(), pipe.vae.sampling_rate)


def main():
    """./run.sh smoke [image|voice|audio …] [--config c.yaml]"""
    OUT.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    config = ROOT / "configs" / "default.yaml"
    if "--config" in args:
        i = args.index("--config")
        config = args[i + 1]
        del args[i:i + 2]
    cfg = load_config(config)
    for w in args or ["image", "voice", "audio"]:
        print(f"=== {w}", flush=True)
        globals()[w](cfg)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
