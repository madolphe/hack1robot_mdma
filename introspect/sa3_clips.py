"""Couche A de la sonorisation avec Stable Audio 3 : un clip par prompt.

Tourne dans son propre environnement (.venv-audio : la bibliothèque stable-audio-3 exige torch 2.7.1), lancé par
sound.py sans rien importer du projet :
    .venv-audio/bin/python -I introspect/sa3_clips.py jobs.json
jobs.json : {"model": "small-sfx", "steps": 8, "cfg_scale": 1.0,
             "clips": [{"prompt": "...", "duration": 4.2, "seed": 1000, "out": "chemin.wav"}, ...]}
Chaque clip est écrit en wav stéréo float32 à la fréquence du modèle (44,1 kHz)."""
import json
import sys
import time

import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel


def main():
    job = json.load(open(sys.argv[1], encoding="utf-8"))
    t0 = time.time()
    model = StableAudioModel.from_pretrained(job.get("model", "small-sfx"))
    sr = model.model.sample_rate
    print(f"chargement {job.get('model', 'small-sfx')} : {time.time() - t0:.1f} s", flush=True)
    t0 = time.time()
    for i, c in enumerate(job["clips"]):
        audio = model.generate(prompt=c["prompt"], duration=c["duration"], steps=job.get("steps", 8),
                               cfg_scale=job.get("cfg_scale", 1.0), seed=c.get("seed", -1))
        sf.write(c["out"], audio[0].float().cpu().numpy().T, sr, subtype="FLOAT")
        if i % 10 == 0:
            print(f"clip {i}/{len(job['clips'])} : {c['prompt'][:80]}", flush=True)
    print(f"{len(job['clips'])} clips en {time.time() - t0:.1f} s, {sr} Hz, "
          f"VRAM max {torch.cuda.max_memory_allocated() / 2**30:.1f} Gio", flush=True)


if __name__ == "__main__":
    main()
