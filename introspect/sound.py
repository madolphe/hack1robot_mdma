"""Son : effets de voix sémantiques (pedalboard) et sonorisation (Stable Audio + synthèse).

Produit audio/voice_fx_NNNN.wav, audio/sono_NNNN.wav, features.json, voice_fx.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import Log, run_config
from . import features as F

SR = 44100
VSR = 24000


def to_sr(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    from math import gcd
    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g, axis=0).astype(np.float32)


# ---------------------------------------------------------------- effets de voix
def word_fx(wav: np.ndarray, f: dict, t: dict, heard: bool, pan_base: float = 0.0, pan_scale: float = 1.0) -> np.ndarray:
    """wav mono 44,1 kHz -> stéréo [n, 2] avec effets pilotés par les traits."""
    from pedalboard import (Bitcrush, Chorus, Compressor, Delay, Distortion, Gain, HighShelfFilter,
                            LowpassFilter, Pedalboard, Phaser, PitchShift, Reverb)

    v, a = f["valence"], f["arousal"]
    conc = f["concreteness"]
    score = t.get("score", 0.1)
    drug = t.get("drug", 0.0)
    fx = []
    if v < 0.4:  # valence négative : voix assombrie
        fx += [PitchShift(semitones=-1 - 3 * (0.4 - v) / 0.4), LowpassFilter(cutoff_frequency_hz=2500 + 4000 * v)]
    elif v > 0.6:  # valence positive : claire, ouverte
        fx += [Chorus(rate_hz=0.8, depth=0.15, mix=0.3), HighShelfFilter(cutoff_frequency_hz=4000, gain_db=4 * (v - 0.6) / 0.4)]
    if a > 0.6:  # éveil élevé : tendue
        fx += [Compressor(threshold_db=-24, ratio=6), Distortion(drive_db=4 + 10 * (a - 0.6)), Delay(delay_seconds=0.06, feedback=0.1, mix=0.2)]
    if score < 0.05:  # mot faible : murmure, voix fantôme
        fx += [Gain(gain_db=-8), LowpassFilter(cutoff_frequency_hz=1800), Bitcrush(bit_depth=8)]
    if heard:  # mot déjà entendu : écho, mémoire
        fx += [Delay(delay_seconds=0.33, feedback=0.45, mix=0.35)]
    if drug > 0:
        d = min(1.0, drug)
        fx += [Phaser(rate_hz=0.5 + 3 * d, depth=0.4 + 0.5 * d, mix=0.3 + 0.5 * d), Distortion(drive_db=2 + 14 * d)]
    # abstrait : très grande réverbération ; concret : sec
    wet = float(np.clip((0.7 - conc) / 0.7, 0, 1))
    if a < 0.35:  # éveil faible : longue réverbération
        wet = max(wet, 0.6)
    if wet > 0.05:
        fx += [Reverb(room_size=0.5 + 0.5 * wet, wet_level=0.15 + 0.6 * wet, dry_level=1 - 0.5 * wet, width=1.0)]
    tail = int(SR * (0.3 + 2.0 * wet + (0.8 if heard else 0)))
    x = np.concatenate([wav, np.zeros(tail, np.float32)])
    y = Pedalboard(fx)(x[None, :].astype(np.float32), SR)[0] if fx else x
    if drug > 0:  # pitch oscillant (vibrato par rééchantillonnage variable)
        d = min(1.0, drug)
        n = len(y)
        tt = np.arange(n) / SR
        warp = tt + (0.004 * d) * np.sin(2 * np.pi * (2 + 3 * d) * tt) / (2 * np.pi * (2 + 3 * d)) * 2 * np.pi
        y = np.interp(np.clip(warp * SR, 0, n - 1), np.arange(n), y).astype(np.float32)
    # panoramique et distance selon la couche d'origine
    layer = t.get("layer", 0.5)
    pan = float(np.clip(pan_base + (layer - 0.5) * 1.6 * pan_scale, -1, 1))
    gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    dist = 1.0 - 0.4 * layer
    return np.stack([y * gl * dist, y * gr * dist], axis=1)


def fx_fragment(run_dir, cfg, k, words, weights, words_dir, feats, tfeats, heard, spk_get,
                voice_name=None, pan_base=0.0, pan_scale=1.0) -> np.ndarray:
    """Effets mot par mot pour un fragment (une tranche, une voix) -> stéréo 44,1 kHz."""
    gap = cfg["voice"].get("gap_s", 0.25)
    poly = cfg["voice"].get("polyphony", False)
    clips, raw_lens = [], []
    for j, word in enumerate(words):
        f = feats.get(word) or {"valence": .5, "arousal": .5, "concreteness": .5}
        t = tfeats.get(k, {}).get(word, {})
        if f["arousal"] < 0.35:  # éveil faible : débit ralenti
            spk = spk_get()
            sp = cfg["voice"]["speed"] * 0.75
            wav = spk.say_as(word, voice_name, sp) if voice_name else spk.say(word, speed=sp)
        else:
            wav, _ = sf.read(words_dir / f"{k:04d}_{j:02d}.wav", dtype="float32")
        w44 = to_sr(wav, VSR)
        raw_lens.append(len(w44))
        clips.append(word_fx(w44, f, t, word in heard, pan_base, pan_scale))
        heard.add(word)
    if not clips:
        return np.zeros((int(0.1 * SR), 2), np.float32)
    if poly:
        hop = int(0.35 * SR)
        n = hop * (len(clips) - 1) + max(len(c) for c in clips)
        out = np.zeros((n, 2), np.float32)
        ws = weights or [1] * len(clips)
        mw = max(ws) or 1
        for i, (c, g) in enumerate(zip(clips, ws)):
            out[i * hop: i * hop + len(c)] += c * (0.35 + 0.65 * g / mw)
    else:
        # rythme de la voix brute : chaque mot démarre après le précédent + silence ;
        # les queues d'effets se chevauchent, seule la dernière dépasse (≤ 1,2 s)
        starts, t = [], 0
        for c, L in zip(clips, raw_lens):
            starts.append(t)
            t += L + int(gap * SR)
        n = min(max(s0 + len(c) for s0, c in zip(starts, clips)), t + int(1.2 * SR))
        out = np.zeros((n, 2), np.float32)
        for s0, c in zip(starts, clips):
            seg = c[: max(0, n - s0)]
            out[s0: s0 + len(seg)] += seg
        fade = min(n, int(0.3 * SR))
        out[n - fade:] *= np.linspace(1, 0, fade)[:, None]
    return out / max(1.0, float(np.abs(out).max()) / 0.95)


def voice_fx(run_dir: Path, cfg: dict, wins, feats, tfeats, log) -> list[float]:
    from .voice import VOICES, Speaker

    meta = json.loads((run_dir / "voice.json").read_text(encoding="utf-8"))
    holder = {}

    def spk_get():
        if "s" not in holder:
            holder["s"] = Speaker(cfg)
        return holder["s"]

    heard: set[str] = set()
    durs, out_meta = [], []
    for k, m in enumerate(meta):
        if "voices" in m:  # voix en parallèle : une piste par voix
            vd = []
            for v, vv in enumerate(m["voices"]):
                if not vv["words"]:
                    vd.append(0.0)
                    continue
                name, _, pan = VOICES[v]
                out = fx_fragment(run_dir, cfg, k, vv["words"], vv["weights"], run_dir / "audio" / f"v{v}" / "words",
                                  feats, tfeats, heard, spk_get, voice_name=name, pan_base=pan, pan_scale=0.5)
                sf.write(run_dir / "audio" / f"v{v}" / f"fx_{k:04d}.wav", out, SR)
                vd.append(round(len(out) / SR, 3))
            d = max(vd) if vd else 0.1
            out_meta.append({"window": k, "duration": d, "voices": vd})
        else:
            out = fx_fragment(run_dir, cfg, k, m["words"], m.get("weights"), run_dir / "audio" / "words",
                              feats, tfeats, heard, spk_get)
            sf.write(run_dir / "audio" / f"voice_fx_{k:04d}.wav", out, SR)
            d = round(len(out) / SR, 3)
            out_meta.append({"window": k, "duration": d})
        durs.append(d)
    (run_dir / "voice_fx.json").write_text(json.dumps(out_meta, indent=1))
    log(f"effets de voix : {len(meta)} tranches, {sum(durs):.1f} s")
    return durs


# ---------------------------------------------------------------- sonorisation
def synth_window(words: list, feats: dict, tf: dict, dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Couche B : drone paramétrique, une voix par mot."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros((n, 2), np.float32)
    env_in = np.minimum(1, t / 0.8) * np.minimum(1, (dur - t) / 0.8).clip(0, 1)
    for word, weight in words[:6]:
        f = feats.get(word) or {}
        x = tf.get(word, {})
        v, a, d = f.get("valence", .5), f.get("arousal", .5), f.get("dominance", .5)
        conc = f.get("concreteness", .5)
        layer = x.get("layer", .5)
        drug = min(1.0, x.get("drug", 0.0))
        base = 55 * 2 ** (2.5 * layer + 1.2 * (1 - d))  # couche -> hauteur, dominance -> grave
        ratios = [1, 1.5, 1.25] if v >= 0.5 else [1, 16 / 15, 2 ** 0.5]  # consonant / dissonant
        vib = 0.003 * np.sin(2 * np.pi * (0.2 + 5 * a) * t)
        detune = 1 + 0.004 * (f.get("olfactory", .15) + f.get("gustatory", .15)) + 0.02 * drug * t / max(dur, 1e-3)
        sig = np.zeros(n)
        bright = f.get("visual", .5)
        for r in ratios:
            ph = 2 * np.pi * base * r * np.cumsum(1 + vib) / sr
            sig += np.sin(ph) + 0.5 * np.sin(ph * detune)
            for h in (2, 3, 4):  # brillance
                sig += bright * 0.3 / h * np.sin(h * ph)
        sig /= len(ratios) * 2
        noise = rng.standard_normal(n)
        k = max(1, int(sr / (200 + 3000 * f.get("auditory", .3))))
        noise = np.convolve(noise, np.ones(k) / k, mode="same")
        sig = (1 - 0.6 * conc) * sig + 0.6 * conc * noise * 2  # concret : bruit filtré
        pulse = 0.65 + 0.35 * np.sin(2 * np.pi * (0.1 + 4 * a) * t)  # éveil -> pulsation
        rough = 1 - f.get("haptic", .3) * 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * 45 * t))  # toucher -> rugosité
        sig = sig * pulse * rough
        hb = f.get("interoceptive", .3) * np.sin(2 * np.pi * 48 * t) * np.exp(-((t * 1.1) % 1) * 9)  # pouls
        sig = sig + hb
        if drug > 0:  # modulation en anneau + saturation
            sig = sig * (1 - drug + drug * np.sin(2 * np.pi * (30 + 200 * drug) * t))
            sig = np.tanh(sig * (1 + 4 * drug))
        if x.get("novelty"):  # attaque percussive
            sig[: int(0.25 * sr)] += np.exp(-np.arange(int(0.25 * sr)) / (0.03 * sr)) * rng.standard_normal(int(0.25 * sr)) * 0.8
        amp = (0.3 + 0.7 * weight / max(1e-6, words[0][1])) * (0.6 + 0.6 * d)
        pan = (layer - 0.5) * 1.4
        gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        out[:, 0] += (sig * amp * gl).astype(np.float32)
        out[:, 1] += (sig * amp * gr).astype(np.float32)
    out *= env_in[:, None]
    return out / max(1e-6, float(np.abs(out).max())) * 0.8


def stable_audio_3(la: dict, clips: list[dict], workdir: Path, log) -> tuple[list[np.ndarray], int]:
    """Clips Stable Audio 3 (`clips` : prompt, duration, seed) générés par introspect/sa3_clips.py dans son
    propre environnement (`layer_a.python`, par défaut .venv-audio : la bibliothèque exige torch 2.7.1)."""
    import subprocess

    from .common import ROOT

    workdir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(clips):
        c["out"] = str(workdir / f"sa3_{i:04d}.wav")
    jobs = workdir / "sa3_jobs.json"
    jobs.write_text(json.dumps({"model": la["model"], "steps": la.get("steps", 8), "cfg_scale": la.get("cfg_scale", 1.0),
                                "clips": clips}, ensure_ascii=False, indent=1), encoding="utf-8")
    py = ROOT / la.get("python", ".venv-audio/bin/python")
    r = subprocess.run([str(py), "-I", str(ROOT / "introspect" / "sa3_clips.py"), str(jobs)],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        log("stable audio 3 :", line)
    if r.returncode != 0:
        raise RuntimeError(f"sa3_clips.py (code {r.returncode}) : {r.stderr.strip().splitlines()[-1:]}")
    out = [sf.read(c["out"], dtype="float32") for c in clips]
    return [a for a, _ in out], out[0][1]


def sono(run_dir: Path, cfg: dict, wins, feats, tfeats, durs, log):
    sc = cfg["sound"]["sono"]
    c = cfg["video"]["crossfade_s"]
    la = sc["layer_a"]
    clips_a = {}
    if la.get("enabled") and la.get("engine") == "stable_audio_3":
        try:
            jobs = [{"prompt": ", ".join(x for x, _ in w["words"][:6]) + ", " + la["suffix"],
                     "duration": round(min(47.0, durs[k] + c + 0.5), 2), "seed": 1000 + k} for k, w in enumerate(wins)]
            clips, sr_a = stable_audio_3(la, jobs, run_dir / "audio" / "sa3", log)
            clips_a = {k: (a if sr_a == SR else to_sr(a, sr_a)) for k, a in enumerate(clips)}
        except Exception as e:
            log("couche A désactivée :", repr(e))
            clips_a = {}
    elif la.get("enabled"):
        try:
            import torch
            from diffusers import StableAudioPipeline

            from .common import cached_snapshot

            pipe = StableAudioPipeline.from_pretrained(cached_snapshot(la["model"]), torch_dtype=torch.float16).to("cuda")
            pipe.set_progress_bar_config(disable=True)
            for k, w in enumerate(wins):
                words = [x for x, _ in w["words"][:6]]
                prompt = ", ".join(words) + ", " + la["suffix"]
                length = min(47.0, durs[k] + c + 0.5)
                g = torch.Generator("cuda").manual_seed(1000 + k)
                a = pipe(prompt, negative_prompt="low quality, speech, vocals", num_inference_steps=la.get("steps", 100),
                         audio_end_in_s=length, generator=g).audios[0]
                clips_a[k] = a.T.float().cpu().numpy()
                if k % 10 == 0:
                    log(f"stable audio {k}/{len(wins)} : {prompt[:80]}")
            sr_a = pipe.vae.sampling_rate
            del pipe
            torch.cuda.empty_cache()
            if sr_a != SR:
                clips_a = {k: to_sr(v, sr_a) for k, v in clips_a.items()}
        except Exception as e:
            log("couche A désactivée :", repr(e))
            clips_a = {}
    rng = np.random.default_rng(1)
    for k, w in enumerate(wins):
        n = int((durs[k] + c) * SR)
        mix = np.zeros((n, 2), np.float32)
        if k in clips_a:
            a = clips_a[k][:n]
            a = a / max(1e-6, float(np.abs(a).max())) * 0.8
            mix[: len(a)] += a * 0.7
        if sc["layer_b"].get("enabled"):
            mix += synth_window(w["words"], feats, tfeats.get(k, {}), durs[k] + c, SR, rng) * (0.5 if k in clips_a else 0.9)
        mix = mix / max(1.0, float(np.abs(mix).max()) / 0.95)
        sf.write(run_dir / "audio" / f"sono_{k:04d}.wav", mix, SR)
    log(f"sonorisation : {len(wins)} clips (couche A : {'oui' if clips_a else 'non'}, couche B : {'oui' if sc['layer_b'].get('enabled') else 'non'})")


def sound(run_dir: Path, cfg: dict | None = None, mode: str | None = None):
    run_dir = Path(run_dir)
    cfg = cfg or run_config(run_dir)
    mode = mode or cfg["sound"]["mode"]
    log = Log(run_dir, "sound")
    wins = json.loads((run_dir / "windows.json").read_text(encoding="utf-8"))
    feats = F.load_or_compute(run_dir, wins, cfg)
    tfeats = F.trace_features(run_dir, wins, cfg)
    log(f"traits : {len(feats)} mots")
    if cfg["sound"]["voice_fx"].get("enabled"):
        durs = voice_fx(run_dir, cfg, wins, feats, tfeats, log)
    else:
        meta = json.loads((run_dir / "voice.json").read_text())
        durs = [m["duration"] for m in meta]
    from .assemble import slice_durations
    durs = slice_durations(durs, cfg)
    if mode in ("sonorisation", "mixte"):
        sono(run_dir, cfg, wins, feats, tfeats, durs, log)
    log.done()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", choices=["voix", "sonorisation", "mixte"])
    a = ap.parse_args(argv)
    sound(Path(a.run), mode=a.mode)


if __name__ == "__main__":
    main()
