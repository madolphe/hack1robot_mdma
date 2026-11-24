"""Passe 2b : voix brute Kokoro, un fragment par tranche (mots nouveaux ou 3 premiers mots)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import Log, run_config
from .lexicons import lang_of

SR = 24000


class Speaker:
    def __init__(self, cfg):
        from kokoro import KPipeline

        self.vc = cfg["voice"]
        self.fr = KPipeline(lang_code="f", repo_id="hexgrad/Kokoro-82M")
        self.en = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M") if self.vc.get("english_voice") else None
        self.cache: dict[tuple, np.ndarray] = {}
        self.pipes = {"f": self.fr}

    def say_as(self, word: str, voice: str, speed: float | None = None) -> np.ndarray:
        """Prononce `word` avec une voix Kokoro donnée (ff_*, af_*, am_*, bm_*…)."""
        from kokoro import KPipeline

        speed = speed or self.vc["speed"]
        key = (word, round(speed, 3), voice)
        if key not in self.cache:
            lang = voice[0]
            if lang not in self.pipes:
                self.pipes[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
            parts = [a.numpy() if hasattr(a, "numpy") else np.asarray(a) for _, _, a in self.pipes[lang](word, voice=voice, speed=speed)]
            wav = np.concatenate(parts).astype(np.float32) if parts else np.zeros(int(0.2 * SR), np.float32)
            self.cache[key] = trim(wav)
        return self.cache[key]

    def say(self, word: str, speed: float | None = None) -> np.ndarray:
        speed = speed or self.vc["speed"]
        use_en = self.en is not None and lang_of(word) == "en"
        key = (word, round(speed, 3), use_en)
        if key not in self.cache:
            pipe, voice = (self.en, self.vc.get("voice_en", "af_heart")) if use_en else (self.fr, self.vc["voice"])
            parts = [a.numpy() if hasattr(a, "numpy") else np.asarray(a) for _, _, a in pipe(word, voice=voice, speed=speed)]
            wav = np.concatenate(parts).astype(np.float32) if parts else np.zeros(int(0.2 * SR), np.float32)
            self.cache[key] = trim(wav)
        return self.cache[key]


def trim(wav: np.ndarray, thr: float = 0.01) -> np.ndarray:
    idx = np.where(np.abs(wav) > thr)[0]
    if len(idx) == 0:
        return wav
    a, b = max(0, idx[0] - 240), min(len(wav), idx[-1] + 2400)
    return wav[a:b]


def words_for(w: dict, mode: str) -> list[str]:
    k = w.get("max_words", 6)
    if mode == "new_words":
        ws = w["new"][:k]
        return ws or [x for x, _ in w["words"][:1]]  # tranche sans mot nouveau : le premier mot
    return [x for x, _ in w["words"][: min(3, k)]]


VOICES = [  # (voix Kokoro, langue, panoramique de base)
    ("ff_siwis", "fr", 0.0),
    ("af_heart", "en", -0.5),
    ("am_michael", "en", 0.5),
    ("bm_george", "en", -0.25),
]


def assign_words(w: dict, n_voices: int) -> list[list[tuple[str, float]]]:
    """Répartit jusqu'à N × max_words mots (nouveaux d'abord, puis les plus forts) entre N voix :
    anglais -> voix anglaises, français -> ff_siwis, le reste à tour de rôle par ordre de poids."""
    k = w.get("max_words", 6)
    weights = dict(w["words"])
    order = list(w["new"]) + [x for x, _ in w["words"] if x not in w["new"]]
    order = order[: n_voices * k]
    slots: list[list[tuple[str, float]]] = [[] for _ in range(n_voices)]
    en_voices = [i for i in range(1, n_voices)]
    rr = 0
    for word in order:
        lang = lang_of(word)
        pref = [0] if lang == "fr" else sorted(en_voices, key=lambda i: len(slots[i]))
        target = next((i for i in pref if len(slots[i]) < k), None)
        if target is None:  # voix préférée pleine : à tour de rôle
            for _ in range(n_voices):
                if len(slots[rr % n_voices]) < k:
                    target = rr % n_voices
                    rr += 1
                    break
                rr += 1
        if target is None:
            break
        slots[target].append((word, weights.get(word, 0.0)))
    return slots


def voice_multi(run_dir: Path, cfg: dict, wins: list[dict], log):
    """N voix en parallèle : une piste par voix, audio/vK/NNNN.wav et audio/vK/words/NNNN_JJ.wav."""
    n = int(cfg["voice"]["n_voices"])
    spk = Speaker(cfg)
    gap = cfg["voice"].get("gap_s", 0.25)
    meta = []
    for k, w in enumerate(wins):
        slots = assign_words(w, n)
        entry = {"window": k, "voices": []}
        for v, words in enumerate(slots):
            name = VOICES[v][0]
            vdir = run_dir / "audio" / f"v{v}"
            (vdir / "words").mkdir(parents=True, exist_ok=True)
            wavs = [spk.say_as(x, name) for x, _ in words]
            for j, x in enumerate(wavs):
                sf.write(vdir / "words" / f"{k:04d}_{j:02d}.wav", x, SR)
            out = build_window(wavs, [g for _, g in words], gap, False) if wavs else np.zeros(1, np.float32)
            sf.write(vdir / f"{k:04d}.wav", out, SR)
            entry["voices"].append({"voice": name, "words": [x for x, _ in words], "weights": [g for _, g in words],
                                    "duration": round(len(out) / SR, 3) if wavs else 0.0})
        entry["words"] = [x for vv in entry["voices"] for x in vv["words"]]
        entry["duration"] = max(vv["duration"] for vv in entry["voices"])
        meta.append(entry)
    (run_dir / "voice.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{n} voix, {len(wins)} tranches, {sum(len(m['words']) for m in meta)} mots dits")


def build_window(wavs: list[np.ndarray], weights: list[float], gap_s: float, polyphony: bool) -> np.ndarray:
    gap = np.zeros(int(gap_s * SR), np.float32)
    if not wavs:
        return np.zeros(int(0.1 * SR), np.float32)
    if not polyphony:
        out = []
        for x in wavs:
            out += [x, gap]
        return np.concatenate(out)
    # polyphonie : départs décalés de 0,35 s, volume proportionnel au poids
    hop = int(0.35 * SR)
    n = hop * (len(wavs) - 1) + max(len(x) for x in wavs)
    out = np.zeros(n, np.float32)
    mw = max(weights) or 1.0
    for i, (x, g) in enumerate(zip(wavs, weights)):
        out[i * hop: i * hop + len(x)] += x * (0.35 + 0.65 * g / mw)
    return out / max(1.0, np.abs(out).max())


def voice(run_dir: Path, cfg: dict | None = None):
    run_dir = Path(run_dir)
    cfg = cfg or run_config(run_dir)
    log = Log(run_dir, "voice")
    vc = cfg["voice"]
    wins = json.loads((run_dir / "windows.json").read_text(encoding="utf-8"))
    adir = run_dir / "audio"
    (adir / "words").mkdir(parents=True, exist_ok=True)
    if int(vc.get("n_voices", 1)) > 1:
        voice_multi(run_dir, cfg, wins, log)
        log.done()
        return
    spk = Speaker(cfg)
    meta = []
    if vc.get("mode") == "phrase":  # un petit LM relie les mots de chaque tranche en une phrase (introspect/phrases.py)
        pj = run_dir / "phrases.json"
        if not pj.exists():
            from .phrases import build
            build(run_dir, cfg)
        phrases = {p["window"]: p for p in json.loads(pj.read_text(encoding="utf-8"))["phrases"]}
        for k, w in enumerate(wins):
            ph = phrases[k]["phrase"]
            out = spk.say(ph)
            sf.write(adir / "words" / f"{k:04d}_00.wav", out, SR)
            sf.write(adir / f"{k:04d}.wav", out, SR)
            meta.append({"window": k, "words": phrases[k]["words"], "phrase": ph, "weights": [1.0],
                         "duration": round(len(out) / SR, 3), "langs": ["fr"]})
        (run_dir / "voice.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"{len(wins)} phrases dites, durée totale {sum(m['duration'] for m in meta):.1f} s")
        log.done()
        return
    for k, w in enumerate(wins):
        ws = words_for(w, vc.get("mode", "new_words"))
        weights = [dict(w["words"]).get(x, 0.0) for x in ws]
        wavs = [spk.say(x) for x in ws]
        for j, x in enumerate(wavs):
            sf.write(adir / "words" / f"{k:04d}_{j:02d}.wav", x, SR)
        out = build_window(wavs, weights, vc.get("gap_s", 0.25), vc.get("polyphony", False))
        sf.write(adir / f"{k:04d}.wav", out, SR)
        meta.append({"window": k, "words": ws, "weights": weights, "duration": round(len(out) / SR, 3),
                     "langs": [lang_of(x) for x in ws]})
    (run_dir / "voice.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{len(wins)} fragments, durée totale {sum(m['duration'] for m in meta):.1f} s")
    log.done()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    voice(Path(a.run))


if __name__ == "__main__":
    main()
