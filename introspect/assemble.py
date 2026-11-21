"""Passe 2d : montage. Images en fondu enchaîné + piste audio calée, MP4 H.264 1080p."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image
from scipy.signal import resample_poly

from .common import Log, run_config

SR = 44100


def ffmpeg_bin() -> str:
    b = shutil.which("ffmpeg")
    if b:
        return b
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def load_audio(path: Path, sr_out: int = SR) -> np.ndarray:
    """Charge un wav en stéréo float32 [n, 2] à 44,1 kHz."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    x = x[:, :2]
    if sr != sr_out:
        from math import gcd

        g = gcd(sr, sr_out)
        x = resample_poly(x, sr_out // g, sr // g, axis=0).astype(np.float32)
    return x


def fit(img: Image.Image, W: int, H: int) -> np.ndarray:
    s = max(W / img.width, H / img.height)
    im = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    l, t = (im.width - W) // 2, (im.height - H) // 2
    return np.asarray(im.crop((l, t, l + W, t + H)).convert("RGB"))


def durations(run_dir: Path, cfg: dict, n: int) -> list[float]:
    vmin = cfg["video"]["min_duration_s"]
    vj = run_dir / "voice_fx.json"
    if not (cfg["sound"]["voice_fx"].get("enabled", True) and vj.exists()):
        vj = run_dir / "voice.json"
    vd = {m["window"]: m["duration"] for m in json.loads(vj.read_text())} if vj.exists() else {}
    return slice_durations([vd.get(k, 0.0) for k in range(n)], cfg)


def slice_durations(voice_durs: list[float], cfg: dict) -> list[float]:
    """Durée de chaque tranche : max(voix, minimum) ; si une durée cible est fixée et
    que le total est plus court, on étire uniformément pour l'atteindre. Avec video.fit_target,
    un total plus long est aussi resserré (sans passer sous le minimum) : la voix d'une tranche
    et la queue de ses effets débordent alors sur l'image suivante."""
    vmin = cfg["video"]["min_duration_s"]
    d = [max(vmin, x) for x in voice_durs]
    T = cfg["video"].get("target_duration_s")
    if T and d and sum(d) < T:
        extra = (T - sum(d)) / len(d)
        d = [x + extra for x in d]
    elif T and d and sum(d) > T and cfg["video"].get("fit_target"):
        for _ in range(8):  # on resserre les tranches au-dessus du minimum jusqu'à tenir la cible
            free = sum(x for x in d if x > vmin)
            over = sum(d) - T
            if over <= 1e-3 or free <= 0:
                break
            s = max(0.0, (free - over) / free)
            d = [max(vmin, x * s) if x > vmin else x for x in d]
    return d


def place(track: np.ndarray, clip: np.ndarray, start: int, gain: float = 1.0):
    end = min(len(track), start + len(clip))
    if end > start:
        track[start:end] += clip[: end - start] * gain


def voice_track_single(run_dir, cfg, durs, total):
    tr = np.zeros((total, 2), np.float32)
    use_fx = cfg["sound"]["voice_fx"].get("enabled", True)
    t = 0
    for k, d in enumerate(durs):
        p = run_dir / "audio" / f"voice_fx_{k:04d}.wav"
        if not (use_fx and p.exists()):
            p = run_dir / "audio" / f"{k:04d}.wav"
        if p.exists():
            place(tr, load_audio(p), t)
        t += int(d * SR)
    return tr


def multi_voice_layout(run_dir, cfg, durs) -> tuple[list, int]:
    """Voix en parallèle : (liste (fichier, départ en échantillons, pan), fin de la dernière voix)."""
    from .voice import VOICES

    meta = json.loads((run_dir / "voice.json").read_text(encoding="utf-8"))
    n = int(cfg["voice"].get("n_voices", 1))
    use_fx = cfg["sound"]["voice_fx"].get("enabled", True)
    busy = [0] * n  # fin du fragment précédent de chaque voix
    starts = np.cumsum([0] + [int(d * SR) for d in durs])
    items, end = [], 0
    for k, m in enumerate(meta):
        for v, vv in enumerate(m.get("voices", [])):
            if not vv["words"]:
                continue
            p = run_dir / "audio" / f"v{v}" / (f"fx_{k:04d}.wav" if use_fx else f"{k:04d}.wav")
            if use_fx and not p.exists():
                p = run_dir / "audio" / f"v{v}" / f"{k:04d}.wav"
            if not p.exists():
                continue
            t0 = max(int(starts[k] + v * durs[k] / n * SR), busy[v])  # décalage, jamais sur soi-même
            L = sf.info(p).frames * SR // sf.info(p).samplerate
            busy[v] = t0 + L
            end = max(end, t0 + L)
            items.append((p, t0, None if use_fx else VOICES[v][2]))
    return items, end


def voice_track(run_dir, cfg, durs, total):
    if int(cfg["voice"].get("n_voices", 1)) > 1:
        tr = np.zeros((total, 2), np.float32)
        items, _ = multi_voice_layout(run_dir, cfg, durs)
        n = int(cfg["voice"]["n_voices"])
        for p, t0, pan in items:
            x = load_audio(p)
            if pan is not None:  # voix brute : panoramique de base de la voix
                x = x * np.array([np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)], np.float32) * 1.41
            place(tr, x / np.sqrt(n / 2) if n > 2 else x, t0)
        return tr
    return voice_track_single(run_dir, cfg, durs, total)


def sono_track(run_dir, cfg, durs, total):
    tr = np.zeros((total, 2), np.float32)
    bed = run_dir / "audio" / "sono_bed.wav"
    if cfg["sound"]["sono"].get("segments") == "phase" and bed.exists():  # nappe continue (sound.sono_bed)
        place(tr, load_audio(bed), 0)
        return tr
    c = int(cfg["video"]["crossfade_s"] * SR)
    t = 0
    for k, d in enumerate(durs):
        p = run_dir / "audio" / f"sono_{k:04d}.wav"
        if p.exists():
            x = load_audio(p)
            n = int(d * SR) + c
            x = x[:n]
            env = np.ones(len(x), np.float32)
            r = min(c, len(x) // 2)
            if r > 0:
                env[:r] = np.linspace(0, 1, r)
                env[-r:] = np.linspace(1, 0, r)
            place(tr, x * env[:, None], max(0, t - c // 2))
        t += int(d * SR)
    return tr


def duck(sono: np.ndarray, voice: np.ndarray, db: float, smooth_s: float = 0.15) -> np.ndarray:
    env = np.abs(voice).max(axis=1)
    win = int(smooth_s * SR)
    env = np.convolve(env, np.ones(win) / win, mode="same")
    active = np.clip(env / (env.max() + 1e-9) * 8, 0, 1)
    gain = 1 - active * (1 - 10 ** (db / 20))
    return sono * gain[:, None]


def build_audio(run_dir: Path, cfg: dict, mode: str, durs: list[float]) -> np.ndarray:
    total = int(sum(durs) * SR) + int(cfg["video"]["crossfade_s"] * SR)
    if int(cfg["voice"].get("n_voices", 1)) > 1 and mode != "sonorisation":
        _, end = multi_voice_layout(run_dir, cfg, durs)
        total = max(total, end + SR)  # dernière image jusqu'à la fin de la dernière voix + 1 s
    if mode == "voix":
        mix = voice_track(run_dir, cfg, durs, total)
    elif mode == "sonorisation":
        mix = sono_track(run_dir, cfg, durs, total)
    else:
        v = voice_track(run_dir, cfg, durs, total)
        s = sono_track(run_dir, cfg, durs, total)
        s = s / (np.abs(s).max() + 1e-9) * 0.5
        mix = v + duck(s, v, cfg["sound"].get("duck_db", -6), cfg["sound"].get("duck_smooth_s", 0.15))
    peak = np.abs(mix).max()
    if peak > 0:
        mix = mix / peak * 0.9
    target = cfg["sound"].get("loudness_db")  # niveau moyen visé (dBFS RMS) ; les crêtes passent par un limiteur
    if target is not None and peak > 0:
        from pedalboard import Limiter, Pedalboard

        def rms_db(x):
            return 20 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-12)
        g = 0.0
        for _ in range(4):  # le limiteur ramène son seuil à 0 dBFS : on corrige le gain jusqu'à tenir la cible
            y = Pedalboard([Limiter(threshold_db=-6.0, release_ms=150)])((mix * 10 ** (g / 20)).T.astype(np.float32), SR).T
            y = y / max(float(np.abs(y).max()), 1e-9) * 10 ** (-1.5 / 20)  # crêtes à -1,5 dBFS : marge pour l'AAC
            g += target - rms_db(y)
            if abs(target - rms_db(y)) < 0.3:
                break
        mix = y
    return mix


def assemble(run_dir: Path, cfg: dict | None = None, mode: str | None = None, raw_voice: bool = False) -> Path:
    run_dir = Path(run_dir)
    cfg = cfg or run_config(run_dir)
    mode = mode or cfg["sound"]["mode"]
    suffix = ""
    if raw_voice:
        cfg["sound"]["voice_fx"]["enabled"] = False
        suffix = "_brute"
    log = Log(run_dir, f"assemble_{mode}")
    vc = cfg["video"]
    W, H, fps, c = vc.get("width", 1920), vc.get("height", 1080), vc["fps"], vc["crossfade_s"]
    wins = json.loads((run_dir / "windows.json").read_text())
    n = len(wins)
    durs = durations(run_dir, cfg, n)
    audio = build_audio(run_dir, cfg, mode, durs)
    wav_path = run_dir / f"audio_{mode}{suffix}.wav"
    sf.write(wav_path, audio, SR)

    # liste d'images : (fichier, durée), intermédiaires répartis dans la tranche
    fj = run_dir / "frames.json"
    manifest = json.loads(fj.read_text()) if fj.exists() else [{"window": k, "file": f"{k:04d}.png", "intermediate": False} for k in range(n)]
    shots = []
    for k in range(n):
        files = [m["file"] for m in manifest if m["window"] == k and m["intermediate"]] + [f"{k:04d}.png"]
        # les intermédiaires préparent la tranche k : ils occupent la 1re moitié de sa durée
        if len(files) > 1:
            dm = durs[k] * 0.5 / (len(files) - 1)
            shots += [(f, dm) for f in files[:-1]] + [(files[-1], durs[k] * 0.5)]
        else:
            shots.append((files[0], durs[k]))
    imgs = {}

    def get(f):
        if f not in imgs:
            imgs.clear() if len(imgs) > 4 else None
            imgs[f] = fit(Image.open(run_dir / "frames" / f), W, H).astype(np.float32)
        return imgs[f]

    starts = np.cumsum([0] + [d for _, d in shots])
    total_t = len(audio) / SR
    nframes = int(total_t * fps)
    out = run_dir / f"out_{mode}{suffix}.mp4"
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-i", str(wav_path), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    half = c / 2
    static: dict[str, bytes] = {}
    for fi in range(nframes):
        t = fi / fps
        i = min(len(shots) - 1, int(np.searchsorted(starts, t, side="right") - 1))
        # fondu : autour de chaque frontière, de t_b - c/2 à t_b + c/2
        if i + 1 < len(shots) and t > starts[i + 1] - half:
            a = (t - (starts[i + 1] - half)) / c
            frame = get(shots[i][0]) * (1 - a) + get(shots[i + 1][0]) * a
        elif i > 0 and t < starts[i] + half:
            a = 0.5 + (t - starts[i]) / c
            frame = get(shots[i - 1][0]) * (1 - a) + get(shots[i][0]) * a
        else:
            f = shots[i][0]
            if f not in static:
                static.clear()
                static[f] = np.clip(get(f), 0, 255).astype(np.uint8).tobytes()
            proc.stdin.write(static[f])
            continue
        proc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg a échoué")
    if mode == cfg["sound"]["mode"] and not suffix:
        shutil.copyfile(out, run_dir / "out.mp4")
    log(f"{out.name} : {n} tranches, {total_t:.1f} s, {nframes} images vidéo")
    log.done()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", choices=["voix", "sonorisation", "mixte"])
    ap.add_argument("--raw-voice", action="store_true", help="voix brute, sans effets")
    a = ap.parse_args(argv)
    assemble(Path(a.run), mode=a.mode, raw_voice=a.raw_voice)


if __name__ == "__main__":
    main()
