"""Données de l'interface finale (interface/index.html), à partir des traces converties dans runs/.

Pour chaque texte : le prompt (question et prompt système), les mots générés avec la plage de tokens de chacun,
et pour chaque token les 3 premiers mots lus par la J-lens à chaque couche de la bande (19-44), avec l'entropie.
Plus les mots-clés et les phases des drogues du catalogue.

    mac/.venv-video/bin/python interface/prepare.py
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT.parent / "hack1robo_activations"
COUPLES = ["q1_rave", "q1_aucun", "q2_eglise", "q2_garde_a_vue", "q3_aucun", "q3_rave"]
DRUGS = ["lsd", "mdma"]
# vidéos et sons déjà générés pour q1_rave : copies légères dans interface/media/ (960x540 sans son + WAV)
MEDIA = {"q1_rave": [
    {"id": "fondus", "run": "q1_rave", "video": "q1_rave_fondus.mp4", "audio": "q1_rave_voix.wav"},
    {"id": "camera3d", "run": "q1_rave", "video": "q1_rave_camera3d.mp4", "audio": "q1_rave_voix.wav"},
    {"id": "phrases", "run": "q1_rave_phrase", "video": "q1_rave_phrases.mp4", "audio": "q1_rave_phrases.wav"},
]}


def envelope(wav: Path, fps: int = 50) -> list[float]:
    """Enveloppe RMS du son, fps valeurs par seconde, normalisée à 1."""
    x, sr = sf.read(str(wav), always_2d=True)
    x = x.mean(1)
    hop = sr // fps
    n = len(x) // hop
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1))
    return [round(float(v), 3) for v in rms / (rms.max() + 1e-9)]


def word_times(run: Path, words: list[dict]) -> tuple[list[float], float]:
    """Instant où chaque mot du texte est atteint dans la vidéo : la tranche k commence au token windows[k].start
    à l'instant où assemble.py la place ; entre deux, interpolation linéaire sur les tokens."""
    import sys
    sys.path.insert(0, str(ROOT))
    from introspect.assemble import durations
    from introspect.common import run_config
    wins = json.loads((run / "windows.json").read_text(encoding="utf-8"))
    durs = durations(run, run_config(run), len(wins))
    starts = np.cumsum([0] + durs)
    tok = [w["start"] for w in wins] + [wins[-1]["end"]]
    return [round(float(np.interp(w["t0"], tok, starts)), 3) for w in words], float(starts[-1])


def session(c: str) -> dict | None:
    run = ROOT / "runs" / c
    if not (run / "trace.jsonl").exists():
        print(f"{c} : run absent (python mac/traces2run.py {PACK / c} --run-id {c})")
        return None
    meta = json.loads((PACK / c / "meta.json").read_text(encoding="utf-8"))
    trace = [json.loads(l) for l in (run / "trace.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    layers = sorted(trace[0]["layers"], key=int)
    tokens, words, nl = [], [], 0
    for t, rec in enumerate(trace):
        tokens.append({"l": [[w for w, _ in rec["layers"][l][:3]] for l in layers],
                       "e": round(mean(rec["entropy"].values()), 3)})
        tok = rec["token"].replace("<0x0A>", "\n")
        if "\n" in tok:
            nl += tok.count("\n")
            tok = tok.replace("\n", "")
            if not tok:
                continue
        piece = tok.replace("▁", " ")
        starts = piece.startswith(" ") or nl > 0 or not words
        piece = piece.lstrip(" ").replace("**", "").replace("#", "")
        if not piece:
            continue
        if starts:
            words.append({"w": piece, "nl": min(nl, 2), "t0": t, "t1": t})
        else:
            words[-1]["w"] += piece
            words[-1]["t1"] = t
        nl = 0
    words = [w for w in words if w["w"].strip()]
    question = meta["question"]
    system = meta["system_prompt"].replace(" Tu dissertes longuement et librement sur le sujet donné.", "").strip()
    return {"id": c, "question": question, "context": system if meta["environment"] != "aucun" else "",
            "environment": meta["environment"], "layers": [int(l) for l in layers], "n_layers": 48,
            "tokens": tokens, "words": words}


def main():
    out = ROOT / "interface" / "data"
    out.mkdir(parents=True, exist_ok=True)
    ids = []
    for c in COUPLES:
        s = session(c)
        if s:
            s["media"] = []
            for m in MEDIA.get(c, []):
                if not (ROOT / "interface" / "media" / m["video"]).exists():
                    continue
                times, total = word_times(ROOT / "runs" / m["run"], s["words"])
                env = envelope(ROOT / "interface" / "media" / m["audio"])
                s["media"].append({"id": m["id"], "video": f"media/{m['video']}", "audio": f"media/{m['audio']}",
                                   "duration": round(len(env) / 50, 2), "times": times, "env": env, "env_fps": 50})
                print(f"  média {m['id']} : {len(env) / 50:.1f} s, mots de {times[0]:.1f} à {times[-1]:.1f} s")
            (out / f"{c}.json").write_text(json.dumps(s, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            ids.append(c)
            print(f"{c} : {len(s['words'])} mots, {len(s['tokens'])} tokens, couches {s['layers'][0]}-{s['layers'][-1]}")
    cat = yaml.safe_load((ROOT / "configs" / "drugs.yaml").read_text(encoding="utf-8"))
    drugs = {d: {"label": cat["drugs"][d]["label"],
                 "phases": [{k: p[k] for k in ("name", "start", "end", "level", "words")} for p in cat["drugs"][d]["phases"]]}
             for d in DRUGS}
    (out / "index.json").write_text(json.dumps({"sessions": ids, "drugs": drugs, "band": cat["band"]}, ensure_ascii=False),
                                    encoding="utf-8")
    print(f"{len(ids)} sessions, drogues {', '.join(DRUGS)} -> {out}")


if __name__ == "__main__":
    main()
