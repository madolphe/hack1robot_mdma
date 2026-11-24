"""Prépare les données du labo de typo (fontlab/index.html) à partir des runs existants.

Pour chaque run de runs/ qui a des tranches (windows.json) : la ligne de temps (début et durée de chaque tranche,
comme assemble.py), le texte dit (phrase du mode « phrase », sinon les mots dits), les images de la tranche,
l'audio, et l'état de drogue enregistré dans la trace s'il existe. Ajoute le catalogue des drogues.

    python fontlab/prepare.py          # puis : python -m http.server 8765  (depuis hack1robo/), ouvrir /fontlab/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from introspect.assemble import durations
from introspect.common import run_config


def run_timeline(d: Path) -> dict | None:
    if not (d / "windows.json").exists() or not (d / "config.yaml").exists():
        return None
    cfg = run_config(d)
    wins = json.loads((d / "windows.json").read_text(encoding="utf-8"))
    durs = durations(d, cfg, len(wins))
    voice = {m["window"]: m for m in json.loads((d / "voice.json").read_text(encoding="utf-8"))} if (d / "voice.json").exists() else {}
    vfx = {m["window"]: m for m in json.loads((d / "voice_fx.json").read_text())} if (d / "voice_fx.json").exists() else {}
    phrases = {}
    if (d / "phrases.json").exists():
        phrases = {p["window"]: p["phrase"] for p in json.loads((d / "phrases.json").read_text(encoding="utf-8"))["phrases"]}
    man = json.loads((d / "frames.json").read_text()) if (d / "frames.json").exists() else []
    trace = [json.loads(l) for l in (d / "trace.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    slices, t = [], 0.0
    for k, (w, dur) in enumerate(zip(wins, durs)):
        imgs = sorted((m for m in man if m["window"] == k and not m.get("intermediate")), key=lambda m: m.get("sub", 0))
        imgs = [f"../runs/{d.name}/frames/{m['file']}" for m in imgs] or [f"../runs/{d.name}/frames/{k:04d}.png"]
        v = voice.get(k, {})
        text = phrases.get(k) or v.get("phrase") or " ".join(v.get("words") or w["new"][:2] or [w["words"][0][0]])
        # état de drogue enregistré (runs avec drogue) : α max et phase sur les tokens de la tranche
        drug = None
        recs = trace[w["start"]:w["end"]]
        if recs and recs[0].get("drug"):
            st = recs[-1]["drug"][0]
            drug = {"name": st.get("word"), "phase": st.get("phase"), "alpha": max(r["drug"][0]["alpha"] for r in recs)}
        slices.append({"k": k, "start": round(t, 3), "dur": round(dur, 3),
                       "voice": round(vfx.get(k, v).get("duration", dur), 3), "text": text,
                       "words": [x for x, _ in w["words"][:6]], "images": imgs, "drug": drug})
        t += dur
    audio = next((f"../runs/{d.name}/{n}" for n in ("audio_voix_brute.wav", "audio_voix.wav") if (d / n).exists()), None)
    return {"id": d.name, "question": cfg["input"].get("phrase"), "environment": (cfg.get("source_traces") or {}).get("environment"),
            "mode": cfg["voice"].get("mode"), "total": round(t, 3), "audio": audio, "slices": slices}


def main():
    out = ROOT / "fontlab" / "data"
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for d in sorted((ROOT / "runs").iterdir()):
        tl = run_timeline(d) if d.is_dir() else None
        if tl:
            (out / f"{d.name}.json").write_text(json.dumps(tl, ensure_ascii=False), encoding="utf-8")
            runs.append({"id": d.name, "question": tl["question"], "environment": tl["environment"], "mode": tl["mode"],
                         "slices": len(tl["slices"]), "total": tl["total"], "audio": bool(tl["audio"])})
            print(f"{d.name} : {len(tl['slices'])} tranches, {tl['total']:.1f} s, audio {'oui' if tl['audio'] else 'non'}")
    cat = yaml.safe_load((ROOT / "configs" / "drugs.yaml").read_text(encoding="utf-8"))
    drugs = {k: {"label": v["label"], "phases": [{"name": p["name"], "start": p["start"], "end": p["end"], "level": p["level"]}
                                                 for p in v["phases"]]} for k, v in cat["drugs"].items()}
    (out / "index.json").write_text(json.dumps({"runs": runs, "drugs": drugs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(runs)} runs, {len(drugs)} drogues -> {out}")


if __name__ == "__main__":
    main()
