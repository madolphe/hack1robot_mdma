"""Expériences pour interface/robot.html, au format commun :
  {id, drug, prompt, ctx, video, audio, duration, words:[{w, nl, t}], slices:[{t0, t1, alpha, phase, words:[[mot, poids]]}]}

- depuis une fiche .md produite par introspect/describe.py (+ la vidéo du run, son mixé inclus) ;
- depuis la session q1_rave déjà préparée (interface/data/q1_rave.json), run sans drogue -> expérience « neutre ».

Dans la fiche : la dose de chaque image est la force perçue d (0..1), positive en profil euphorique, négative en
profil sombre (descente, contrecoup) ; le texte écrit est dédoublonné entre images qui se chevauchent, et chaque mot
est placé dans le temps à proportion de sa position dans le texte de l'image.

    mac/.venv-video/bin/python interface/prepare_md.py interface/media/<fiche>.md --video media/<video>.mp4 --id mdma_v7
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "interface" / "data"


def ts(s: str) -> float:  # "1:03.18" -> 63.18
    m, sec = s.split(":")
    return int(m) * 60 + float(sec)


def strip_overlap(prev: str, cur: str) -> str:
    """Retire du début de cur le recouvrement avec la fin de prev (les images se chevauchent de quelques tokens)."""
    best = 0
    for k in range(1, min(len(prev), len(cur), 80) + 1):
        if prev.endswith(cur[:k]):
            best = k
    return cur[best:] if best >= 3 else cur


def parse_md(path: Path) -> dict:
    md = path.read_text(encoding="utf-8")
    consigne = re.search(r"consigne : « (.*?) »", md, re.S)
    drug = re.search(r"\*\*Drogue\*\* : (\w+)", md)
    blocks = re.split(r"\n### Image \d+/\d+ · ", md)[1:]
    slices, text_parts = [], []
    prev_text = ""
    for b in blocks:
        t0, t1 = [ts(x) for x in re.match(r"([\d:.]+) → ([\d:.]+)", b).groups()]
        phase = re.search(r"phase « (.*?) »", b).group(1)
        d = float(re.search(r"force perçue d = ([\d.]+)", b).group(1))
        profile = re.search(r"\*\*Profil de phase\*\* : (\w+)", b).group(1)
        text = re.search(r"\*\*Texte écrit par le modèle\*\* \(tokens [^)]*\) : « (.*?) »\n- \*\*Mots lus", b, re.S).group(1)
        words = [[w.strip(), float(p)] for w, p in re.findall(r"([^,]+?) ([\d.]+)(?:,|$)", re.search(r"\*\*Mots lus \(J-lens\), avec poids\*\* : (.*)", b).group(1))]
        lyric = re.search(r"\*\*Paroles\*\* : « (.*?) »", b)
        alpha = -d if profile == "sombre" else d
        slices.append({"t0": t0, "t1": t1, "alpha": round(alpha, 3), "phase": phase, "profile": profile, "words": words,
                       "lyric": lyric.group(1) if lyric else ""})
        own = strip_overlap(prev_text, text)
        prev_text = text
        text_parts.append((t0, t1, own))
    # mots du texte, avec leurs sauts de ligne et l'instant où ils apparaissent
    out_words = []
    for t0, t1, txt in text_parts:
        toks = re.findall(r"\n+|[^\s]+", txt)
        n = sum(1 for x in toks if x.strip()) or 1
        j, nl = 0, 0
        for x in toks:
            if not x.strip():
                nl = min(2, x.count("\n"))
                continue
            out_words.append({"w": x, "nl": nl, "t": round(t0 + (t1 - t0) * j / n, 3)})
            j, nl = j + 1, 0
    return {"drug": drug.group(1) if drug else "neutral", "prompt": consigne.group(1).strip() if consigne else "",
            "ctx": "", "duration": slices[-1]["t1"], "words": out_words, "slices": slices}


def from_q1_rave() -> dict:
    """Run sans drogue : texte et instants depuis interface/data/q1_rave.json, mots J-space par tranche de 14 tokens."""
    s = json.loads((OUT / "q1_rave.json").read_text(encoding="utf-8"))
    media = s["media"][0]
    words = [{"w": w["w"], "nl": w["nl"], "t": t} for w, t in zip(s["words"], media["times"])]
    slices = []
    for i, w in enumerate(s["words"][::8]):  # une « image » tous les 8 mots : mots J-space agrégés sur ses tokens
        lo = w["t0"]; hi = s["words"][min(len(s["words"]) - 1, (i + 1) * 8)]["t0"]
        count = {}
        for t in range(lo, max(lo + 1, hi)):
            for k, ws in enumerate(s["tokens"][t]["l"]):
                if ws and ws[0]:
                    count[ws[0]] = count.get(ws[0], 0) + 1 + k / 40
        tot = sum(count.values()) or 1
        top = sorted(count.items(), key=lambda x: -x[1])[:12]
        t0 = media["times"][i * 8]; t1 = media["times"][min(len(media["times"]) - 1, (i + 1) * 8)]
        slices.append({"t0": t0, "t1": t1, "alpha": 0, "phase": "", "profile": "neutre", "words": [[a, round(b / tot, 3)] for a, b in top], "lyric": ""})
    return {"drug": "neutral", "prompt": s["question"], "ctx": s["context"], "duration": media["duration"], "words": words, "slices": slices,
            "video": media["video"], "audio": media["audio"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md", nargs="?")
    ap.add_argument("--video", help="chemin servi de la vidéo (relatif à interface/), son mixé inclus")
    ap.add_argument("--id", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    exps = {}
    idx = OUT / "experiences.json"
    if idx.exists():
        exps = json.loads(idx.read_text(encoding="utf-8"))
    if a.md:
        e = parse_md(Path(a.md))
        e["video"], e["audio"] = a.video, None
        e["id"] = a.id or Path(a.md).stem
        (OUT / f"exp_{e['id']}.json").write_text(json.dumps(e, ensure_ascii=False), encoding="utf-8")
        exps[e["drug"]] = e["id"]
        print(f"{e['id']} : {e['drug']}, {len(e['slices'])} images, {len(e['words'])} mots, {e['duration']:.1f} s, "
              f"α de {min(s['alpha'] for s in e['slices'])} à {max(s['alpha'] for s in e['slices'])}")
    if (OUT / "q1_rave.json").exists():
        e = from_q1_rave(); e["id"] = "neutre_q1_rave"
        (OUT / f"exp_{e['id']}.json").write_text(json.dumps(e, ensure_ascii=False), encoding="utf-8")
        exps.setdefault("neutral", e["id"])
        print(f"{e['id']} : {len(e['slices'])} tranches, {len(e['words'])} mots, {e['duration']:.1f} s")
    idx.write_text(json.dumps(exps, ensure_ascii=False, indent=1), encoding="utf-8")
    print("expériences :", exps)


if __name__ == "__main__":
    sys.exit(main())
