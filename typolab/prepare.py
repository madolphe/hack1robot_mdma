"""Données du labo de typo (typolab/index.html) : 6 textes générés par Gemma, découpés en mots, avec pour chaque mot
l'entropie moyenne de la lecture J-lens (couches 19-44) sur ses tokens : l'incertitude du réseau à ce moment-là.

Prérequis : les runs correspondants (python mac/traces2run.py <paquet>/<couple> --run-id <couple>).

    mac/.venv-video/bin/python typolab/prepare.py
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT.parent / "hack1robo_activations"
COUPLES = ["q1_aucun", "q1_rave", "q2_eglise", "q2_garde_a_vue", "q3_aucun", "q3_rave"]


def words_of(trace: list[dict]) -> list[dict]:
    """Regroupe les tokens Gemma en mots : un mot commence à chaque pièce « ▁… » ou après un saut de ligne."""
    out: list[dict] = []
    nl = 0
    for rec in trace:
        tok = rec["token"].replace("<0x0A>", "\n")
        ent = mean(rec["entropy"].values())
        if "\n" in tok:
            nl += tok.count("\n")
            tok = tok.replace("\n", "")
            if not tok:
                continue
        piece = tok.replace("▁", " ")
        starts = piece.startswith(" ") or nl > 0 or not out
        piece = piece.lstrip(" ")
        if not piece:
            continue
        if starts:
            out.append({"w": piece, "nl": min(nl, 2), "ents": [ent]})
        else:
            out[-1]["w"] += piece
            out[-1]["ents"].append(ent)
        nl = 0
    for o in out:
        o["ent"] = round(mean(o.pop("ents")), 4)
        o["w"] = o["w"].replace("**", "").replace("#", "").strip()  # balises Markdown du texte généré
    return [o for o in out if o["w"]]


def main():
    texts = []
    for c in COUPLES:
        run = ROOT / "runs" / c
        if not (run / "trace.jsonl").exists():
            print(f"{c} : run absent, ignoré (python mac/traces2run.py {PACK / c} --run-id {c})")
            continue
        meta = json.loads((PACK / c / "meta.json").read_text(encoding="utf-8"))
        trace = [json.loads(l) for l in (run / "trace.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        ws = words_of(trace)
        texts.append({"id": c, "question": meta["question"], "environment": meta["environment"], "words": ws})
        print(f"{c} : {len(ws)} mots, entropie {min(w['ent'] for w in ws):.2f}-{max(w['ent'] for w in ws):.2f}")
    out = ROOT / "typolab" / "data"
    out.mkdir(parents=True, exist_ok=True)
    (out / "texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    print(f"{len(texts)} textes -> {out / 'texts.json'}")


if __name__ == "__main__":
    main()
