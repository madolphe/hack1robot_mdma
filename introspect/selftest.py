"""Tests rapides sur CPU (sans GPU ni poids de modèle, sauf le tokenizer) : ./run.sh test [config]

Vérifie la config résolue, les mots du tokenizer, les tirages de drogue et les tranches."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from .common import ROOT, finalize, load_config


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f" : {detail}" if detail else ""), flush=True)
    return bool(cond)


def main():
    from transformers import AutoTokenizer

    from . import windows
    from .drugs import Schedule, is_single
    from .lensmodel import model_source, single_token_id, stop_token_ids, word_starts

    cfg = finalize(load_config(sys.argv[1] if len(sys.argv) > 1 else ROOT / "configs" / "default.yaml"))
    tok = AutoTokenizer.from_pretrained(model_source(cfg))
    ok = True

    words = word_starts(tok, len(tok))
    ok &= check("débuts de mots", len(words) > 10000, f"{len(words)} tokens sur {len(tok)}")
    for w in ("silence", "ocean", "dream"):
        tid = single_token_id(tok, w)
        ok &= check(f"« {w} » en un token", tid is not None and words.get(tid, "").lower() == w, f"id {tid}")
    accented = [w for w in words.values() if any(c in w for c in "éèàç")]
    ok &= check("mots accentués décodés", len(accented) > 50, ", ".join(accented[:5]))

    class _M:  # stop_token_ids ne lit que generation_config
        generation_config = None
    stops = stop_token_ids(_M(), tok)
    ok &= check("tokens d'arrêt", stops and all(isinstance(i, int) for i in stops), str(sorted(stops)))

    msgs = [{"role": "system", "content": cfg["input"]["system"]}, {"role": "user", "content": "Une machine peut-elle rêver ?"}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                     **(cfg["input"].get("chat_template_kwargs") or {}))
    ok &= check("gabarit sans bloc de réflexion ouvert", not prompt.rstrip().endswith("<think>"), repr(prompt[-40:]))

    sched = Schedule("lsd", 1.0, 400, 1, tok=tok, log=lambda *a: None)
    phases = [sched.phase_name(s) for s in range(0, 400, 40)]
    ok &= check("phases LSD dans l'ordre", phases[0] == "Montée" and phases[-1] == "Retour", " → ".join(dict.fromkeys(phases)))
    drawn = sched.draw(0)
    ok &= check("tirage de mots à un token", drawn and all(is_single(tok, w) for w in drawn), str(drawn))

    trace = []
    pieces = tok.encode(" Le silence est une forme de langage et le langage rêve", add_special_tokens=False)
    for i, t in enumerate(pieces * 6):
        trace.append({"step": i, "token_id": t, "token": tok.convert_ids_to_tokens(t), "text": tok.decode([t]),
                      "layers": {"20": [["silence", 0.3], ["whisper", 0.2]], "21": [["void", 0.1]]}, "entropy": {}, "drug": []})
    late = windows.step_scores(trace, cfg)[40]  # « silence » est dans les 32 derniers tokens écrits
    ok &= check("filtre des échos", "silence" not in late and "whisper" in late, str(sorted(late)))
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in trace))
        wins = windows.build(Path(d), cfg)
    ok &= check("tranches", len(wins) >= 2 and all(x["prompt"] for x in wins), f"{len(wins)} tranches")
    print("TOUT VA BIEN" if ok else "ÉCHECS", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
