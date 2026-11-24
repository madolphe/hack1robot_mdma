"""Mode voix « phrase » : un petit LM (via OpenRouter) relie les mots lus dans chaque tranche en une phrase courte,
pour que l'oral soit un monologue continu plutôt qu'une suite de mots isolés.

Les mots lus par la J-lens sont gardés tels quels, dans l'ordre si possible ; le LM n'ajoute que des liaisons.
Chaque phrase voit les précédentes, pour la continuité. Résultat : phrases.json dans le dossier du run.

    python -m introspect.phrases --run runs/q1_rave_phrase [--model liquid/lfm-3-1.7b:free]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from .common import Log, run_config

# petit modèle sans raisonnement : les petits modèles gratuits (lfm-3-1.7b:free, nemotron-3.5-lightning:free) imposent
# un raisonnement qui épuise les tokens sans produire la phrase, et le niveau gratuit renvoie souvent 429
DEFAULT_PHRASE_MODEL = "google/gemma-3-4b-it"

SYSTEM = (
    "Tu écris le monologue intérieur d'une machine qui pense, à voix basse. "
    "On te donne quelques mots qui viennent de traverser sa pensée. Écris UNE seule phrase courte en français "
    "(entre 6 et {max_len} mots) qui relie ces mots. Règles : garde chaque mot donné exactement tel quel "
    "(même s'il est en anglais), dans l'ordre donné si possible ; n'ajoute que des liaisons simples "
    "(articles, prépositions, verbes courts) ; ton fragmentaire, sensoriel, sans explication ; la phrase prolonge "
    "les phrases précédentes. Réponds uniquement par la phrase, sans guillemets."
)


def words_of(w: dict, k: int) -> list[str]:
    """Mêmes mots que le mode new_words de la voix : les nouveaux de la tranche, sinon le premier."""
    ws = w["new"][:k]
    return ws or [x for x, _ in w["words"][:1]]


def clean(s: str) -> str:
    s = s.strip().splitlines()[0] if s.strip() else ""
    return s.strip().strip('"«»“”').strip()


def missing(phrase: str, words: list[str]) -> list[str]:
    low = phrase.lower()
    return [w for w in words if not re.search(r"(?<!\w)" + re.escape(w.lower()) + r"(?!\w)", low)]


def build(run_dir: Path, cfg: dict | None = None, model: str | None = None) -> list[dict]:
    from .openrouter import OpenRouterError, chat

    run_dir = Path(run_dir)
    cfg = cfg or run_config(run_dir)
    vc = cfg["voice"]
    model = model or vc.get("phrase_model") or DEFAULT_PHRASE_MODEL
    max_len = int(vc.get("phrase_max_words", 12))
    question = cfg["input"].get("phrase") or ""
    log = Log(run_dir, "phrases")
    wins = json.loads((run_dir / "windows.json").read_text(encoding="utf-8"))
    out, prev = [], []
    for k, w in enumerate(wins):
        words = words_of(w, w.get("max_words", 6))
        context = " ".join(prev[-3:]) or "(début)"
        prompt = (f"Sujet sous-jacent : {question}\nPhrases précédentes : {context}\n"
                  f"Mots à relier : {', '.join(words)}\nPhrase :")
        best, tries = None, 0
        for tries in range(1, 4):
            try:
                cand = clean(chat(prompt, model=model, system=SYSTEM.format(max_len=max_len), max_tokens=80, temperature=0.8))
            except OpenRouterError as e:
                log(f"tranche {k} : essai {tries} en échec ({e})")
                time.sleep(3 * tries)
                continue
            if not cand or len(cand.split()) > max_len + 4:
                continue
            if best is None or len(missing(cand, words)) < len(missing(best, words)):
                best = cand  # on garde la tentative qui conserve le plus de mots lus
            if not missing(best, words):
                break
        phrase = best or ", ".join(words)  # repli : les mots lus seuls
        prev.append(phrase)
        out.append({"window": k, "words": words, "phrase": phrase, "missing": missing(phrase, words), "tries": tries})
        log(f"{k:2d} {words} -> {phrase}")
    (run_dir / "phrases.json").write_text(json.dumps({"model": model, "phrases": out}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    log(f"{len(out)} phrases, {sum(1 for o in out if o['missing'])} avec des mots manquants")
    log.done()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default=None)
    a = ap.parse_args(argv)
    build(Path(a.run), model=a.model)


if __name__ == "__main__":
    main()
