"""Drogues v2 : catalogue de phases, intensité lissée, tirages de mots reproductibles.

python -m introspect.drugs --check [config]   -> runs/setup/drug_tokens_<modèle>.json (contrôle de tokenisation)"""
from __future__ import annotations

import functools
import json
import random

import torch
import yaml

from .common import ROOT

CATALOG = ROOT / "configs" / "drugs.yaml"


@functools.lru_cache(None)
def catalog() -> dict:
    data = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    for d in data["drugs"].values():
        for p in d["phases"]:
            seen, words = set(), []
            for w in p.get("words") or []:
                if w.lower() not in seen:
                    seen.add(w.lower()); words.append(w)
            p["words"] = words
    return data


def names() -> list[str]:
    return list(catalog()["drugs"])


class Schedule:
    """α(t), phase et mots tirés à chaque pas pour une drogue du catalogue."""

    def __init__(self, name: str, intensity: float, n_tokens: int, seed: int, tok=None, log=print,
                 scale: float = 1.0):
        cat = catalog()
        if name not in cat["drugs"]:
            raise ValueError(f"drogue inconnue : {name}")
        self.name = name
        self.label = cat["drugs"][name]["label"]
        self.phases = cat["drugs"][name]["phases"]
        # `scale` adapte les niveaux, calibrés sur Gemma 3 12B, à un autre modèle (Qwen3.5-4B : ×15)
        self.levels = {int(k): float(v) * scale for k, v in cat["levels"].items()}
        self.cap = float(cat.get("alpha_cap", 0.006)) * scale
        self.every = int(cat.get("draw_every", 12))
        self.k = int(cat.get("draw_k", 3))
        self.trans = float(cat.get("transition", 0.05))
        self.band = cat.get("band", [0.45, 0.70])
        self.intensity = float(intensity)
        self.T = max(1, n_tokens)
        self.rng = random.Random(seed)
        self.words: list[str] = []
        self.prev: set[str] = set()
        self.log = log
        if tok is not None:  # écarte les mots qui ne sont pas un seul token, sauf si la phase n'a plus rien
            for p in self.phases:
                single = [w for w in p["words"] if is_single(tok, w)]
                p["words_used"] = single if len(single) >= self.k else p["words"]
        else:
            for p in self.phases:
                p["words_used"] = p["words"]

    def phase_at(self, x: float) -> int:
        for i, p in enumerate(self.phases):
            if p["start"] <= x < p["end"]:
                return i
        return len(self.phases) - 1

    def base_alpha(self, i: int) -> float:
        return self.levels[int(self.phases[i]["level"])]

    def alpha(self, step: int) -> float:
        x = step / self.T
        i = self.phase_at(x)
        a = self.base_alpha(i)
        h = self.trans / 2
        # lissage linéaire autour des frontières de phase, sur `transition` de la prise
        if i > 0 and x < self.phases[i]["start"] + h:
            b = self.phases[i]["start"]
            f = (x - (b - h)) / self.trans
            a = (1 - f) * self.base_alpha(i - 1) + f * a
        elif i + 1 < len(self.phases) and x > self.phases[i]["end"] - h:
            b = self.phases[i]["end"]
            f = (x - (b - h)) / self.trans
            a = (1 - f) * a + f * self.base_alpha(i + 1)
        return min(self.cap, a * self.intensity)

    def draw(self, step: int) -> list[str] | None:
        """Nouveau tirage tous les `every` pas ; renvoie la liste si elle change."""
        if step % self.every != 0 and self.words:
            return None
        pool = self.phases[self.phase_at(step / self.T)]["words_used"]
        if not pool:
            self.words = []
            return self.words
        cand = [w for w in pool if w not in self.prev] or list(pool)
        self.words = self.rng.sample(cand, min(self.k, len(cand)))
        self.prev = set(self.words)
        return self.words

    def phase_name(self, step: int) -> str:
        return self.phases[self.phase_at(step / self.T)]["name"]


def is_single(tok, word: str) -> bool:
    from .lensmodel import single_token_id

    return single_token_id(tok, word) is not None


class CatalogDrug:
    """Adaptateur pour record.Hooks : expose .alpha, .layers, .vec (par couche) et l'état du pas."""

    def __init__(self, sched: Schedule, reader, log=print):
        self.s = sched
        self.reader = reader
        n = reader.n_layers
        lo, hi = sched.band
        self.layers = {l for l in reader.lens.source_layers if lo * n <= l <= hi * n}
        self.vec: dict[int, torch.Tensor] = {}
        self.alpha = 0.0
        self.cache: dict[str, dict[int, torch.Tensor]] = {}
        self.spec = {"word": sched.label, "on_prompt": False}
        self.log = log

    def word_vecs(self, w: str) -> dict[int, torch.Tensor]:
        if w not in self.cache:
            self.cache[w] = {l: self.reader.direction(w, l, self.log) for l in self.layers}
        return self.cache[w]

    def update(self, step: int):
        if step < 0:
            self.alpha = 0.0
            return
        self.alpha = self.s.alpha(step)
        new = self.s.draw(step)
        if new is not None:
            if new and self.alpha > 0:
                vs = [self.word_vecs(w) for w in new]
                self.vec = {}
                for l in self.layers:
                    v = sum(x[l] for x in vs)
                    self.vec[l] = v / v.norm()
            elif not new:
                self.vec = {}
        if not self.vec:
            self.alpha = 0.0

    def state(self, step: int) -> dict:
        return {"word": self.s.label, "alpha": round(self.alpha, 5), "phase": self.s.phase_name(max(step, 0)),
                "words": list(self.s.words)}


def check(config: str | None = None):
    from transformers import AutoTokenizer

    from .common import load_config
    from .lensmodel import model_source
    cfg = load_config(config or ROOT / "configs" / "default.yaml")
    model = cfg["model"]
    tok = AutoTokenizer.from_pretrained(model_source(cfg))
    report = {}
    tot = {"kept": 0, "multi": 0}
    for name, d in catalog()["drugs"].items():
        for p in d["phases"]:
            kept = [w for w in p["words"] if is_single(tok, w)]
            multi = {w: tok.convert_ids_to_tokens(tok.encode(" " + w, add_special_tokens=False)) for w in p["words"] if w not in kept}
            report[f"{name}/{p['name']}"] = {"n": len(p["words"]), "kept": kept, "multi_token_dropped": multi}
            tot["kept"] += len(kept); tot["multi"] += len(multi)
    out = ROOT / "runs" / "setup" / f"drug_tokens_{model.split('/')[-1]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"total": tot, "phases": report}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(tot)
    for k, v in report.items():
        print(f"{k:40s} {len(v['kept']):2d}/{v['n']:2d}  écartés: {list(v['multi_token_dropped'])[:8]}")


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:  # python -m introspect.drugs --check [config.yaml]
        rest = [a for a in sys.argv[1:] if a != "--check"]
        check(rest[0] if rest else None)
