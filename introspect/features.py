"""Fiche de traits par mot (0..1), mise en cache dans features.json.

Sources : NRC VAD (en + fr), Lancaster (modalités), Brysbaert (concrétude), et en
repli une projection d'embeddings multilingues sur des axes définis par des paires
de mots ancres."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lexicons as L

AXES = {  # trait : (pôle haut, pôle bas)
    "valence": (["joyeux", "heureux", "happy", "joy"], ["triste", "malheureux", "sad", "grief"]),
    "arousal": (["agité", "excité", "frantic", "excited"], ["calme", "paisible", "calm", "sleepy"]),
    "dominance": (["puissant", "dominant", "powerful", "control"], ["faible", "soumis", "weak", "helpless"]),
    "concreteness": (["pierre", "table", "stone", "chair"], ["idée", "abstrait", "idea", "concept"]),
    "brightness": (["lumineux", "brillant", "bright", "light"], ["sombre", "obscur", "dark", "shadow"]),
    "warmth": (["chaud", "brûlant", "warm", "hot"], ["froid", "glacé", "cold", "ice"]),
    "roughness": (["rugueux", "rêche", "rough", "harsh"], ["doux", "lisse", "soft", "smooth"]),
}
SENSES = ["auditory", "visual", "haptic", "olfactory", "gustatory", "interoceptive"]


class Embedder:
    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer

        self.m = SentenceTransformer(name, device="cpu")
        self.axes = {}
        for k, (hi, lo) in AXES.items():
            e = self.m.encode(hi + lo, normalize_embeddings=True)
            self.axes[k] = e[: len(hi)].mean(0) - e[len(hi):].mean(0)

    def project(self, words: list[str]) -> dict[str, dict[str, float]]:
        if not words:
            return {}
        E = self.m.encode(words, normalize_embeddings=True)
        raw = {k: E @ v for k, v in self.axes.items()}
        out = {}
        for k, r in raw.items():
            # mise à l'échelle robuste sur le vocabulaire du run
            lo, hi = np.percentile(r, 5), np.percentile(r, 95)
            raw[k] = np.clip((r - lo) / (hi - lo + 1e-9), 0, 1)
        for i, w in enumerate(words):
            out[w] = {k: float(raw[k][i]) for k in raw}
        return out


def compute(words: list[str], cfg: dict) -> dict[str, dict]:
    fc = cfg["sound"]["features"]
    lex = set(fc.get("lexicons") or [])
    emb = None
    if fc.get("fallback") == "embeddings":
        try:
            emb = Embedder(fc.get("embed_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")).project(words)
        except Exception as e:  # repli : valeurs neutres
            print("embeddings indisponibles :", e)
    out = {}
    nen, nfr, lan, con = L.nrc_en(), L.nrc_fr(), L.lancaster(), L.concreteness()
    for w in words:
        f = {"source": []}
        en = w
        if w not in nen and w in nfr:
            en = nfr[w][1]
        e = (emb or {}).get(w, {})
        # VAD
        vad = None
        if "nrc_vad" in lex:
            vad = nen.get(w) or (nfr[w][0] if w in nfr else None) or nen.get(en)
        if vad:
            f.update(valence=vad[0], arousal=vad[1], dominance=vad[2]); f["source"].append("nrc")
        else:
            f.update(valence=e.get("valence", 0.5), arousal=e.get("arousal", 0.5), dominance=e.get("dominance", 0.5))
        # concrétude
        c = con.get(w, con.get(en)) if "concreteness" in lex else None
        if c is not None:
            f["concreteness"] = c; f["source"].append("brysbaert")
        else:
            f["concreteness"] = e.get("concreteness", 0.5)
        # modalités sensorielles
        s = lan.get(w, lan.get(en)) if "lancaster" in lex else None
        if s:
            f.update(s); f["source"].append("lancaster")
        else:
            f.update(auditory=0.3, visual=e.get("brightness", 0.5), haptic=e.get("roughness", 0.3),
                     olfactory=0.15, gustatory=0.15, interoceptive=e.get("warmth", 0.3))
        f["brightness"] = e.get("brightness", f["visual"])
        f["warmth"] = e.get("warmth", 0.5)
        f["roughness"] = e.get("roughness", f["haptic"])
        if not f["source"]:
            f["source"] = ["embeddings" if e else "neutre"]
        out[w] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}
    return out


def trace_features(run_dir: Path, wins: list[dict], cfg: dict | None = None) -> dict[int, dict[str, dict]]:
    """Traits issus de la trace, par tranche et par mot : couche d'origine moyenne,
    score J-lens, nouveauté, entropie moyenne, drogue."""
    from .windows import load_trace

    trace = load_trace(run_dir)
    if not trace:
        return {}
    layers = sorted(trace[0]["layers"], key=int)
    lo, hi = int(layers[0]), int(layers[-1])
    full = 0.004 * ((cfg or {}).get("drug") or {}).get("alpha_scale", 1.0)  # α du niveau 4 = drogue pleine
    # sound.drug_fx : force des effets « drogués » du son (modulation en anneau, saturation, phaser, distorsion),
    # 1 = pleine force au niveau 4 ; en dessous, le son garde la trace de la drogue sans sonner comme une panne
    fx = float(((cfg or {}).get("sound") or {}).get("drug_fx", 1.0))
    out = {}
    for w in wins:
        recs = trace[w["start"]: w["end"]]
        ent = float(np.mean([np.mean(list(r["entropy"].values())) for r in recs])) if recs else 0.0
        drug = max((sum(d["alpha"] for d in r["drug"]) for r in recs), default=0.0)
        per = {}
        for word, weight in w["words"]:
            num = den = 0.0
            for r in recs:
                for l in layers:
                    for ww, p in r["layers"][l]:
                        if ww == word:
                            num += int(l) * p; den += p
            layer = (num / den) if den else (lo + hi) / 2
            per[word] = {"layer": round((layer - lo) / max(1, hi - lo), 4), "score": weight,
                         "novelty": 1.0 if word in w["new"] else 0.0, "entropy": round(ent, 3), "drug": round(min(1.0, drug / full) * fx, 3)}
        out[w["index"]] = per
    return out


def load_or_compute(run_dir: Path, wins: list[dict], cfg: dict) -> dict[str, dict]:
    p = Path(run_dir) / "features.json"
    words = sorted({w for x in wins for w, _ in x["words"]})
    cache = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    missing = [w for w in words if w not in cache]
    if missing:
        cache.update(compute(missing if cache else words, cfg))
        p.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return cache
