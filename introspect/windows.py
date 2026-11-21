"""Tranches glissantes : agrège trace.jsonl en windows.json."""
from __future__ import annotations

import json
from pathlib import Path

from .common import Log


def load_trace(run_dir: Path) -> list[dict]:
    with open(Path(run_dir) / "trace.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def step_scores(trace: list[dict], cfg: dict) -> list[dict[str, float]]:
    ro = cfg["readout"]
    ee = ro.get("exclure_echos", {})
    lw = ro.get("layer_weights")
    n_echo = ee.get("last_n_tokens", 32)
    out = []
    for i, rec in enumerate(trace):
        layers = sorted(rec["layers"], key=int)
        weights = lw if lw else [1.0] * len(layers)
        sc: dict[str, float] = {}
        for w_l, l in zip(weights, layers):
            for w, p in rec["layers"][l]:
                sc[w] = sc.get(w, 0.0) + w_l * p
        if ee.get("enabled", True):
            # "text" : morceau décodé (tout tokenizer) ; "token" seul dans les traces v1 (SentencePiece)
            recent = "".join(t.get("text", t["token"].replace("▁", " ")) for t in trace[max(0, i - n_echo + 1): i + 1]).lower()
            recent_words = set(recent.split())
            sc = {w: s for w, s in sc.items() if w not in recent_words}
        out.append(sc)
    return out


def make_prompt(words: list[str], cfg: dict) -> str:
    ic = cfg["image"]
    rep = ic.get("repeat_top", 0) or 0
    words = words[: ic.get("prompt_words", 12) or 12]
    ws = words[:rep] + words if rep else words
    tpl = ic.get("prompt_template") or "{words}, {style}"
    return tpl.format(words=", ".join(ws), style=ic.get("style", "")).strip(", ")


def plan(cfg: dict, n_tokens: int) -> tuple[int, int, int]:
    """(taille W, pas S, mots dits max par tranche) ; si video.target_duration_s est
    défini, vise ~slice_target_s secondes par tranche."""
    import math

    wc, vc, vid = cfg["windows"], cfg["voice"], cfg["video"]
    W, S, maxw = wc["size"], wc["stride"], vc.get("max_words", 6)
    T = vid.get("target_duration_s")
    if T:
        d = vid.get("slice_target_s", 3.0)
        n = max(2, round(T / d))
        S = max(1, (n_tokens - W) // (n - 1))
        if S > W:  # tranches jointives : taille = pas
            W = S
            S = max(1, (n_tokens - W) // (n - 1))
        maxw = max(1, int((d - 0.4) // (0.75 + vc.get("gap_s", 0.25))))
    return W, S, maxw


def build(run_dir: Path, cfg: dict) -> list[dict]:
    log = Log(run_dir, "windows")
    trace = load_trace(run_dir)
    scores = step_scores(trace, cfg)
    wc = cfg["windows"]
    M, decay = wc["top_m"], wc.get("decay", 0.0)
    W, S, maxw = plan(cfg, len(scores))
    style = cfg["image"]["style"]
    rep = cfg["image"].get("repeat_top", 0)
    wins, prev = [], set()
    n = len(scores)
    starts = list(range(0, max(1, n - W + 1), S)) if n >= W else [0]
    for k, s in enumerate(starts):
        e = min(n, s + W)
        agg: dict[str, float] = {}
        for j in range(s, e):
            f = (1.0 - decay) ** (e - 1 - j) if decay else 1.0
            for w, v in scores[j].items():
                agg[w] = agg.get(w, 0.0) + f * v
        top = sorted(agg.items(), key=lambda x: -x[1])[:M]
        tot = sum(v for _, v in top) or 1.0
        words = [[w, round(v / tot, 4)] for w, v in top]
        names = [w for w, _ in words]
        new = [w for w in names if w not in prev]
        prev = set(names)
        drug = [max((t["drug"][i]["alpha"] for t in trace[s:e]), default=0.0) for i in range(len(trace[0]["drug"]))] if trace and trace[0]["drug"] else []
        prompt_words = names[:rep] + names if rep else names
        wins.append({
            "index": k, "start": s, "end": e, "words": words, "new": new, "max_words": maxw,
            "drug_alpha_max": drug,
            "prompt": make_prompt(names, cfg),
        })
    (Path(run_dir) / "windows.json").write_text(json.dumps(wins, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{len(wins)} tranches (W={W}, S={S}, M={M}, mots dits ≤ {maxw}) sur {n} pas")
    if wins:
        log("tranche 0 :", [w for w, _ in wins[0]["words"]])
    log.done()
    return wins
