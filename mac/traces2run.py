"""Traces pré-enregistrées (paquet hack1robo_activations) -> run du pipeline (trace.jsonl, text.txt, windows.json).

Refait la lecture J-lens à partir des activations, comme lensmodel.Reader pendant `record` :
bande readout.band (couches 19..44), top_k sur tout le vocabulaire puis filtrage en mots entiers
(min_len, lettres seules, mots vides), entropie de chaque lecture. Pas besoin de Gemma.

    python mac/traces2run.py ~/dev/mdma/hack1robo_activations/q1_rave --run-id q1_rave
    python mac/traces2run.py .../q3_eglise --run-id q3_eglise --target 30   # vidéo de ~30 s
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub"))
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import load_file

from introspect.common import load_config, new_run_dir
from introspect.stopwords import ALL as STOPWORDS
from introspect import windows


def word_table(tokenizer_json: Path, V: int, cfg: dict) -> dict[int, str]:
    """id -> mot, mêmes règles que Reader._build_word_mask."""
    tj = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    vocab = {i: s for s, i in tj["model"]["vocab"].items()}
    for t in tj.get("added_tokens", []):
        vocab[t["id"]] = t["content"]
    min_len = cfg["readout"]["min_len"]
    stop = STOPWORDS | set(cfg["readout"].get("stopwords_extra") or [])
    out = {}
    for i in range(V):
        p = vocab.get(i)
        if not p or not p.startswith("▁"):
            continue
        w = p[1:]
        if len(w) < min_len or not all(unicodedata.category(c).startswith("L") for c in w):
            continue
        if w.lower() in stop:
            continue
        out[i] = w.lower()
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("couple", help="dossier q<n>_<environnement> du paquet")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--target", type=float, default=None, help="video.target_duration_s (s)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    src = Path(a.couple).expanduser().resolve()
    pack = src.parent
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    cfg = load_config(a.config)
    # le run reflète la génération enregistrée, pas une nouvelle
    cfg["input"]["phrase"] = meta["question"]
    cfg["input"]["system"] = meta["system_prompt"]
    cfg["input"]["environment"] = None  # system_prompt contient déjà le contexte
    cfg["generation"] = {**cfg["generation"], **meta["generation"], "block_eos": meta.get("eos_blocked", True)}
    cfg["drug"], cfg["drugs"] = {}, []
    cfg["auto_tokens"] = False
    if a.target is not None:
        cfg["video"]["target_duration_s"] = a.target or None
    cfg["source_traces"] = {"couple": meta["couple"], "environment": meta["environment"], "path": str(src)}
    cfg["run_tag"] = meta["couple"]
    d = new_run_dir(cfg, a.run_id)

    dev = a.device
    print(f"{meta['couple']} : {meta['question']} | {meta['environment']} -> {d}")
    u = load_file(str(pack / "unembed" / "unembed.safetensors"))
    info = json.loads((pack / "unembed" / "unembed.json").read_text())
    W_U = u["W_U"].to(dev, torch.float32)  # [V, d]
    g = (1.0 + u["final_norm_weight"]).to(dev, torch.float32)
    eps = info["eps"]
    word_of = word_table(pack / "unembed" / "tokenizer.json", W_U.shape[0], cfg)

    from huggingface_hub import hf_hub_download
    ck = torch.load(hf_hub_download(meta["lens"]["repo"], meta["lens"]["file"]), map_location="cpu", weights_only=True)
    n_layers = meta["tensors"]["resid"][0]
    lo, hi = cfg["readout"]["band"]
    source = sorted(int(l) for l in ck["J"])
    band = [l for l in source if lo * n_layers <= l <= hi * n_layers]
    print(f"bande lue : {band[0]}..{band[-1]} ({len(band)} couches), top_k={cfg['readout']['top_k']}")

    resid = load_file(str(src / "acts.safetensors"))["resid"]  # [48, T, d] bf16
    P, N = meta["n_prompt_tokens"], meta["n_generated_tokens"]
    top_k = cfg["readout"]["top_k"]
    layers_words: dict[int, list] = {}
    layers_ent: dict[int, list] = {}
    for l in band:
        J = ck["J"][l].to(dev, torch.float32)
        h = resid[l, P:P + N].to(dev, torch.float32) @ J.T  # [N, d]
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * g
        probs = torch.softmax(h @ W_U.T, dim=-1)  # [N, V]
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)
        val, idx = probs.topk(top_k, dim=-1)
        val, idx = val.cpu().tolist(), idx.cpu().tolist()
        rows = []
        for vs, ids in zip(val, idx):
            agg: dict[str, float] = {}
            for v, i in zip(vs, ids):
                w = word_of.get(i)
                if w is not None:
                    agg[w] = agg.get(w, 0.0) + v
            rows.append(sorted(([w, round(s, 6)] for w, s in agg.items()), key=lambda x: -x[1]))
        layers_words[l], layers_ent[l] = rows, [round(e, 4) for e in ent.cpu().tolist()]
        del probs
        print(f"  couche {l} : {', '.join(w for w, _ in rows[N // 2][:5])}", flush=True)

    with open(d / "trace.jsonl", "w", encoding="utf-8") as f:
        for s in range(N):
            rec = {
                "step": s,
                "token_id": meta["token_ids"][P + s],
                "token": meta["tokens"][P + s],
                "layers": {str(l): layers_words[l][s] for l in band},
                "entropy": {str(l): layers_ent[l][s] for l in band},
                "drug": [],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (d / "text.txt").write_text(meta["generated_text"], encoding="utf-8")
    wins = windows.build(d, cfg)
    print(f"{N} tokens -> {len(wins)} tranches")
    for w in wins[:3]:
        print("  ", w["prompt"][:120])
    print(d)


if __name__ == "__main__":
    main()
