"""États (vecteurs de pilotage) pour la drogue, sur le modèle de la config : extraction, balayage de dose, chargement.

Méthode du projet mdma : vecteur d'un état à la couche l = moyenne des activations (sortie du bloc l) sur les réponses
écrites dans cet état (prompt « persona », configs/states.yaml), moins la même moyenne en état sobre ; normalisé.
La drogue l'injecte comme les mots : h += α ‖h‖ u_l aux couches de la bande droguée, au token généré.

./run.sh py states extract --config configs/small.yaml [--states synesthesia dissolution afterglow]
./run.sh py states sweep   --config configs/small.yaml [--alphas 0.02 0.04 0.08 0.12 0.16]
-> vectors/<modèle>/trip_states.pt (unit [n_couches, d] par état, dose α recommandée, scores du juge)
   et trip_states.json (métadonnées lisibles), réponses et balayage dans runs/setup/."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from .common import ROOT, load_config

CATALOG = ROOT / "configs" / "states.yaml"
JUDGE_SYSTEM = "You are a careful literary critic. You rate texts on a scale from 0 to 9 and answer with a single digit."


def catalog() -> dict:
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8"))


def pack_path(cfg: dict) -> Path:
    return ROOT / "vectors" / cfg["model"].split("/")[-1].lower() / "trip_states.pt"


def load_pack(cfg: dict) -> dict:
    p = pack_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} : lancer d'abord ./run.sh py states extract --config <config>")
    return torch.load(p, weights_only=True)


def drug_layers(n_layers: int, band=(0.45, 0.70)) -> list[int]:
    return [l for l in range(n_layers) if band[0] * n_layers <= l <= band[1] * n_layers]


class Model:
    def __init__(self, cfg: dict):
        import jlens

        from .lensmodel import load_model

        self.cfg = cfg
        self.model, self.tok = load_model(cfg)
        self.tok.padding_side = "left"
        self.layers = jlens.from_hf(self.model, self.tok, force_bos=False).layers
        self.extra = cfg["input"].get("chat_template_kwargs") or {}

    def chat(self, user: str, system: str | None = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **self.extra)

    @torch.no_grad()
    def generate(self, prompts: list[str], inject: dict[int, torch.Tensor] | None = None, alpha: float = 0.0,
                 max_new_tokens: int = 128, seed: int = 0) -> list[str]:
        """Génération par lots ; avec `inject`, h += α ‖h‖ u_l aux couches données, sur les tokens générés seulement."""
        handles = []
        for l, u in (inject or {}).items():
            u = u.to(self.model.device, torch.bfloat16)

            def hook(module, inputs, output, u=u):
                h = output[0] if not torch.is_tensor(output) else output
                if h.shape[1] != 1:  # pré-remplissage du prompt : pas de drogue, comme dans record
                    return None
                h = h + alpha * h.float().norm(dim=-1, keepdim=True).to(h.dtype) * u
                return (h,) + tuple(output[1:]) if not torch.is_tensor(output) else h
            handles.append(self.layers[l].register_forward_hook(hook))
        try:
            torch.manual_seed(seed)
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8, top_p=0.95,
                                      pad_token_id=self.tok.pad_token_id)
        finally:
            for h in handles:
                h.remove()
        return [self.tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in out]

    @torch.no_grad()
    def mean_acts(self, prompts: list[str], answers: list[str]) -> torch.Tensor:
        """Sortie de chaque bloc moyennée sur les tokens de la réponse -> [N, n_couches, d] (un texte à la fois)."""
        out = []
        for p, a in zip(prompts, answers):
            n = len(self.tok(p, add_special_tokens=False).input_ids)
            ids = self.tok(p + a, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
            acts = [None] * len(self.layers)

            def mk(i):
                def hook(module, inputs, output):
                    h = output[0] if not torch.is_tensor(output) else output
                    acts[i] = h[0, n:].float().mean(0).cpu()
                return hook
            hs = [layer.register_forward_hook(mk(i)) for i, layer in enumerate(self.layers)]
            try:
                self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
            finally:
                for h in hs:
                    h.remove()
            out.append(torch.stack(acts))
        return torch.stack(out)

    @torch.no_grad()
    def judge(self, texts: list[str], criterion: str) -> list[float]:
        """Note attendue de 0 à 9, lue sur la distribution du prochain token parmi les chiffres (modèle non drogué)."""
        digits = [self.tok.convert_tokens_to_ids(str(d)) for d in range(10)]
        out = []
        for t in texts:
            prompt = self.chat(f"Text:\n<<<\n{t}\n>>>\n\nOn a scale from 0 (not at all) to 9 (extremely), how strongly "
                               f"does this text convey {criterion}? Answer with a single digit.", JUDGE_SYSTEM)
            ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
            logits = self.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1, digits].float()
            out.append(float((logits.softmax(-1) * torch.arange(10, device=logits.device)).sum()))
        return out


def persona(cat: dict, system: str) -> str:
    return system + "\n\n" + cat["style"]


def extract(cfg: dict, keys: list[str]):
    cat = catalog()
    M = Model(cfg)
    t0 = time.time()
    qs = cat["train_questions"]
    conds = {"sober": cat["sober"], **{k: cat["states"][k]["system"] for k in keys}}
    answers, acts = {}, {}
    for k, system in conds.items():
        prompts = [M.chat(q, persona(cat, system)) for q in qs]
        answers[k] = M.generate(prompts, max_new_tokens=128, seed=0)
        acts[k] = M.mean_acts(prompts, answers[k])
        print(f"{k:12s} {len(qs)} réponses et activations ({time.time() - t0:.0f} s) : {answers[k][0][:110]!r}", flush=True)
    sober = acts["sober"].mean(0)  # [L, d]
    typical = acts["sober"].norm(dim=-1).mean(0)  # ‖h‖ typique par couche
    pack = {"model": cfg["model"], "method": "moyenne des activations sur les réponses, persona moins sobre",
            "n_layers": len(M.layers), "states": {}}
    for k in keys:
        d = acts[k].mean(0) - sober
        pack["states"][k] = {"unit": (d / d.norm(dim=-1, keepdim=True)).half(), "label": cat["states"][k]["label"],
                             "criterion": cat["states"][k]["criterion"],
                             "ratio": [round(float(x), 3) for x in d.norm(dim=-1) / typical]}
    ks = list(pack["states"])
    lm = drug_layers(len(M.layers))
    U = torch.stack([pack["states"][k]["unit"][lm].float() for k in ks])  # [S, Lb, d]
    cos = torch.einsum("ald,bld->ab", U, U) / len(lm)
    print("cosinus moyen entre états sur les couches droguées :", {f"{a}/{b}": round(float(cos[i, j]), 2)
          for i, a in enumerate(ks) for j, b in enumerate(ks) if i < j})
    for k in ks:
        r = pack["states"][k]["ratio"]
        print(f"{k:12s} ‖v‖ / ‖h‖ typique, couches {lm[0]}..{lm[-1]} : {[r[l] for l in lm]}")
    save(cfg, pack)
    (ROOT / "runs" / "setup" / f"states_answers_{cfg['model'].split('/')[-1]}.json").write_text(
        json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")


def sweep(cfg: dict, alphas: list[float]):
    cat = catalog()
    pack = load_pack(cfg)
    M = Model(cfg)
    lm = drug_layers(pack["n_layers"])
    prompts = [M.chat(q) for q in cat["test_questions"]]
    rows, t0 = [], time.time()
    base = M.generate(prompts, max_new_tokens=100, seed=1)
    for k, s in pack["states"].items():
        crit = cat["states"][k]["criterion"]
        for a in [0.0] + alphas:
            outs = base if a == 0 else M.generate(prompts, {l: s["unit"][l].float() for l in lm}, a, max_new_tokens=100, seed=1)
            tr, co = M.judge(outs, crit), M.judge(outs, cat["coherence"])
            row = {"state": k, "alpha": a, "trait": sum(tr) / len(tr), "coherence": sum(co) / len(co),
                   "example": outs[0][:160]}
            row["score"] = row["trait"] * row["coherence"] / 81
            rows.append(row)
            print(f"{k:12s} α {a:5.3f} : trait {row['trait']:.2f}  cohérence {row['coherence']:.2f}  score {row['score']:.2f}"
                  f"  ({time.time() - t0:.0f} s) {row['example'][:90]!r}", flush=True)
        best = max((r for r in rows if r["state"] == k), key=lambda r: r["score"])
        s.update(alpha=best["alpha"], scores={x: round(best[x], 2) for x in ("trait", "coherence", "score")})
    pack["sweep"] = rows
    save(cfg, pack)


def save(cfg: dict, pack: dict):
    p = pack_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, p)
    meta = {k: v for k, v in pack.items() if k not in ("states", "sweep")}
    meta["states"] = {k: {kk: vv for kk, vv in s.items() if kk != "unit"} for k, s in pack["states"].items()}
    meta["sweep"] = pack.get("sweep", [])
    p.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(p, f"({p.stat().st_size / 1e6:.1f} Mo)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "sweep"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--states", nargs="+", default=["synesthesia", "dissolution", "afterglow"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.02, 0.04, 0.08, 0.12, 0.16])
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd == "extract":
        extract(cfg, a.states)
    else:
        sweep(cfg, a.alphas)


if __name__ == "__main__":
    main()
