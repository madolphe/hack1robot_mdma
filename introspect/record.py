"""Passe 1 : génération token par token, capture du résidu, lecture J-lens, drogue.

Écrit trace.jsonl, text.txt puis windows.json (via windows.build)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .common import Log, load_config, new_run_dir
from .lensmodel import Reader, drug_alpha, load_lens, load_model, stop_token_ids
from . import windows as winmod


def build_prompt(tok, cfg: dict) -> torch.Tensor:
    phrase = cfg["input"].get("phrase") or cfg["input"]["phrases"][cfg["input"].get("phrase_index", 0)]
    msgs = []
    if cfg["input"].get("system"):
        msgs.append({"role": "system", "content": cfg["input"]["system"]})
    msgs.append({"role": "user", "content": phrase})
    # options du gabarit, par ex. {enable_thinking: false} pour que Qwen3.5 disserte sans bloc <think>
    extra = cfg["input"].get("chat_template_kwargs") or {}
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, **extra)
    return ids["input_ids"], phrase


def sample(logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sp, si = probs.sort(descending=True)
    keep = sp.cumsum(-1) - sp < top_p
    sp = sp * keep
    sp = sp / sp.sum()
    choice = torch.multinomial(sp, 1, generator=gen)
    return int(si[choice])


class FixedDrug:
    """Drogue v1 : un mot, un profil temporel (constant, rampe, pulsation, sinusoïde)."""

    def __init__(self, spec: dict, layers: set, vec: dict):
        self.spec, self.layers, self.vec, self.alpha = spec, layers, vec, 0.0

    def update(self, step: int):
        if step < 0:
            self.alpha = drug_alpha(self.spec, 0) if self.spec.get("on_prompt") else 0.0
        else:
            self.alpha = drug_alpha(self.spec, step)

    def state(self, step: int) -> dict:
        return {"word": self.spec["word"], "alpha": round(self.alpha, 5)}


class Hooks:
    """Un hook par bloc : applique les drogues puis capture la sortie de bloc.

    Le flux résiduel additionne les sorties des blocs : un vecteur injecté à la couche k est encore présent tel quel
    aux couches suivantes, et la lentille le relit (sa direction est faite pour cela). `injected[l]` garde la somme
    de ce qui a été ajouté jusqu'à la couche l pour le token courant ; avec `subtract`, `read(l)` rend h − injected[l] :
    la lecture montre ce que les couches ont calculé en réponse à la drogue, pas la poussée elle-même. Le modèle,
    lui, reçoit toujours h avec la drogue."""

    def __init__(self, blocks, capture_layers, drugs, reader: Reader, subtract: bool = True):
        self.capture_layers = set(capture_layers)
        self.drugs = drugs  # objets avec .layers, .vec {couche: v}, .alpha, update(step), state(step)
        self.subtract = subtract
        self.step = -1  # -1 = prompt
        self.captured: dict[int, torch.Tensor] = {}
        self.injected: dict[int, torch.Tensor | None] = {}
        self.carried = None  # injection cumulée à la position du token courant (fp32)
        self.handles = []
        layers = self.capture_layers | {l for d in drugs for l in d.layers}
        for l in sorted(layers):
            self.handles.append(blocks[l].register_forward_hook(self._make(l)))

    @property
    def alphas(self):
        return [d.alpha for d in self.drugs]

    def set_step(self, step: int):
        self.step = step
        self.captured, self.injected, self.carried = {}, {}, None
        for d in self.drugs:
            d.update(step)

    def read(self, l: int) -> torch.Tensor:
        """Résidu lu par la lentille à la couche l : sans l'injection cumulée si `subtract`."""
        inj = self.injected.get(l)
        h = self.captured[l].float()
        return h - inj if self.subtract and inj is not None else h

    def _make(self, l):
        def hook(module, inputs, output):
            is_tuple = not torch.is_tensor(output)
            h = output[0] if is_tuple else output
            changed = False
            for d in self.drugs:
                a = d.alpha
                if a != 0.0 and l in d.layers and l in d.vec:
                    v = d.vec[l].to(h.dtype)
                    if self.step < 0:  # prompt : toutes les positions
                        n = h.float().norm(dim=-1, keepdim=True).to(h.dtype)
                        h = h + a * n * v
                    else:
                        n = h[:, -1:].float().norm(dim=-1, keepdim=True).to(h.dtype)
                        new = h[:, -1:] + a * n * v
                        delta = (new[0, -1].float() - h[0, -1].float())  # ce qui a vraiment été ajouté (arrondi bf16 compris)
                        self.carried = delta if self.carried is None else self.carried + delta
                        h = torch.cat([h[:, :-1], new], dim=1)
                    changed = True
            if l in self.capture_layers and self.step >= 0:
                self.captured[l] = h[0, -1].detach()
                self.injected[l] = self.carried
            if changed:
                return (h,) + tuple(output[1:]) if is_tuple else h
            return None
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def prepare_drugs(cfg, reader: Reader, log, tok=None):
    out = []
    n = reader.n_layers
    for spec in cfg.get("drugs") or []:
        lo, hi = spec.get("band", cfg["readout"]["band"])
        layers = [l for l in reader.lens.source_layers if lo * n <= l <= hi * n]
        vec = {l: reader.direction(spec["word"], l, log) for l in layers}
        log(f"drogue '{spec['word']}' α={spec.get('alpha')} profil={spec.get('profile','constant')} couches {layers[0]}..{layers[-1]} ({len(layers)})")
        out.append(FixedDrug(spec, set(layers), vec))
    dc = cfg.get("drug") or {}
    if dc.get("name"):
        from .drugs import CatalogDrug, Schedule, catalog

        states = None
        if any("state" in p for p in catalog()["drugs"].get(dc["name"], {}).get("phases", [])):
            from .states import load_pack
            states = load_pack(cfg)["states"]  # vecteurs d'état du modèle (introspect/states.py)
        sched = Schedule(dc["name"], dc.get("intensity", 1.0), cfg["generation"]["max_new_tokens"],
                         cfg["generation"]["seed"], tok=tok, log=log, scale=dc.get("alpha_scale", 1.0), states=states)
        cd = CatalogDrug(sched, reader, log)
        log(f"drogue catalogue '{sched.label}' ×{sched.intensity} sur {sched.T} tokens, couches {min(cd.layers)}..{max(cd.layers)} ; "
            + " | ".join(f"{p['name']} {p['start']:.2f}-{p['end']:.2f} "
                         + (f"état {p['state']} α {sched.base_alpha(i):.3f}" if "state" in p else f"niv.{p['level']}")
                         for i, p in enumerate(sched.phases)))
        out.append(cd)
    return out


@torch.no_grad()
def record(cfg: dict, run_dir: Path, model=None, tok=None, lens=None) -> Path:
    log = Log(run_dir, "record")
    seed = cfg["generation"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if model is None:
        log("chargement de", cfg["model"])
        model, tok = load_model(cfg)
    if lens is None:
        lens = load_lens(cfg)
    reader = Reader(model, tok, lens, cfg)
    log(f"{reader.n_layers} couches, bande lue : {reader.band[0]}..{reader.band[-1]} ({len(reader.band)} couches)")
    drugs = prepare_drugs(cfg, reader, log, tok)
    js = cfg["readout"].get("save_jspace") or {}
    jrec = None
    if js.get("enabled"):  # J-space pour l'analyse (introspect/jspace.py) : capture seule, la génération ne change pas
        from .jspace import Recorder

        jrec = Recorder(reader, lens.source_layers if js.get("layers", "all") == "all" else reader.band)
        log(f"J-space enregistré sur les couches {jrec.layers[0]}..{jrec.layers[-1]} ({len(jrec.layers)})")
    subtract = cfg["readout"].get("subtract_injection", True)
    hooks = Hooks(reader.lm.layers, set(reader.band) | set(jrec.layers if jrec else []), drugs, reader, subtract)
    if drugs:
        log("lecture " + ("sans l'injection cumulée (readout.subtract_injection)" if subtract else "brute : la lentille relit aussi l'injection"))

    input_ids, phrase = build_prompt(tok, cfg)
    input_ids = input_ids.to(model.device)
    log("phrase :", phrase, f"| prompt {input_ids.shape[1]} tokens")
    gen = torch.Generator(device=model.device).manual_seed(seed)
    g = cfg["generation"]
    stop_ids = stop_token_ids(model, tok)
    block_eos = g.get("block_eos", False)  # v2 : la prise dure toute la vidéo

    trace_f = open(run_dir / "trace.jsonl", "w", encoding="utf-8")
    out_ids: list[int] = []
    try:
        hooks.set_step(-1)
        # le modèle crée son propre cache (Qwen3.5 mêle attention et couches linéaires à état)
        out = model(input_ids=input_ids, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[0, -1]
        for step in range(g["max_new_tokens"]):
            if block_eos:
                logits = logits.clone()
                logits[list(stop_ids)] = float("-inf")
            rp = float(g.get("repetition_penalty", 1.0))
            if rp != 1.0 and out_ids:  # pénalité de répétition (convention HF) sur les tokens déjà écrits
                seen = torch.tensor(sorted(set(out_ids[-int(g.get("repetition_window", 0) or len(out_ids)):])),
                                    device=logits.device)
                logits = logits.clone()
                v = logits[seen]
                logits[seen] = torch.where(v > 0, v / rp, v * rp)
            nxt = sample(logits, g["temperature"], g["top_p"], gen)
            if nxt in stop_ids:
                log(f"fin de génération (token d'arrêt) au pas {step}")
                break
            out_ids.append(nxt)
            hooks.set_step(step)
            out = model(input_ids=torch.tensor([[nxt]], device=model.device), past_key_values=cache, use_cache=True)
            logits = out.logits[0, -1]
            H = torch.stack([hooks.read(l) for l in reader.band])
            words, ent = reader.read(H)
            rec = {
                "step": step,
                "token_id": nxt,
                "text": tok.decode([nxt]),
                "token": tok.convert_ids_to_tokens(nxt),
                "layers": {str(l): w for l, w in zip(reader.band, words)},
                "entropy": {str(l): round(e, 4) for l, e in zip(reader.band, ent)},
                "drug": [d.state(step) for d in drugs],
            }
            trace_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if jrec:
                jrec.add(hooks, nxt, rec["text"], hooks.alphas)
            if step % 50 == 0:
                top = [w for w, _ in words[len(words) // 2][:5]]
                log(f"pas {step} : {top}")
    finally:
        trace_f.close()
        hooks.remove()
    text = tok.decode(out_ids, skip_special_tokens=True)
    (run_dir / "text.txt").write_text(text, encoding="utf-8")
    log(f"{len(out_ids)} tokens générés")
    if jrec and out_ids:
        d = jrec.save(run_dir / "jspace.pt", model=cfg["model"], lens=dict(cfg["lens"]), band=reader.band, phrase=phrase,
                      subtract_injection=bool(subtract and drugs))
        log(f"jspace.pt : h et jh {tuple(d['h'].shape)} en {d['dtype']}")
    log.done()
    winmod.build(run_dir, cfg)
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    d = new_run_dir(cfg, a.run_id)
    record(cfg, d)
    print(d)


if __name__ == "__main__":
    main()
