"""J-space d'un run, pour l'analyse : résidus h et vecteurs J @ h, par token généré et par couche de la lentille.

Écrit par record quand readout.save_jspace.enabled est vrai, dans runs/<id>/jspace.pt (torch.load) :
    h         [T, L, d]  sortie de bloc à la position du token généré, après l'injection de la drogue
    jh        [T, L, d]  J[l] @ h : l'espace où lit la lentille (logits = unembed(jh), unembed = norme finale + lm_head)
    layers    [L]        couches enregistrées (`layers: all` = toutes celles de la lentille, `band` = bande lue)
    token_ids [T], text [T] (morceau décodé), alpha [T, n_drogues]
    model, lens {repo, file}, band, phrase, dtype ; h et jh en fp16 (fp32 si une valeur dépasse 60 000)
T × L × d × 2 octets par tenseur : ~32 Mo chacun pour 200 tokens sur Qwen3.5-4B (31 couches, d = 2560).

python -m introspect.jspace runs/<id>           -> résumé
python -m introspect.jspace runs/<id> --check   -> recharge le modèle et vérifie que unembed(jh) redonne la trace"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


class Recorder:
    def __init__(self, reader, layers):
        self.layers = list(layers)
        self.J = torch.stack([reader.lens.jacobians[l] for l in self.layers]).to(reader.device, torch.float32)  # [L,d,d]
        self.h, self.jh, self.token_ids, self.text, self.alpha = [], [], [], [], []

    def add(self, captured: dict, token_id: int, text: str, alphas: list[float]):
        H = torch.stack([captured[l] for l in self.layers]).float()  # [L,d]
        self.h.append(H.cpu())
        self.jh.append(torch.einsum("lij,lj->li", self.J, H).cpu())
        self.token_ids.append(token_id); self.text.append(text); self.alpha.append(alphas)

    def save(self, path: Path, **meta) -> dict:
        h, jh = torch.stack(self.h), torch.stack(self.jh)
        dt = torch.float16 if max(float(h.abs().max()), float(jh.abs().max())) < 6e4 else torch.float32
        data = {"h": h.to(dt), "jh": jh.to(dt), "layers": self.layers, "token_ids": self.token_ids, "text": self.text,
                "alpha": self.alpha, "dtype": str(dt), **meta}
        torch.save(data, path)
        return data


def load(run_dir) -> dict:
    return torch.load(Path(run_dir) / "jspace.pt", weights_only=True)


@torch.no_grad()
def check(run_dir: Path, data: dict, steps: int = 20):
    """Relit les mots de la trace depuis jh (couches de la bande) et compte les top-5 identiques."""
    from .common import run_config
    from .lensmodel import Reader, load_lens, load_model

    cfg = run_config(run_dir)
    model, tok = load_model(cfg)
    reader = Reader(model, tok, load_lens(cfg), cfg)
    trace = [json.loads(l) for l in open(run_dir / "trace.jsonl", encoding="utf-8")]
    idx = [data["layers"].index(l) for l in reader.band]
    same = total = 0
    for t in range(0, len(trace), max(1, len(trace) // steps)):
        words, _ = reader.read_jspace(data["jh"][t, idx].float())
        for l, w in zip(reader.band, words):
            same += [x for x, _ in w[:5]] == [x for x, _ in trace[t]["layers"][str(l)][:5]]
            total += 1
    print(f"top-5 de la trace retrouvés depuis jh : {same}/{total} (couche × pas)")
    # et J @ h recalculé depuis h
    J = torch.stack([reader.lens.jacobians[l] for l in data["layers"]]).float()
    err = (torch.einsum("lij,tlj->tli", J, data["h"].float()) - data["jh"].float()).norm() / data["jh"].float().norm()
    print(f"écart relatif entre jh et J @ h recalculé : {float(err):.2e}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    run_dir = Path(a.run)
    data = load(run_dir)
    size = (run_dir / "jspace.pt").stat().st_size / 2**20
    print(f"{run_dir / 'jspace.pt'} : {size:.0f} Mio, {data['dtype']}, modèle {data['model']}, lentille {data['lens']['file']}")
    print(f"h {tuple(data['h'].shape)}, jh {tuple(data['jh'].shape)} (tokens, couches, d), couches {data['layers'][0]}..{data['layers'][-1]}")
    print(f"norme moyenne par couche, h : {[round(float(x)) for x in data['h'].float().norm(dim=-1).mean(0)]}")
    print(f"norme moyenne par couche, jh : {[round(float(x)) for x in data['jh'].float().norm(dim=-1).mean(0)]}")
    if a.check:
        check(run_dir, data)


if __name__ == "__main__":
    main()
