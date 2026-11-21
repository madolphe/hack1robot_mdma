"""Étape 2 : valide la lecture manuelle (formule Neuronpedia, fp32) contre lens.apply."""
from __future__ import annotations

import json
import sys

import torch

from .common import ROOT, load_config
from .lensmodel import Reader, load_lens, load_model


@torch.no_grad()
def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else ROOT / "configs" / "default.yaml")
    model, tok = load_model(cfg)
    lens = load_lens(cfg)
    print(lens)
    reader = Reader(model, tok, lens, cfg)
    lm = reader.lm
    prompt = "Fact: The currency used in the country shaped like a boot is"
    lens_logits, model_logits, input_ids = lens.apply(lm, prompt, positions=[-1])

    # lecture manuelle, résidus capturés par nos propres hooks
    from jlens.hooks import ActivationRecorder
    final = lm.n_layers - 1
    with ActivationRecorder(lm.layers, at=sorted(set(lens.source_layers) | {final})) as rec:
        lm.forward(input_ids)
        acts = {i: rec.activations[i][0, -1].float() for i in rec.activations}

    report = {"layers": {}, "n_layers": lm.n_layers, "source_layers": lens.source_layers}
    agree = 0
    for l in lens.source_layers:
        J = lens.jacobians[l].to("cuda", torch.float32)
        manual = reader.unembed((acts[l] @ J.T)[None])[0]
        a = manual.topk(5).indices.tolist()
        b = lens_logits[l][0].topk(5).indices.tolist()
        ok = a == b
        agree += ok
        report["layers"][l] = {"manual": tok.convert_ids_to_tokens(a), "jlens": tok.convert_ids_to_tokens(b), "same": ok,
                               "overlap": len(set(a) & set(b))}
    final_manual = reader.unembed(acts[final][None])[0]
    real = model_logits[0].to(final_manual.device)
    report["final_layer"] = {
        "manual_top5": tok.convert_ids_to_tokens(final_manual.topk(5).indices.tolist()),
        "model_top5": tok.convert_ids_to_tokens(real.topk(5).indices.tolist()),
        "max_abs_diff": float((final_manual - real.float()).abs().max()),
    }
    # vrais logits du modèle complet (forward HF standard)
    full = model(input_ids=input_ids).logits[0, -1].float()
    report["final_layer"]["hf_forward_top5"] = tok.convert_ids_to_tokens(full.topk(5).indices.tolist())
    report["summary"] = f"top-5 identiques sur {agree}/{len(lens.source_layers)} couches"
    out = ROOT / "runs" / "setup" / f"validate_lens_{cfg['model'].split('/')[-1]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    for l in lens.source_layers[:: max(1, len(lens.source_layers) // 8)]:
        r = report["layers"][l]
        print(l, r["manual"], "same" if r["same"] else f"overlap {r['overlap']}/5 {r['jlens']}")
    print("final:", report["final_layer"])
    print(report["summary"])


if __name__ == "__main__":
    main()
