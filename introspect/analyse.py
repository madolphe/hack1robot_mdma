"""Analyse d'un run à partir de son J-space (jspace.pt) : contrôles automatiques et page HTML autonome.

./run.sh py analyse runs/<id> [--baseline runs/<témoin>]   -> runs/<id>/analyse.html (et analyse.json)

Le témoin est le même run sans drogue (même question, même graine, voir configs/variants/small_sober.yaml) :
sur le préfixe commun, il permet de vérifier l'injection elle-même."""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
from collections import Counter
from pathlib import Path

import torch

from .common import ROOT, run_config
from .jspace import load as load_jspace


def r(x, n=4):
    """Arrondi à n chiffres significatifs pour le JSON (NaN -> None)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return float(f"{x:.{n}g}")


class Decoder:
    """Décode jh avec la norme finale et lm_head du modèle, et calcule les directions de drogue."""

    def __init__(self, reader, tok, units: dict | None = None):
        self.reader, self.tok = reader, tok
        self.cache: dict[int, str] = {}
        self.dirs: dict[tuple[str, int], torch.Tensor] = {}
        self.units = units or {}  # vecteurs d'état {état: [n_couches, d]} pour les drogues par états

    def step_dir(self, words: list[str], states: dict, l: int):
        """(direction unitaire injectée à la couche l, norme du mélange) ; None sans drogue à ce pas."""
        if states:
            v = sum(w * self.units[k][l].float() for k, w in states.items())
            return v / v.norm(), float(v.norm())
        v = self.drug_vec(words, l)
        return (v, 1.0) if v is not None else None

    @torch.no_grad()
    def signature(self, state: str, l: int, n: int = 12) -> list[str]:
        """Mots que la lentille lit dans le vecteur d'état lui-même (à la couche l)."""
        J = self.reader.lens.jacobians[l].to(self.reader.device, torch.float32)
        p = self.reader.unembed((J @ self.units[state][l].float().to(self.reader.device))[None])[0]
        p = p.masked_fill(~self.reader.word_mask, float("-inf"))
        out = []
        for i in p.topk(5 * n).indices.tolist():
            w = self.reader.word_of[i]
            if w not in out:
                out.append(w)
            if len(out) == n:
                break
        return out

    def piece(self, i: int) -> str:
        if i not in self.cache:
            self.cache[i] = self.tok.decode([i])
        return self.cache[i]

    def word_ids(self, w: str) -> list[int]:
        return self.reader.word_token_ids(w, log=lambda *a: None)

    @torch.no_grad()
    def directions(self, words: set[str], layers: list[int]):
        """Comme Reader.direction, pour tous les mots et toutes les couches d'un coup."""
        g = self.reader.gain()
        words = sorted(w for w in words if (w, layers[0]) not in self.dirs)
        if not words:
            return
        rows = [(w, self.word_ids(w)) for w in words]
        for l in layers:
            J = self.reader.lens.jacobians[l].to(self.reader.device, torch.float32)
            for w, ids in rows:
                vs = (g * self.reader.W_U[ids]) @ J  # [n_tok, d] = (J.T @ (g ⊙ W_U[t]))ᵀ
                v = (vs / vs.norm(dim=-1, keepdim=True)).mean(0)
                self.dirs[(w, l)] = (v / v.norm()).cpu()

    def drug_vec(self, words: list[str], l: int):
        """Direction injectée par CatalogDrug : somme normalisée des directions des mots tirés."""
        if not words:
            return None
        v = sum(self.dirs[(w, l)] for w in words)
        return v / v.norm()

    @torch.no_grad()
    def run(self, d: dict, drug_words: list[list[str]], k=5) -> dict:
        """Métriques [T, L] d'un run : normes, entropie, top-k, rang du token suivant, part des mots-drogue."""
        jh, h, ids = d["jh"], d["h"], d["token_ids"]
        T, L, _ = jh.shape
        dev = self.reader.device
        ent, p1, rank, pnext, mass, conv = (torch.full((T, L), float("nan")) for _ in range(6))
        top_ids = torch.zeros((T, L, k), dtype=torch.long)
        top_p = torch.zeros((T, L, k))
        word_ids = [sorted({i for w in ws for i in self.word_ids(w)}) for ws in drug_words]
        last = jh[:, -1].float()
        for li in range(L):
            logits = self.reader.unembed(jh[:, li].to(dev).float())  # [T, V]
            logp = logits.log_softmax(-1)
            p = logp.exp()
            ent[:, li] = (-(p * logp).sum(-1)).cpu()
            v, i = p.topk(k, dim=-1)
            top_ids[:, li], top_p[:, li] = i.cpu(), v.cpu()
            p1[:, li] = v[:, 0].cpu()
            nxt = torch.tensor(ids[1:] + [ids[-1]], device=dev)
            tgt = logits.gather(1, nxt[:, None])
            rank[:, li] = (logits > tgt).sum(-1).float().cpu()
            pnext[:, li] = p.gather(1, nxt[:, None])[:, 0].cpu()
            for t, wi in enumerate(word_ids):
                if wi:
                    mass[t, li] = float(p[t, wi].sum())
            conv[:, li] = torch.nn.functional.cosine_similarity(jh[:, li].float(), last, dim=-1)
        rank[-1], pnext[-1] = float("nan"), float("nan")  # pas de token suivant après le dernier
        return {"ent": ent, "p1": p1, "rank": rank, "pnext": pnext, "mass": mass, "conv": conv,
                "hn": h.float().norm(dim=-1), "jn": jh.float().norm(dim=-1), "top_ids": top_ids, "top_p": top_p}


def load_run(run_dir: Path) -> dict:
    trace = [json.loads(l) for l in open(run_dir / "trace.jsonl", encoding="utf-8")]
    return {"dir": run_dir, "cfg": run_config(run_dir), "js": load_jspace(run_dir), "trace": trace,
            "windows": json.loads((run_dir / "windows.json").read_text(encoding="utf-8"))}


def drug_info(trace) -> tuple[list[list[str]], list[float], list[str]]:
    words, alpha, phase = [], [], []
    for rec in trace:
        s = rec["drug"][0] if rec["drug"] else {}
        words.append(list(s.get("words") or ([s["word"]] if s.get("word") and "words" not in s else [])))
        alpha.append(float(s.get("alpha", 0.0)))
        phase.append(s.get("phase", ""))
    return words, alpha, phase


def drug_states(trace) -> list[dict]:
    """Mélange d'états injecté à chaque pas ({} pour une drogue par mots ou sans drogue)."""
    return [(rec["drug"][0].get("states") or {}) if rec["drug"] else {} for rec in trace]


def thumbs(run_dir: Path, width=320) -> dict[int, str]:
    from PIL import Image

    out = {}
    for f in sorted((run_dir / "frames").glob("[0-9][0-9][0-9][0-9].png")):
        im = Image.open(f).convert("RGB")
        im.thumbnail((width, width))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=72)
        out[int(f.stem)] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return out


def checks(main, base, m, b, dec, layers, band, drug_layers, info) -> list[dict]:
    """Contrôles automatiques ; status : good | warning | critical | info."""
    out = []
    js = main["js"]
    for name in ("h", "jh"):
        bad = int((~torch.isfinite(js[name].float())).sum())
        out.append({"status": "critical" if bad else "good", "title": f"{name} : valeurs finies",
                    "detail": f"{bad} valeurs NaN ou infinies" if bad else f"aucune NaN ni infinie ({js['dtype']})"})
    # cohérence trace / jspace : la lecture filtrée refaite depuis jh redonne-t-elle la trace ?
    band_idx = [layers.index(l) for l in band]
    same = total = 0
    for t, rec in enumerate(main["trace"]):
        ws, _ = dec.reader.read_jspace(js["jh"][t, band_idx].float())
        for l, w in zip(band, ws):
            same += [x for x, _ in w[:5]] == [x for x, _ in rec["layers"][str(l)][:5]]
            total += 1
    out.append({"status": "good" if same / max(total, 1) > 0.97 else "warning", "title": "Trace et J-space concordent",
                "detail": f"la lecture refaite depuis jh redonne les 5 premiers mots de la trace pour {same}/{total} "
                          "(pas × couche ; écarts dus au fp16)"})
    # dernière couche : le token généré suivant
    med = torch.nanmedian(m["rank"][:-1, -1]).item()
    top1 = float((m["rank"][:-1, -1] == 0).float().mean())
    out.append({"status": "good" if top1 > 0.4 else "warning", "title": "La dernière couche prédit le texte",
                "detail": f"couche {layers[-1]} : le token réellement écrit ensuite est 1er dans {top1:.0%} des pas "
                          f"(rang médian {med:.0f} ; échantillonnage à température 0,8)"})
    # couches dont la lecture ne bouge pas
    stuck = []
    for li, l in enumerate(layers):
        tok, n = Counter(m["top_ids"][:, li, 0].tolist()).most_common(1)[0]
        if n / len(m["top_ids"]) > 0.5:
            stuck.append(f"{l} ({dec.piece(tok)!r} {n / len(m['top_ids']):.0%})")
    out.append({"status": "info" if stuck else "good", "title": "Couches à lecture figée",
                "detail": ("top-1 identique dans plus de la moitié des pas : " + ", ".join(stuck)) if stuck else "aucune"})
    # couches où ‖J·h‖ sort du lot
    medj = m["jn"].median(0).values
    ratio = medj / medj.median()
    odd = [f"{l} (×{float(x):.1f})" for l, x in zip(layers, ratio) if x > 2.5 or x < 0.4]
    out.append({"status": "info" if odd else "good", "title": "Gain de la lentille par couche",
                "detail": ("‖J·h‖ médian loin de la médiane des couches : " + ", ".join(odd) + ". Sans effet sur la "
                           "lecture : la norme finale efface l'échelle avant le décodage") if odd
                else "‖J·h‖ médian homogène d'une couche à l'autre"})
    # tokens aberrants (norme de h, rapport à la médiane de la couche)
    ratio_h = m["hn"] / m["hn"].median(0).values
    hits = ((ratio_h > 3) | (ratio_h < 1 / 3)).nonzero().tolist()
    toks = Counter(t for t, _ in hits)
    desc = ", ".join(f"pas {t} {main['js']['text'][t]!r} ({n} couches)" for t, n in toks.most_common(6))
    out.append({"status": "warning" if hits else "good", "title": "Tokens à norme aberrante",
                "detail": f"‖h‖ à plus de ×3 de la médiane de sa couche : {desc}" if hits
                else f"aucun : ‖h‖ reste entre ×{float(ratio_h.min()):.2f} et ×{float(ratio_h.max()):.2f} de la médiane de sa couche"})
    out.append(echo_filter(main))
    if base is None:
        return out + self_reading(main, m, None, layers, band, drug_layers, info)
    # préfixe commun avec le témoin sobre
    a, bb = main["js"]["token_ids"], base["js"]["token_ids"]
    n = 0
    while n < min(len(a), len(bb)) and a[n] == bb[n]:
        n += 1
    out.append({"status": "info", "title": "Préfixe commun avec le témoin", "detail": f"{n} tokens identiques, puis les textes divergent"})
    if n:
        first = min(drug_layers)
        hm, hb = main["js"]["h"][:n].float(), base["js"]["h"][:n].float()
        below = [li for li, l in enumerate(layers) if l < first]
        diff = float((hm[:, below] - hb[:, below]).abs().max()) if below else 0.0
        out.append({"status": "good" if diff == 0 else "critical", "title": "Sous la drogue, rien ne bouge",
                    "detail": f"couches 0..{first - 1} sur le préfixe commun : écart max {diff:.3g} entre drogué et sobre "
                              "(attendu : exactement 0)"})
        # injection à la première couche droguée : Δh = α ‖h‖ v
        li = layers.index(first)
        cos, ratio = [], []
        for t in range(n):
            sd = dec.step_dir(info["words"][t], info["states"][t], first)
            if sd is None or info["alpha"][t] == 0:
                continue
            v, scale = sd
            dlt = hm[t, li] - hb[t, li]
            cos.append(float(torch.nn.functional.cosine_similarity(dlt, v, dim=0)))
            ratio.append(float(dlt.norm() / (info["alpha"][t] * scale * hb[t, li].norm())))
        if cos:
            ok = min(cos) > 0.99 and all(abs(x - 1) < 0.05 for x in ratio)
            out.append({"status": "good" if ok else "warning", "title": f"Injection mesurée à la couche {first}",
                        "detail": f"Δh = h drogué − h sobre : cos(Δh, direction) min {min(cos):.4f}, "
                                  f"‖Δh‖ / (α‖h‖) entre {min(ratio):.3f} et {max(ratio):.3f} (attendu : 1 et 1)"})
    out += self_reading(main, m, b, layers, band, drug_layers, info)
    return out


def echo_filter(main) -> dict:
    """Part du score des mots que le filtre des échos retire (mots déjà écrits dans les derniers tokens)."""
    import copy

    from .windows import step_scores

    cfg = main["cfg"]
    off = copy.deepcopy(cfg)
    off["readout"]["exclure_echos"] = {"enabled": False}
    on_s, off_s = step_scores(main["trace"], cfg), step_scores(main["trace"], off)
    kept = sum(sum(s.values()) for s in on_s)
    full = sum(sum(s.values()) for s in off_s)
    removed = 1 - kept / max(full, 1e-9)
    return {"status": "info" if removed < 0.05 else "good", "title": "Filtre des échos",
            "detail": f"il retire {removed:.1%} du score des mots lus" + (
                ". Peu : les mots intérieurs sont surtout en anglais et le texte en français, la comparaison "
                "exacte ne reconnaît pas les traductions (« simple, seemingly, door » pour « apparemment simple, "
                "ouvre une porte »)" if removed < 0.05 else "")}


def self_reading(main, m, b, layers, band, drug_layers, info) -> list[dict]:
    """La lentille relit-elle l'injection ? Part des mots marqués (mots injectés, ou signature lexicale de l'état
    injecté) dans la lecture, puis dans les mots de la vidéo."""
    out = []
    if not any(info["alpha"]):
        return out
    states = info["kind"] == "states"
    what = "mots de la signature de l'état injecté" if states else "mots injectés"
    in_band = [li for li, l in enumerate(layers) if l in drug_layers]
    above = [li for li, l in enumerate(layers) if l > max(drug_layers) and l in band]
    k = min(len(m["mass"]), len(b["mass"])) if b else len(m["mass"])

    def share(x, idx):
        return float(torch.nanmean(x[:k, idx]))
    base_txt = f" ; témoin sobre, mêmes mots aux mêmes pas : {share(b['mass'], in_band):.2%}" if b else ""
    overlap = sorted(set(band) & set(drug_layers))
    m_raw = m.get("raw")
    if main["js"].get("subtract_injection") and m_raw:
        ok = share(m["mass"], in_band) < share(m_raw["mass"], in_band) / 3
        # un état n'est pas construit à partir de la lentille : sa signature se relit peu, et ce qui en reste après
        # correction est surtout la réaction du modèle (son texte et sa pensée prennent la couleur de l'état)
        out.append({"status": "good" if ok else "info" if states else "warning", "title": "Lecture sans l'injection",
                    "detail": f"la lentille lit h moins la drogue injectée aux couches précédentes. Part des {what} "
                              f"dans les couches {min(drug_layers)}..{max(drug_layers)} : {share(m_raw['mass'], in_band):.1%} "
                              f"en lecture brute, {share(m['mass'], in_band):.2%} après correction{base_txt} ; au-dessus "
                              f"({layers[above[0]]}..{layers[above[-1]]}) : {share(m_raw['mass'], above):.1%} brute, "
                              f"{share(m['mass'], above):.2%} corrigée. Le reste vient des couches du modèle elles-mêmes, "
                              "qui reprennent la direction injectée : c'est sa réaction"})
    elif overlap and above:
        out.append({"status": "warning", "title": "La lentille relit l'injection",
                    "detail": f"part de la lecture prise par les {what} : {share(m['mass'], in_band):.1%} dans les couches "
                              f"injectées {min(drug_layers)}..{max(drug_layers)}{base_txt}, {share(m['mass'], above):.1%} au-dessus "
                              f"({layers[above[0]]}..{layers[above[-1]]}). La bande lue pour la vidéo ({band[0]}..{band[-1]}) "
                              f"contient les couches injectées {overlap[0]}..{overlap[-1]} (readout.subtract_injection corrige)"})
    inj = total = 0
    seen = Counter()
    for w in main["windows"]:
        drawn = {x.lower() for ws in info["marks"][w["start"]:w["end"]] for x in ws}
        for word, _ in w["words"]:
            total += 1
            if word in drawn:
                inj += 1
                seen[word] += 1
    if total:
        out.append({"status": "warning" if inj / total > 0.05 and not states else "good" if inj / total <= 0.05 else "info",
                    "title": "Mots de la vidéo = " + ("signature de l'état" if states else "mots injectés"),
                    "detail": f"{inj}/{total} mots des tranches ({inj / total:.0%}) sont des {what} pendant la tranche"
                              + (" : " + ", ".join(w for w, _ in seen.most_common(10)) if seen else "")})
    if states:
        sig = info["signatures"]
        out.append({"status": "info", "title": "Signature lexicale des états",
                    "detail": f"ce que la lentille lit dans chaque vecteur d'état (couche {max(drug_layers)}) : "
                              + " ; ".join(f"{k} : {', '.join(v[:8])}" for k, v in sig.items())})
    return out


def pca2(xs: list[torch.Tensor]) -> list[list[list[float]]]:
    """Projection des runs sur les 2 premières composantes principales de leur union."""
    X = torch.cat(xs).float()
    mu = X.mean(0)
    _, _, V = torch.pca_lowrank(X - mu, q=2, center=False)
    return [[[r(float(a), 3), r(float(c), 3)] for a, c in ((x.float() - mu) @ V[:, :2]).tolist()] for x in xs]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--baseline")
    a = ap.parse_args(argv)
    from .lensmodel import Reader, load_lens, load_model

    main_run = load_run(Path(a.run))
    base = load_run(Path(a.baseline)) if a.baseline else None
    cfg = main_run["cfg"]
    model, tok = load_model(cfg)
    reader = Reader(model, tok, load_lens(cfg), cfg)
    dec = Decoder(reader, tok)
    layers = main_run["js"]["layers"]
    band = [l for l in reader.band if l in layers]
    from .drugs import catalog
    lo, hi = catalog().get("band", [0.45, 0.70]) if (cfg.get("drug") or {}).get("name") else \
        ((cfg.get("drugs") or [{}])[0].get("band", [0.45, 0.70]))
    drug_layers = [l for l in reader.lens.source_layers if lo * reader.n_layers <= l <= hi * reader.n_layers]
    words, alpha, phase = drug_info(main_run["trace"])
    states = drug_states(main_run["trace"])
    dec.directions({w for ws in words for w in ws}, layers)
    info = {"words": words, "states": states, "alpha": alpha, "kind": "states" if any(states) else "words",
            "marks": words, "labels": [", ".join(ws) for ws in words], "signatures": {}}
    if info["kind"] == "states":
        # drogue par états : les « mots marqués » sont la signature lexicale de l'état dominant à chaque pas
        from .states import load_pack
        dec.units = {k: s["unit"] for k, s in load_pack(cfg)["states"].items()}
        info["signatures"] = {k: dec.signature(k, max(drug_layers)) for k in sorted({k for s in states for k in s})}
        info["marks"] = [info["signatures"][max(s, key=s.get)] if s else [] for s in states]
        info["labels"] = [" + ".join(f"{k} {w:.0%}" for k, w in s.items()) for s in states]

    m = dec.run(main_run["js"], info["marks"])
    if main_run["js"].get("subtract_injection") and "inj" in main_run["js"]:
        # lecture brute, pour comparer : J @ h = J @ (h − inj) + J @ inj
        J = torch.stack([reader.lens.jacobians[l] for l in layers]).float()
        raw = main_run["js"]["jh"].float() + torch.einsum("lij,tlj->tli", J, main_run["js"]["inj"].float())
        m["raw"] = dec.run({**main_run["js"], "jh": raw}, info["marks"])
    # témoin : part dans sa lecture des mots que la drogue marquait au même pas du run drogué
    b = dec.run(base["js"], info["marks"][: len(base["js"]["token_ids"])]) if base else None
    # cos(h, direction injectée) par couche
    cosd = torch.full(m["hn"].shape, float("nan"))
    for t in range(len(words)):
        for li, l in enumerate(layers):
            sd = dec.step_dir(words[t], states[t], l)
            if sd is not None:
                cosd[t, li] = float(torch.nn.functional.cosine_similarity(main_run["js"]["h"][t, li].float(), sd[0], dim=0))
    chk = checks(main_run, base, m, b, dec, layers, band, drug_layers, info)

    def pack(run, met, extra=None):
        js = run["js"]
        T, L = met["hn"].shape
        tops = [[[dec.piece(int(i)) for i in met["top_ids"][t, l]] for l in range(L)] for t in range(T)]
        topp = [[[r(float(p), 3) for p in met["top_p"][t, l]] for l in range(L)] for t in range(T)]
        out = {"id": run["dir"].name, "tokens": js["text"], "alpha": [r(x[0] if x else 0.0, 3) for x in js["alpha"]],
               "tops": tops, "topp": topp,
               "metrics": {k: [[r(float(v), 4) for v in row] for row in met[k].tolist()]
                           for k in ("hn", "jn", "ent", "p1", "rank", "pnext", "mass", "conv")}}
        out["metrics"].update(extra or {})
        return out

    pcs = {}
    for li, l in enumerate(layers):
        xs = [main_run["js"]["jh"][:, li]] + ([base["js"]["jh"][:, li]] if base else [])
        pcs[l] = pca2(xs)
    data = {
        "run": main_run["dir"].name, "baseline": base["dir"].name if base else None,
        "question": main_run["js"]["phrase"], "model": cfg["model"], "lens": cfg["lens"]["file"],
        "drug": cfg.get("drug") or None, "layers": layers, "band": band, "drug_layers": drug_layers,
        "phase": phase, "drug_words": info["marks"], "drug_labels": info["labels"], "drug_kind": info["kind"],
        "checks": chk,
        "main": pack(main_run, m, {"cosd": [[r(float(v), 4) for v in row] for row in cosd.tolist()]}),
        "base": pack(base, b) if base else None,
        "pca": pcs,
        "windows": [{"index": w["index"], "start": w["start"], "end": w["end"], "words": w["words"][:8],
                     "alpha": (w.get("drug_alpha_max") or [0])[0]} for w in main_run["windows"]],
        "frames": thumbs(main_run["dir"]),
        "text": (main_run["dir"] / "text.txt").read_text(encoding="utf-8"),
        "text_base": (base["dir"] / "text.txt").read_text(encoding="utf-8") if base else None,
    }
    out_dir = main_run["dir"]
    (out_dir / "analyse.json").write_text(json.dumps({k: v for k, v in data.items() if k != "frames"}, ensure_ascii=False))
    for c in chk:
        print(f"[{c['status']:8s}] {c['title']} : {c['detail']}")
    tpl = ROOT / "introspect" / "analyse.html"
    if tpl.exists():
        payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
        (out_dir / "analyse.html").write_text(tpl.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload),
                                              encoding="utf-8")
        print(out_dir / "analyse.html")


if __name__ == "__main__":
    main()
