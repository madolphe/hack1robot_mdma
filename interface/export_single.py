"""Exporte interface/bureau.html en un seul fichier HTML autonome (vidéos, données, typo.js embarqués).

Le fichier s'ouvre d'un double-clic dans Chrome, sans serveur ni connexion : les polices (Recursive, Roboto Flex,
Roboto Mono, sous-ensembles latin et latin-ext) sont embarquées depuis interface/fonts/fonts.css.

    python interface/export_single.py                 # -> interface/dist/hack1robo_bureau.html
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drugs", default="mdma,champignons,neutral", help="expériences à embarquer (clés de experiences.json)")
    ap.add_argument("--out", default=str(HERE / "dist" / "hack1robo_bureau.html"))
    ap.add_argument("--google-fonts", action="store_true", help="garder Google Fonts au lieu d'embarquer interface/fonts")
    a = ap.parse_args()

    html = (HERE / "bureau.html").read_text(encoding="utf-8")
    typo = (HERE / "typo.js").read_text(encoding="utf-8")
    exps = json.loads((HERE / "data" / "experiences.json").read_text(encoding="utf-8"))
    keep = {k: exps[k] for k in a.drugs.split(",") if k in exps}

    js, media = {"data/experiences.json": keep}, {}
    for eid in keep.values():
        e = json.loads((HERE / "data" / f"exp_{eid}.json").read_text(encoding="utf-8"))
        js[f"data/exp_{eid}.json"] = e
        for p in (e.get("video"), e.get("audio")):
            if p and p not in media:
                media[p] = base64.b64encode((HERE / p).read_bytes()).decode("ascii")

    # médias : un <script type=text/plain> par fichier (lu sans passer par le parseur JS)
    blocks = [f'<script type="text/plain" id="m{i}" data-path="{p}">{b64}</script>' for i, (p, b64) in enumerate(media.items())]
    mime = lambda p: "audio/wav" if p.endswith(".wav") else "video/mp4"
    shim = """<script>
/* export autonome : fetch() sert les données et les médias embarqués */
const EMBED_JSON = %s, EMBED_MIME = %s;
const _fetch = window.fetch.bind(window);
window.fetch = async (u, o) => { u = String(u);
  if (u in EMBED_JSON) return new Response(JSON.stringify(EMBED_JSON[u]), {headers: {"Content-Type": "application/json"}});
  const el = document.querySelector(`script[data-path="${u}"]`);
  if (el) return _fetch(`data:${EMBED_MIME[u]};base64,` + el.textContent.trim());
  return _fetch(u, o); };
</script>""" % (json.dumps(js, ensure_ascii=False), json.dumps({p: mime(p) for p in media}))

    fonts = HERE / "fonts" / "fonts.css"
    if not a.google_fonts and fonts.exists():  # polices embarquées à la place de Google Fonts
        css = re.sub(r"url\(([\w.-]+\.woff2)\)", lambda m: "url(data:font/woff2;base64," + base64.b64encode((fonts.parent / m.group(1)).read_bytes()).decode() + ")",
                     fonts.read_text(encoding="utf-8"))
        html = re.sub(r'<link rel="preconnect"[^>]*>\n?', "", html)
        html = re.sub(r'<link href="https://fonts.googleapis.com[^>]*>', lambda m: "<style>\n" + css + "</style>", html)

    html = html.replace('<script src="typo.js"></script>', "\n".join(blocks) + "\n" + shim + "\n<script>\n" + typo + "\n</script>")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"{out} : {out.stat().st_size / 1e6:.1f} Mo, expériences {list(keep)}, médias {list(media)}")


if __name__ == "__main__":
    main()
