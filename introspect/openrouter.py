"""Client OpenRouter : client OpenAI pointé sur https://openrouter.ai/api/v1.

La clé vient de la variable d'environnement OPENROUTER_API_KEY (jamais d'un fichier du dépôt).
Les modèles se choisissent par slug : https://openrouter.ai/models

    export OPENROUTER_API_KEY=...                      # créée sur https://openrouter.ai/keys
    python -m introspect.openrouter "Le silence est-il une forme de langage ?"
    python -m introspect.openrouter "..." --model anthropic/claude-sonnet-5.5 --system "Réponds en une phrase."
    python -m introspect.openrouter "A surreal dreamlike oil painting of silence" --image out.png   # image
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-opus-5.5"
DEFAULT_IMAGE_MODEL = "recraft/recraft-v4.1-flash-v2:free"
APP_HEADERS = {"X-OpenRouter-Title": "hack1robo"}  # attribution facultative de l'app sur openrouter.ai


class OpenRouterError(RuntimeError):
    """Erreur lisible : clé absente ou refusée, crédit épuisé, quota, modèle inconnu, réseau."""


def client():
    from openai import OpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError("OPENROUTER_API_KEY n'est pas définie : crée une clé sur https://openrouter.ai/keys "
                              "puis `export OPENROUTER_API_KEY=...`")
    return OpenAI(base_url=BASE_URL, api_key=key, default_headers=APP_HEADERS, max_retries=2, timeout=120)


def _call(model: str, **kw):
    """Appel chat completions avec des erreurs lisibles."""
    import openai

    try:
        r = client().chat.completions.create(model=model, **kw)
    except openai.AuthenticationError as e:
        raise OpenRouterError(f"clé refusée (401) : vérifie OPENROUTER_API_KEY. {e.message}") from e
    except openai.RateLimitError as e:
        raise OpenRouterError(f"trop de requêtes (429), réessaie plus tard. {e.message}") from e
    except openai.NotFoundError as e:
        raise OpenRouterError(f"modèle introuvable : {model!r} (liste : https://openrouter.ai/models). {e.message}") from e
    except openai.APIStatusError as e:
        hint = " : crédit insuffisant, recharge sur https://openrouter.ai/settings/credits" if e.status_code == 402 else ""
        raise OpenRouterError(f"erreur API {e.status_code}{hint}. {e.message}") from e
    except openai.APIConnectionError as e:
        raise OpenRouterError(f"OpenRouter injoignable : {e}") from e
    if not r.choices:
        raise OpenRouterError(f"réponse sans contenu : {r}")
    return r


def image(prompt: str, out: str | Path, model: str = DEFAULT_IMAGE_MODEL, aspect_ratio: str | None = "16:9") -> Path:
    """Génère une image (modèle texte -> image) et l'enregistre dans `out`. Renvoie le chemin écrit."""
    import base64

    body = {"modalities": ["image"]}
    if aspect_ratio:
        body["image_config"] = {"aspect_ratio": aspect_ratio}
    r = _call(model, messages=[{"role": "user", "content": prompt}], extra_body=body)
    msg = r.choices[0].message
    images = getattr(msg, "images", None) or (msg.model_extra or {}).get("images") or []
    if not images:
        raise OpenRouterError(f"aucune image dans la réponse (texte : {msg.content!r})")
    first = images[0]
    url = (first.get("image_url") or {}).get("url") if isinstance(first, dict) else first.image_url.url
    if not url or not url.startswith("data:"):
        raise OpenRouterError(f"format d'image inattendu : {str(url)[:80]!r}")
    header, b64 = url.split(",", 1)
    ext = header.split("/")[1].split(";")[0]  # png, jpeg, webp, svg+xml…
    out = Path(out)
    if out.suffix.lstrip(".") != ext.replace("jpeg", "jpg").replace("+xml", ""):
        out = out.with_suffix("." + ext.replace("jpeg", "jpg").replace("+xml", ""))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(b64))
    return out


def images_api(prompt: str, out: str | Path, model: str, aspect_ratio: str | None = "16:9",
               resolution: str | None = None, seed: int | None = None, refs: list[str | Path] | None = None) -> dict:
    """API Images d'OpenRouter (POST /api/v1/images), pour les modèles qui l'exigent (ex. tencent/hy-image-*).
    `refs` : images de référence (fichiers locaux ou URL https) pour l'image-à-image.
    Pas de température : la variation se règle avec `seed`. Renvoie {"path", "cost", "usage"}."""
    import base64
    import mimetypes

    import httpx

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError("OPENROUTER_API_KEY n'est pas définie : crée une clé sur https://openrouter.ai/keys")
    body: dict = {"model": model, "prompt": prompt}
    if aspect_ratio:
        body["aspect_ratio"] = aspect_ratio
    if resolution:
        body["resolution"] = resolution
    if seed is not None:
        body["seed"] = seed
    if refs:
        items = []
        for r in refs:
            if str(r).startswith(("http://", "https://", "data:")):
                url = str(r)
            else:
                mt = mimetypes.guess_type(str(r))[0] or "image/png"
                url = f"data:{mt};base64," + base64.b64encode(Path(r).read_bytes()).decode()
            items.append({"type": "image_url", "image_url": {"url": url}})
        body["input_references"] = items
    try:
        resp = httpx.post(f"{BASE_URL}/images", json=body, timeout=300,
                          headers={"Authorization": f"Bearer {key}", **APP_HEADERS})
    except httpx.HTTPError as e:
        raise OpenRouterError(f"OpenRouter injoignable : {e}") from e
    if resp.status_code != 200:
        hint = {401: "clé refusée", 402: "crédit insuffisant", 404: "modèle ou route introuvable",
                429: "trop de requêtes"}.get(resp.status_code, "erreur API")
        raise OpenRouterError(f"{hint} ({resp.status_code}) : {resp.text[:300]}")
    data = resp.json()
    if not data.get("data"):
        raise OpenRouterError(f"aucune image dans la réponse : {str(data)[:300]}")
    img = data["data"][0]
    ext = (img.get("media_type") or "image/png").split("/")[1].split(";")[0].replace("jpeg", "jpg")
    out = Path(out).with_suffix("." + ext)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(img["b64_json"]))
    usage = data.get("usage") or {}
    return {"path": out, "cost": usage.get("cost"), "usage": usage}


def chat(prompt: str, model: str = DEFAULT_MODEL, system: str | None = None,
         max_tokens: int = 1024, temperature: float | None = None) -> str:
    """Une requête chat completions ; renvoie le texte de la réponse."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    kw = {"temperature": temperature} if temperature is not None else {}
    r = _call(model, messages=msgs, max_tokens=max_tokens, **kw)
    text = r.choices[0].message.content or ""
    if r.choices[0].finish_reason == "length":
        text += "\n[réponse tronquée : augmenter max_tokens]"
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Une requête OpenRouter de bout en bout.")
    ap.add_argument("prompt")
    ap.add_argument("--model", default=None, help=f"défaut : {DEFAULT_MODEL} (texte), {DEFAULT_IMAGE_MODEL} (image)")
    ap.add_argument("--image", metavar="FICHIER", help="générer une image et l'enregistrer ici")
    ap.add_argument("--aspect", default="16:9", help="format de l'image (16:9, 1:1, 9:16…)")
    ap.add_argument("--images-api", action="store_true", help="passer par /api/v1/images (ex. tencent/hy-image-*)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ref", action="append", default=None, help="image de référence (répétable)")
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args(argv)
    try:
        if a.image and a.images_api:
            r = images_api(a.prompt, a.image, a.model or DEFAULT_IMAGE_MODEL, a.aspect, seed=a.seed, refs=a.ref)
            print(r["path"], f"coût : {r['cost']} $", r["usage"])
        elif a.image:
            print(image(a.prompt, a.image, a.model or DEFAULT_IMAGE_MODEL, a.aspect))
        else:
            print(chat(a.prompt, a.model or DEFAULT_MODEL, a.system, a.max_tokens))
    except OpenRouterError as e:
        print(f"erreur : {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
