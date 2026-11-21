"""Utilitaires partagés : configuration, dossiers de run, journal."""
from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "weights" / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def cached_snapshot(repo: str) -> str:
    """Dossier du dépôt HF dans le cache s'il y est, sinon l'identifiant tel quel. Pour charger un dépôt dont
    on n'a téléchargé qu'une partie (Stable Audio : dossiers diffusers sans les checkpoints à la racine) :
    à partir de l'identifiant, diffusers refuse un instantané incomplet."""
    d = Path(os.environ["HF_HUB_CACHE"]) / ("models--" + repo.replace("/", "--"))
    ref = d / "refs" / "main"
    if ref.exists() and (d / "snapshots" / ref.read_text().strip()).is_dir():
        return str(d / "snapshots" / ref.read_text().strip())
    return repo


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    """Charge une config ; elle peut hériter d'une autre via `base: chemin`."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is None and path.resolve() != (ROOT / "configs" / "default.yaml").resolve():
        base = str(ROOT / "configs" / "default.yaml")
    if base:
        bp = Path(base)
        if not bp.is_absolute():
            bp = (path.parent / bp) if (path.parent / bp).exists() else ROOT / bp
        cfg = deep_merge(load_config(bp), cfg)
    return cfg


def new_run_dir(cfg: dict, run_id: str | None = None) -> Path:
    finalize(cfg)  # clés v2 résolues une fois, la config du run garde les valeurs effectives
    if run_id is None:
        tag = cfg.get("run_tag", "run")
        run_id = f"{_dt.datetime.now():%Y%m%d-%H%M%S}-{tag}"
    d = ROOT / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return d


def run_config(run_dir: Path) -> dict:
    with open(Path(run_dir) / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Log:
    """Journal simple : stdout + runs/<id>/log.txt, avec chronométrage."""

    def __init__(self, run_dir: Path, stage: str):
        self.path = Path(run_dir) / "log.txt"
        self.stage = stage
        self.t0 = time.time()

    def __call__(self, *msg):
        line = f"[{_dt.datetime.now():%H:%M:%S}] [{self.stage}] " + " ".join(str(m) for m in msg)
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def done(self):
        dt = time.time() - self.t0
        self(f"terminé en {dt:.1f} s")
        timings = Path(self.path).parent / "timings.json"
        data = json.loads(timings.read_text()) if timings.exists() else {}
        data[self.stage] = round(dt, 1)
        timings.write_text(json.dumps(data, indent=2))
        return dt


# ------------------------------------------------------------------ v2 : résolution des clés
def n_slices(cfg: dict) -> int | None:
    T = cfg["video"].get("target_duration_s")
    if not T:
        return None
    return max(2, round(T / cfg["video"].get("slice_target_s", 3.0)))


def finalize(cfg: dict) -> dict:
    """Résout les clés v2 (environnement, drogue du catalogue, durée -> nombre de tokens).
    Idempotent : une config déjà résolue n'est pas modifiée une seconde fois."""
    if cfg.get("_resolved"):
        return cfg
    inp = cfg["input"]
    env = inp.get("environment")
    if env and env != "aucun":
        envs = yaml.safe_load((ROOT / "configs" / "environments.yaml").read_text(encoding="utf-8"))
        if env not in envs:
            raise ValueError(f"environnement inconnu : {env}")
        e = envs[env]
        inp["system"] = e["prompt"] + " " + inp.get("system", "")
        if inp.get("environment_style"):
            cfg["image"]["style"] = (cfg["image"].get("style", "") + ", " + e["image_style"]).strip(", ")
            cfg["sound"]["sono"]["layer_a"]["suffix"] = e["sound_texture"]
    dc = cfg.get("drug") or {}
    if dc.get("name") in (None, "", "aucune"):
        cfg["drug"] = {}
    n = n_slices(cfg)
    if n and cfg.get("auto_tokens", True):
        S, W = cfg["windows"]["stride"], cfg["windows"]["size"]
        cfg["generation"]["max_new_tokens"] = int(min(1500, max(200, S * n + W)))
        cfg["generation"]["block_eos"] = True
    cfg["_resolved"] = True
    return cfg
