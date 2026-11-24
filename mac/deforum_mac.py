"""Mouvement de caméra 3D façon Deforum à travers les images d'un run (sans diffusion).

Pour chaque tranche : carte de profondeur de l'image (Depth Anything V2 Small), puis une caméra virtuelle
qui avance dans la scène (translation z), tourne doucement et fait varier son champ de vision (FOV).
On reprojette toujours l'image clé d'origine, donc rien ne se dégrade d'une image vidéo à l'autre.
Transition A -> B : le fond de A s'efface d'abord au profit de B, qui arrive de plus loin, comme si l'on
traversait A. Le son est celui du montage normal (audio_voix.wav, produit par assemble.py).

    python mac/deforum_mac.py --run runs/q1_rave --seconds 10 --preview     # aperçu 960x540, 10 s
    python mac/deforum_mac.py --run runs/q1_rave                            # vidéo complète 1920x1080
    python mac/deforum_mac.py --run runs/q1_rave --deforum FovVid1CompSettings.txt   # mouvement d'un réglage Deforum
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from introspect.assemble import durations, fit
from introspect.common import Log, run_config

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def depth_map(pipe, img: Image.Image, W: int, H: int) -> np.ndarray:
    """Profondeur métrique approximative : 1 (près) à ~7 (loin), à la taille W x H."""
    disp = np.asarray(pipe(img)["depth"].resize((W, H), Image.BILINEAR), np.float32)
    from scipy.ndimage import gaussian_filter
    disp = gaussian_filter(disp, sigma=max(2, W // 240))
    disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)  # Depth Anything : grand = proche
    return 1.0 / (0.15 + 0.85 * disp)


def rot(rx: float, ry: float, rz: float, dev) -> torch.Tensor:
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)
    return (Rz @ Ry @ Rx).to(dev)


class Warper:
    """Reprojection inverse : pour chaque pixel de la caméra déplacée, où lire dans l'image clé."""

    def __init__(self, W: int, H: int, dev):
        self.W, self.H, self.dev = W, H, dev
        v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                              torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
        self.u, self.v = u - (W - 1) / 2, v - (H - 1) / 2

    def __call__(self, img: torch.Tensor, depth: torch.Tensor, tz: float, R: torch.Tensor,
                 fov_src: float, fov_out: float) -> torch.Tensor:
        f_src = (self.W / 2) / math.tan(math.radians(fov_src) / 2)
        f_out = (self.W / 2) / math.tan(math.radians(fov_out) / 2)
        # inversion itérative : on cherche le point source dont la profondeur, une fois la caméra déplacée,
        # retombe sur ce pixel (une seule lecture au même endroit dédouble les objets proches)
        d4 = depth[None, None]
        zs = depth
        for _ in range(3):
            z = (zs - tz).clamp_min(0.05)
            P = torch.stack([self.u / f_out * z, self.v / f_out * z, z], -1) @ R.T  # caméra déplacée, puis rotation
            Pz = (P[..., 2] + tz).clamp_min(0.05)  # retour dans le repère de l'image clé
            us, vs = P[..., 0] / Pz * f_src, P[..., 1] / Pz * f_src
            grid = torch.stack([us / ((self.W - 1) / 2), vs / ((self.H - 1) / 2)], -1)[None]
            zs = F.grid_sample(d4, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=True)


def camera(t: float, d: float, k: int, a, unit: float, h_in: float) -> tuple[float, tuple, float]:
    """Pose de caméra au temps t (s) d'une image clé de durée d. t peut être négatif (arrivée) ou > d (départ).
    Vitesse d'avance constante : a.speed par `unit` secondes (durée moyenne d'une tranche)."""
    s = t / d
    # avance en unités de profondeur (1 = plan le plus proche) ; 0 au début de la transition d'arrivée,
    # jamais négative : reculer ferait voir hors de l'image
    tz = min(0.85, a.speed * max(0.0, t + h_in) / unit)
    sgn = 1 if k % 2 == 0 else -1  # alterne le sens de rotation d'une image clé à l'autre
    rz = math.radians(a.roll) * sgn * s
    ry = math.radians(a.yaw) * math.sin(math.pi * s) * sgn
    # respiration du FOV, toujours <= FOV source : on ne voit jamais hors de l'image
    fov = a.fov - a.fov_swing * (0.5 + 0.5 * math.sin(2 * math.pi * (s * 0.5 + 0.25 * k)))
    return tz, (0.0, ry, rz), fov


def parse_schedule(txt: str, n: int) -> np.ndarray:
    """Courbe Deforum "0: (0.00), 10: (0.55), …" -> valeur à chaque image 0..n-1 (interpolation linéaire
    entre images clés, valeur tenue avant la première et après la dernière, comme Deforum)."""
    import re

    pts = sorted((int(f), float(eval(v, {"__builtins__": {}}, {}))) for f, v in re.findall(r"(\d+)\s*:\s*\(([^)]*)\)", txt))
    if not pts:
        return np.zeros(n)
    xs, ys = zip(*pts)
    return np.interp(np.arange(n), xs, ys)


class DeforumMotion:
    """Mouvement lu dans un réglage Deforum (mode 3D). Les courbes y sont des vitesses par image :
    translation_z en unités Deforum (÷200), rotations en degrés par image. On les intègre depuis l'arrivée de
    chaque image clé ; la courbe de `max_frames` images est rejouée en boucle sur toute la vidéo."""

    def __init__(self, path: Path, fps: int, total: float, z_scale: float, cycle_fps: float | None):
        d = json.loads(Path(path).read_text())
        self.n = int(d.get("max_frames", 150))
        self.cfps = cycle_fps or fps  # images de la courbe par seconde de vidéo
        self.fps = fps
        self.z_scale = z_scale
        n_total = int(total * fps) + 2
        g = (np.arange(n_total) / fps * self.cfps).astype(int) % self.n  # image de la courbe à chaque image vidéo
        step = self.cfps / fps  # images de courbe parcourues par image vidéo
        curve = lambda k: parse_schedule(d.get(k, "0:(0)"), self.n)[g] * step
        self.C = {k: np.concatenate([[0], np.cumsum(curve(k))]) for k in
                  ("translation_z", "rotation_3d_x", "rotation_3d_y", "rotation_3d_z")}
        self.fov = parse_schedule(d.get("fov_schedule", "0:(60)"), self.n)[g]
        self.desc = (f"{Path(path).name} : {self.n} images de courbe, {self.cfps:g} par seconde de vidéo, "
                     f"FOV {self.fov.min():g}-{self.fov.max():g}°")

    def pose(self, t_from: float, t: float):
        i0 = max(0, min(len(self.fov) - 1, int(round(t_from * self.fps))))
        i1 = max(0, min(len(self.fov) - 1, int(round(t * self.fps))))
        dz = (self.C["translation_z"][i1] - self.C["translation_z"][i0]) / 200 * self.z_scale
        tz = 0.85 * (1 - math.exp(-max(0.0, dz) / 0.85))  # approche douce du plan le plus proche, jamais au-delà
        rx, ry, rz = (math.radians(self.C[k][i1] - self.C[k][i0]) for k in ("rotation_3d_x", "rotation_3d_y", "rotation_3d_z"))
        return tz, (rx, ry, rz), float(self.fov[i1])


def color_match(arr: np.ndarray, ref: np.ndarray, w: float) -> np.ndarray:
    """Transfert de moyenne et d'écart-type par canal en YCbCr (proche du `color_coherence` de Deforum),
    dosé par w."""
    if w <= 0:
        return arr
    a = np.asarray(Image.fromarray(arr).convert("YCbCr"), np.float32)
    r = np.asarray(Image.fromarray(ref).convert("YCbCr"), np.float32)
    m = (a - a.mean((0, 1))) / (a.std((0, 1)) + 1e-6) * r.std((0, 1)) + r.mean((0, 1))
    out = np.clip(a * (1 - w) + m * w, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(out, "YCbCr").convert("RGB"))


def keyframes(run_dir: Path, durs: list[float]) -> list[dict]:
    """Images clés dans l'ordre : toutes les images d'une tranche (frames.json, clé `sub`) se partagent
    sa durée à parts égales."""
    fj = run_dir / "frames.json"
    man = json.loads(fj.read_text()) if fj.exists() else []
    out, t = [], 0.0
    for k, d in enumerate(durs):
        files = sorted((m for m in man if m["window"] == k and not m.get("intermediate")), key=lambda m: m.get("sub", 0))
        files = [m["file"] for m in files] or [f"{k:04d}.png"]
        for j, f in enumerate(files):
            out.append({"file": f, "start": t + d * j / len(files), "dur": d / len(files), "slice": k, "sub": j})
        t += d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seconds", type=float, default=None, help="ne rendre que le début (aperçu)")
    ap.add_argument("--preview", action="store_true", help="960x540")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--speed", type=float, default=0.35, help="avance par tranche moyenne (unités de profondeur)")
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--fov-swing", type=float, default=12.0, help="amplitude de variation du FOV (degrés)")
    ap.add_argument("--roll", type=float, default=4.0, help="rotation autour de l'axe de visée par image clé (degrés)")
    ap.add_argument("--yaw", type=float, default=3.0)
    ap.add_argument("--xfade", type=float, default=1.6, help="transition entre deux tranches (s)")
    ap.add_argument("--sub-xfade", type=float, default=0.9, help="transition entre images d'une même tranche (s)")
    ap.add_argument("--emerge", type=float, default=0.3,
                    help="taille de l'image suivante quand elle apparaît au centre (fraction de l'écran)")
    ap.add_argument("--color-coherence", type=float, default=0.6,
                    help="rapproche les couleurs de chaque image clé de la précédente (0 = non, 1 = copie la palette)")
    ap.add_argument("--deforum", default=None, help="réglage Deforum (JSON) : reprend translation_z, rotation_3d_*, fov_schedule")
    ap.add_argument("--z-scale", type=float, default=1.0, help="facteur sur translation_z du réglage Deforum")
    ap.add_argument("--cycle-fps", type=float, default=None,
                    help="images de courbe Deforum par seconde (défaut : fps de la vidéo)")
    ap.add_argument("--overscan", type=float, default=None,
                    help="cadrage plus serré que l'image source (1.15 = 15 %%) pour cacher les bords lors des rotations ; "
                         "défaut 1.15 avec --deforum, 1.0 sinon")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run_dir = Path(a.run)
    cfg = run_config(run_dir)
    log = Log(run_dir, "deforum")
    vc = cfg["video"]
    W, H = (960, 540) if a.preview else (vc.get("width", 1920), vc.get("height", 1080))
    fps = a.fps or vc["fps"]
    wins = json.loads((run_dir / "windows.json").read_text())
    durs = durations(run_dir, cfg, len(wins))
    unit = float(np.mean(durs))
    K = keyframes(run_dir, durs)
    total = sum(durs) + vc["crossfade_s"]
    if a.seconds:
        total = min(total, a.seconds)
    K = [kf for kf in K if kf["start"] < total + a.xfade]
    starts = np.array([kf["start"] for kf in K])
    # durée pendant laquelle chaque image clé émerge au centre de la précédente, avant son début
    # (longue si elle ouvre une tranche, jamais plus de 85 % de la durée de la précédente)
    lead = [0.0] + [min(a.xfade if K[i]["sub"] == 0 else a.sub_xfade, K[i - 1]["dur"] * 0.85) for i in range(1, len(K))]
    motion = DeforumMotion(Path(a.deforum), fps, total + a.xfade, a.z_scale, a.cycle_fps) if a.deforum else None
    if motion:
        log("mouvement Deforum :", motion.desc)
    fov_src = float(motion.fov.max()) if motion else a.fov
    overscan = a.overscan or (1.15 if motion else 1.0)
    wav = run_dir / "audio_voix.wav"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    from transformers import pipeline
    dp = pipeline("depth-estimation", model=DEPTH_MODEL, device=dev)
    keys, depths = [], []
    t0 = time.time()
    prev = None
    for kf in K:
        arr = fit(Image.open(run_dir / "frames" / kf["file"]).convert("RGB"), W, H)
        if prev is not None:
            arr = color_match(arr, prev, a.color_coherence)
        prev = arr
        keys.append(torch.from_numpy(arr).to(dev).permute(2, 0, 1)[None].float() / 255)
        depths.append(torch.from_numpy(depth_map(dp, Image.fromarray(arr), W, H)).to(dev))
    del dp
    log(f"{len(K)} images clés sur {len({kf['slice'] for kf in K})} tranches, profondeur en {time.time() - t0:.0f} s, "
        f"rendu {W}x{H} à {fps} i/s sur {total:.1f} s")

    out = Path(a.out) if a.out else run_dir / f"deforum{'_preview' if a.preview or a.seconds else ''}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-"]
    if wav.exists():
        cmd += ["-i", str(wav), "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    warp = Warper(W, H, dev)
    cards: dict = {}
    nframes = int(total * fps)
    t0 = time.time()
    with torch.no_grad():
        for fi in range(nframes):
            t = fi / fps
            i = max(0, min(len(K) - 1, int(np.searchsorted(starts, t, side="right") - 1)))

            def view(j, at=None):
                tt = t if at is None else at
                if motion:
                    tz, (rx, ry, rz), fov = motion.pose(K[j]["start"], tt)
                else:
                    tz, (rx, ry, rz), fov = camera(tt - K[j]["start"], K[j]["dur"], j, a, unit, 0.0)
                fov = math.degrees(2 * math.atan(math.tan(math.radians(fov) / 2) / overscan))
                return warp(keys[j], depths[j], tz, rot(rx, ry, rz, dev), fov_src, fov)

            frame = view(i)
            # zoom infini : l'image clé suivante apparaît petite au centre (là où l'on va) et grandit au rythme
            # de l'avance jusqu'à remplir l'écran à son début ; on ne recule jamais
            j = i + 1
            if j < len(K) and t >= K[j]["start"] - lead[j]:
                L, t_end = lead[j], K[j]["start"]
                p = min(1.0, max(0.0, (t - (t_end - L)) / L))  # 0 -> 1 pendant l'émergence
                s0 = a.emerge
                sc = 1.0 / (1.0 + (1.0 / s0 - 1.0) * (1.0 - p))  # perspective : s0 -> 1
                if j not in cards:
                    cards.clear()
                    cards[j] = view(j, at=K[j]["start"])  # B tel qu'il sera à son début
                gx, gy = warp.u / ((W - 1) / 2) / sc, warp.v / ((H - 1) / 2) / sc
                card = F.grid_sample(cards[j], torch.stack([gx, gy], -1)[None], mode="bilinear",
                                     padding_mode="reflection", align_corners=True)
                # masque radial doux (pas de rectangle) : large au début, se resserre sur les bords à la fin
                rr = torch.sqrt(gx * gx + gy * gy)  # ellipse : 0 au centre, 1 au milieu des bords de B
                f = 0.9 * (1.0 - p) + 0.05
                m = ((1 - rr) / f).clamp(0, 1)
                m = m * m * (3 - 2 * m)
                cover = min(1.0, max(0.0, (p - 0.8) / 0.2))  # à la fin, B couvre tout l'écran
                m = torch.maximum(m, torch.full_like(m, cover * cover * (3 - 2 * cover)))
                # jamais hors de la surface de B (sinon on voit ses bords étirés)
                inside = ((1 - torch.maximum(gx.abs(), gy.abs())) / 0.15).clamp(0, 1)
                inside = inside * inside * (3 - 2 * inside)
                m = m * inside
                # B commence derrière les objets proches de A : il ne perce d'abord que là où A est lointain
                far = (depths[i] - 1) / (depths[i].max() - 1 + 1e-6)
                m = m * ((far - (1 - 1.4 * p)) / 0.25 + 0.5).clamp(0, 1)
                fade = min(1.0, p / 0.35)
                m = m * fade * fade * (3 - 2 * fade)
                frame = frame * (1 - m) + card * m
            px = (frame[0].clamp(0, 1) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            proc.stdin.write(px.tobytes())
            if fi % (fps * 5) == 0:
                log(f"{t:5.1f} s / {total:.1f} s (image clé {i}, tranche {K[i]['slice']}), "
                    f"{fi / max(1e-6, time.time() - t0):.1f} i/s")
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg a échoué")
    log(f"{out.name} : {nframes} images en {time.time() - t0:.0f} s")
    log.done()
    print(out)


if __name__ == "__main__":
    main()
