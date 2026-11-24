"""Boucle Deforum 3D avec Sana-Sprint img2img : on avance dans l'image, et la diffusion la répare / la transforme
à chaque pas. Script autonome (ne dépend pas du reste du dépôt hack1robo).

Principe, à chaque image vidéo f :
  1. profondeur de l'image courante (Depth Anything V2 Small) ;
  2. déplacement 3D de la caméra pendant une image (translation_z, rotation_3d_*, fov) lu dans un réglage Deforum,
     puis reprojection de l'image courante (rétroaction : on part toujours de l'image précédente) ;
  3. toutes les `diffusion_cadence` images : contraste, bruit (noise_schedule), puis img2img Sana-Sprint avec le
     prompt du moment et la force strength_schedule ; correction des couleurs sur l'image 0 (color_coherence).

Force (strength) : dans Deforum, strength = part de l'image GARDÉE (0.675 -> on débruite 32,5 %).
  --strength-mode angle (défaut) : le bruit est injecté à l'angle θ = (1 - strength)·π/2 (Sana-Sprint, SCM :
      x_t = cos θ·image + sin θ·bruit), via max_timesteps -> force continue, 2 pas de débruitage.
  --strength-mode steps : mécanisme diffusers standard (strength), arrondi au nombre de pas (paliers de 1/steps).

Exemples :
    python sana_deforum_test.py --settings ../FovVid1CompSettings.txt --frames 30 --W 768 --H 432
    python sana_deforum_test.py --prompt "A surreal dreamlike oil painting of silence, ocean" --frames 60
    python sana_deforum_test.py --init-image depart.png --prompts-json prompts.json --frames 120 --big
Sorties dans --out (défaut ./out_sana_deforum/<horodatage>) : frames/, video.mp4, timings.json, run.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # le client Xet peut rester bloqué à 0 octet

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_16 = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
REPO_06 = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"  # encodeur de texte et VAE identiques au 1.6B
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

DEFAULT_SETTINGS = {  # mouvement de FovVid1CompSettings.txt, si aucun réglage n'est fourni
    "max_frames": 150,
    "translation_z": "0: (0.00), 1: (0.02), 2: (0.04), 3: (0.06), 4: (0.10), 5: (0.11), 6: (0.19), 7: (0.22), 8: (0.20), "
                     "9: (0.21), 10: (0.55), 11: (0.57), 12: (0.62), 13: (0.36), 14: (0.81), 15: (0.73), 16: (1.15), "
                     "17: (1.43), 18: (1.04), 19: (1.65), 20: (1.60), 21: (2.10), 22: (1.90), 23: (1.23), 24: (2.03), "
                     "25: (1.79), 26: (2.56), 27: (2.14), 28: (2.49), 29: (2.41), 30: (1.33), 44: (1.82),  90: (6.75),  "
                     "150: (-1), 151: (2.97)",
    "rotation_3d_x": "0:(0), 150:(0.315)",
    "rotation_3d_y": "0:(0.1),75:(0.2),150:(0.3)",
    "rotation_3d_z": "0:(0), 150:(-0.315)",
    "fov_schedule": "0: (90)",
    "strength_schedule": "0: (0.675)",
    "noise_schedule": "0: (0.065)",
    "contrast_schedule": "0: (1.0)",
    "diffusion_cadence": 2,
    "color_coherence": "Match Frame 0 LAB",
}


# ----------------------------------------------------------------------------- courbes Deforum
def parse_schedule(txt, n: int) -> np.ndarray:
    """"0: (0.00), 10: (0.55)" -> valeur à chaque image 0..n-1 (interpolation linéaire, valeur tenue aux bords)."""
    if isinstance(txt, (int, float)):
        return np.full(n, float(txt))
    pts = sorted((int(f), float(eval(v, {"__builtins__": {}}, {"t": 0}))) for f, v in re.findall(r"(\d+)\s*:\s*\(([^)]*)\)", str(txt)))
    if not pts:
        return np.zeros(n)
    xs, ys = zip(*pts)
    return np.interp(np.arange(n), xs, ys)


def prompt_keys(args, settings) -> list[tuple[int, str]]:
    if args.prompt:
        return [(0, args.prompt)]
    if args.prompts_json:
        d = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
    else:
        d = settings.get("prompts") or {"0": "A surreal dreamlike oil painting, a visual metaphor of silence, memory, ocean"}
    return sorted((int(k), v) for k, v in d.items())


# ----------------------------------------------------------------------------- géométrie
def rot(rx, ry, rz, dev):
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)
    return (Rz @ Ry @ Rx).to(dev)


class Warper:
    """Reprojection inverse avec profondeur : pour chaque pixel de la caméra déplacée, où lire dans l'image courante.
    Inversion itérative (3 passes) pour éviter de dédoubler les objets proches."""

    def __init__(self, W, H, dev):
        self.W, self.H, self.dev = W, H, dev
        v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                              torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
        self.u, self.v = u - (W - 1) / 2, v - (H - 1) / 2

    def __call__(self, img, depth, tz, R, fov):
        f = (self.W / 2) / math.tan(math.radians(fov) / 2)
        d4, zs = depth[None, None], depth
        for _ in range(3):
            z = (zs - tz).clamp_min(0.05)
            P = torch.stack([self.u / f * z, self.v / f * z, z], -1) @ R.T
            Pz = (P[..., 2] + tz).clamp_min(0.05)
            grid = torch.stack([P[..., 0] / Pz * f / ((self.W - 1) / 2), P[..., 1] / Pz * f / ((self.H - 1) / 2)], -1)[None]
            zs = F.grid_sample(d4, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]
        return F.grid_sample(img, grid, mode="bicubic", padding_mode="reflection", align_corners=True).clamp(0, 1)


class Depth:
    def __init__(self, dev):
        from transformers import pipeline
        self.p = pipeline("depth-estimation", model=DEPTH_MODEL, device=dev)

    def __call__(self, img_t, W, H, dev):
        """Profondeur 1 (près) .. ~7 (loin) à partir de l'image [1,3,H,W] dans [0,1]."""
        from scipy.ndimage import gaussian_filter
        pil = Image.fromarray((img_t[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8))
        disp = np.asarray(self.p(pil)["depth"].resize((W, H), Image.BILINEAR), np.float32)
        disp = gaussian_filter(disp, sigma=max(2, W // 240))
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)  # grand = proche
        return torch.from_numpy(1.0 / (0.15 + 0.85 * disp)).to(dev)


# ----------------------------------------------------------------------------- couleur
def to_lab(x):  # x : [H,W,3] float 0..1 (sRGB) -> L*a*b*
    x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = x @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def from_lab(lab):
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    inv = lambda t: np.where(t ** 3 > 0.008856, t ** 3, (t - 16 / 116) / 7.787)
    xyz = np.stack([inv(fx), inv(fy), inv(fz)], -1) * np.array([0.95047, 1.0, 1.08883])
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    x = np.clip(xyz @ M.T, 0, 1)
    return np.where(x > 0.0031308, 1.055 * x ** (1 / 2.4) - 0.055, 12.92 * x)


def match_lab(img_t, ref_lab, w):
    """Transfert moyenne / écart-type en L*a*b* vers l'image de référence (Match Frame 0 LAB), dosé par w."""
    if ref_lab is None or w <= 0:
        return img_t
    x = img_t[0].permute(1, 2, 0).float().cpu().numpy()
    lab = to_lab(x)
    m = (lab - lab.mean((0, 1))) / (lab.std((0, 1)) + 1e-6) * ref_lab[1] + ref_lab[0]
    out = from_lab(lab * (1 - w) + m * w)
    return torch.from_numpy(out.astype(np.float32)).permute(2, 0, 1)[None].to(img_t.device)


# ----------------------------------------------------------------------------- Sana
class Sana:
    binning = False  # True : diffusers ramène W x H à une résolution standard ~1024 px (bien plus lent en petite taille)

    def __init__(self, big, dev, dtype):
        from diffusers import SanaSprintImg2ImgPipeline, SanaSprintPipeline, SanaTransformer2DModel
        kw = {} if big else {"transformer": SanaTransformer2DModel.from_pretrained(REPO_06, subfolder="transformer", torch_dtype=dtype)}
        self.t2i = SanaSprintPipeline.from_pretrained(REPO_16, torch_dtype=dtype, **kw).to(dev)
        self.i2i = SanaSprintImg2ImgPipeline(**self.t2i.components)
        for p in (self.t2i, self.i2i):
            p.set_progress_bar_config(disable=True)
        self.dev, self.cache = dev, {}

    @torch.no_grad()
    def embeds(self, prompt):
        """Encodage du prompt une seule fois (Gemma-2-2B), réutilisé à chaque pas."""
        if prompt not in self.cache:
            e, m = self.t2i.encode_prompt(prompt, device=self.dev)
            self.cache[prompt] = (e, m)
        return self.cache[prompt]

    def free_text_encoder(self):
        """Les prompts sont pré-encodés : l'encodeur de texte ne sert plus. Le libérer évite de saturer la mémoire
        (sur un Mac 16 Go, il fait swapper et ralentit fortement chaque diffusion)."""
        enc = self.t2i.text_encoder
        for p in (self.t2i, self.i2i):
            p.text_encoder = None
        del enc
        import gc
        gc.collect()
        if self.dev == "cuda":
            torch.cuda.empty_cache()
        elif self.dev == "mps":
            torch.mps.empty_cache()

    def mixed(self, pa, pb, w):
        """Mélange linéaire de deux prompts (transition douce entre images clés de prompt)."""
        ea, ma = self.embeds(pa)
        if w <= 0 or pb is None:
            return ea, ma
        eb, mb = self.embeds(pb)
        n = max(ea.shape[1], eb.shape[1])
        pad = lambda e, m: (F.pad(e, (0, 0, 0, n - e.shape[1])), F.pad(m, (0, n - m.shape[1])))
        (ea, ma), (eb, mb) = pad(ea, ma), pad(eb, mb)
        return ea * (1 - w) + eb * w, torch.maximum(ma, mb)

    @torch.no_grad()
    def text2img(self, emb, W, H, gen):
        e, m = emb
        return self.t2i(prompt_embeds=e, prompt_attention_mask=m, width=W, height=H, num_inference_steps=2,
                        generator=gen, output_type="pt", use_resolution_binning=self.binning).images

    @torch.no_grad()
    def img2img(self, img_t, emb, keep, W, H, gen, mode, steps):
        """img_t : [1,3,H,W] 0..1. keep = strength Deforum (part gardée). Renvoie (image, force effective débruitée)."""
        e, m = emb
        denoise = float(np.clip(1.0 - keep, 0.02, 1.0))
        kw = dict(image=img_t, prompt_embeds=e, prompt_attention_mask=m, width=W, height=H, generator=gen, output_type="pt",
                  use_resolution_binning=self.binning)
        if mode == "angle":
            theta = denoise * math.pi / 2
            out = self.i2i(strength=1.0, num_inference_steps=2, max_timesteps=theta,
                           intermediate_timesteps=theta * (1.3 / 1.5708), **kw).images
            return out, denoise
        init = min(steps * denoise, steps)
        eff = (steps - int(max(steps - init, 0))) / steps
        out = self.i2i(strength=denoise, num_inference_steps=steps,
                       intermediate_timesteps=None if steps != 2 else 1.3, **kw).images
        return out, eff


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


# ----------------------------------------------------------------------------- boucle
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", help="réglage Deforum (JSON) ; défaut : mouvement de FovVid1CompSettings.txt")
    ap.add_argument("--prompt", help="un seul prompt pour toute la vidéo")
    ap.add_argument("--prompts-json", help='{"0": "prompt…", "60": "prompt…"} (images vidéo -> prompt)')
    ap.add_argument("--prompt-blend", type=int, default=12, help="images de transition entre deux prompts (0 = coupure)")
    ap.add_argument("--init-image", help="image de départ (sinon générée par Sana à partir du 1er prompt)")
    ap.add_argument("--frames", type=int, default=None, help="nombre d'images vidéo (défaut : max_frames du réglage)")
    ap.add_argument("--W", type=int, default=768)
    ap.add_argument("--H", type=int, default=432)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--cadence", type=int, default=None, help="diffusion toutes les N images (défaut : réglage)")
    ap.add_argument("--strength", type=float, default=None, help="force Deforum fixe (part gardée), remplace le réglage")
    ap.add_argument("--noise", type=float, default=None, help="bruit ajouté avant chaque diffusion, remplace noise_schedule "
                    "(0.065 dans le réglage ; s'accumule en grain si trop fort)")
    ap.add_argument("--strength-mode", choices=["angle", "steps"], default="angle")
    ap.add_argument("--steps", type=int, default=4, help="pas de débruitage en mode steps")
    ap.add_argument("--z-scale", type=float, default=1.0, help="facteur sur translation_z")
    ap.add_argument("--color", type=float, default=1.0, help="dosage de la correction couleur sur l'image 0 (0 = non)")
    ap.add_argument("--big", action="store_true", help="transformer 1.6B au lieu de 0.6B")
    ap.add_argument("--binning", action="store_true",
                    help="laisser diffusers calculer à une résolution standard ~1024 px puis redimensionner (lent, peut-être plus net)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if dev != "cpu" else torch.float32  # fp16 : image noire avec le 1.6B (mesuré sur Mac)
    settings = DEFAULT_SETTINGS | (json.loads(Path(a.settings).read_text()) if a.settings else {})
    N = a.frames or int(settings.get("max_frames", 150))
    sched = lambda k, default: parse_schedule(settings.get(k, default), N)
    tzs, rxs, rys, rzs = (sched(k, "0:(0)") for k in ("translation_z", "rotation_3d_x", "rotation_3d_y", "rotation_3d_z"))
    fovs, keeps = sched("fov_schedule", "0:(90)"), sched("strength_schedule", "0:(0.65)")
    if a.strength is not None:
        keeps[:] = a.strength
    noises, contrasts = sched("noise_schedule", "0:(0.0)"), sched("contrast_schedule", "0:(1.0)")
    if a.noise is not None:
        noises[:] = a.noise
    cadence = max(1, int(a.cadence or float(settings.get("diffusion_cadence", 1))))
    use_color = a.color > 0 and str(settings.get("color_coherence", "")).lower() not in ("none", "", "false")
    pk = prompt_keys(a, settings)
    W, H = a.W // 32 * 32, a.H // 32 * 32

    out = Path(a.out or f"out_sana_deforum/{time.strftime('%Y%m%d-%H%M%S')}")
    (out / "frames").mkdir(parents=True, exist_ok=True)
    run = {"device": dev, "dtype": str(dtype), "W": W, "H": H, "frames": N, "fps": a.fps, "cadence": cadence,
           "strength_mode": a.strength_mode, "model": "1.6B" if a.big else "0.6B", "prompts": pk,
           "torch": torch.__version__}
    print(json.dumps(run, ensure_ascii=False, indent=1))

    T = {"load": 0.0, "encode": 0.0, "depth": 0.0, "warp": 0.0, "diffuse": 0.0, "color": 0.0}
    t0 = time.time()
    Sana.binning = a.binning
    sana, depth = Sana(a.big, dev, dtype), Depth(dev)
    warp = Warper(W, H, dev)
    sync(dev); T["load"] = time.time() - t0

    t0 = time.time()
    for _, p in pk:
        sana.embeds(p)
    sana.free_text_encoder()  # tous les prompts sont connus : on libère Gemma-2-2B (~5 Go)
    sync(dev); T["encode"] = time.time() - t0

    def prompt_at(f):
        i = max(k for k in range(len(pk)) if pk[k][0] <= f)
        cur = pk[i][1]
        if a.prompt_blend > 0 and i + 1 < len(pk) and f > pk[i + 1][0] - a.prompt_blend:
            return cur, pk[i + 1][1], 1 - (pk[i + 1][0] - f) / a.prompt_blend
        return cur, None, 0.0

    gen = torch.Generator("cpu").manual_seed(a.seed)
    if a.init_image:
        im = Image.open(a.init_image).convert("RGB").resize((W, H), Image.LANCZOS)
        img = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None].to(dev)
    else:
        img = sana.text2img(sana.embeds(pk[0][1]), W, H, gen).float().clamp(0, 1)
    ref_lab = None
    if use_color:
        lab0 = to_lab(img[0].permute(1, 2, 0).float().cpu().numpy())
        ref_lab = (lab0.mean((0, 1)), lab0.std((0, 1)))
    Image.fromarray((img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(out / "frames" / "00000.png")

    effs, t_start = [], time.time()
    for f in range(1, N):
        t1 = time.time()
        d = depth(img, W, H, dev)
        sync(dev); T["depth"] += time.time() - (t1)
        t1 = time.time()
        R = rot(math.radians(rxs[f]), math.radians(rys[f]), math.radians(rzs[f]), dev)
        img = warp(img.float(), d, tzs[f] / 200 * a.z_scale, R, float(fovs[f]))
        sync(dev); T["warp"] += time.time() - t1
        if f % cadence == 0:
            t1 = time.time()
            x = ((img - 0.5) * contrasts[f] + 0.5).clamp(0, 1)
            if noises[f] > 0:
                x = (x + torch.randn(x.shape, generator=gen).to(dev) * noises[f]).clamp(0, 1)
            pa, pb, w = prompt_at(f)
            x, eff = sana.img2img(x.to(dtype), sana.mixed(pa, pb, w), float(keeps[f]), W, H, gen, a.strength_mode, a.steps)
            img = x.float().clamp(0, 1)
            effs.append(eff)
            sync(dev); T["diffuse"] += time.time() - t1
            t1 = time.time()
            img = match_lab(img, ref_lab, a.color) if use_color else img
            T["color"] += time.time() - t1
        Image.fromarray((img[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)).save(out / "frames" / f"{f:05d}.png")
        if f % max(1, N // 10) == 0 or f == N - 1:
            el = time.time() - t_start
            print(f"image {f}/{N - 1} : {el:.0f} s écoulées, ~{el / f * (N - 1 - f):.0f} s restantes", flush=True)

    n_diff = max(1, len(effs))
    timings = {k: round(v, 2) for k, v in T.items()} | {
        "loop_total_s": round(time.time() - t_start, 1), "diffusions": len(effs),
        "s_per_diffusion": round(T["diffuse"] / n_diff, 2), "s_per_frame": round((time.time() - t_start) / max(1, N - 1), 2),
        "effective_denoise": sorted(set(round(e, 3) for e in effs)),
        "peak_mem_gb": round((torch.cuda.max_memory_allocated() if dev == "cuda" else
                              torch.mps.driver_allocated_memory() if dev == "mps" else 0) / 1e9, 1)}
    (out / "timings.json").write_text(json.dumps(timings, indent=1))
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1))
    print(json.dumps(timings, indent=1))

    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(out / "frames" / "%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out / "video.mp4")], check=False)
        print(out / "video.mp4")
    else:
        print("ffmpeg absent : images dans", out / "frames")


if __name__ == "__main__":
    main()
