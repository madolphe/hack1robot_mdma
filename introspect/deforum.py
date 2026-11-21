"""Vidéo « Deforum » d'un run : on s'enfonce dans l'image, et SANA-Sprint la répare et l'invente à chaque pas.

Méthode de la branche visuels (sana_deforum/sana_deforum_test.py, consignes dans sana_deforum/GUIDELINES.md),
calée sur les tranches du run : le prompt d'image de chaque tranche devient une image clé de prompt au moment où
la tranche commence (fondu des embeddings sur --prompt-blend images), et le son du montage est ajouté.
À chaque image vidéo : profondeur (Depth Anything V2 Small), déplacement 3D de la caméra (translation_z,
rotation_3d_*, fov du réglage Deforum) et reprojection de l'image précédente ; toutes les `cadence` images,
contraste, bruit, img2img SANA-Sprint (force = part gardée) et couleurs ramenées sur l'image 0.

./run.sh py deforum --run runs/<id> [--cadence 3] [--fps 15] [--strength 0.22] [--no-binning] [--small]
-> runs/<id>/out_deforum.mp4 (1920x1080, son du montage), deforum/NNNNN.png, deforum.json (réglages et temps)"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .assemble import durations, ffmpeg_bin
from .common import ROOT, Log, run_config
from .render import letterboxed
from .windows import make_prompt

sys.path.insert(0, str(ROOT / "sana_deforum"))
from sana_deforum_test import DEFAULT_SETTINGS, Depth, Sana, Warper, parse_schedule, rot, sync  # noqa: E402

_M_XYZ = torch.tensor([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_M_RGB = torch.tensor([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])


def to_lab_t(x: torch.Tensor) -> torch.Tensor:
    """sana_deforum_test.to_lab sur le GPU : x [H,W,3] sRGB 0..1 -> L*a*b* (mêmes formules, ~50 fois plus rapide)."""
    x = torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    xyz = x @ _M_XYZ.to(x).T / _WHITE.to(x)
    f = torch.where(xyz > 0.008856, xyz.clamp_min(0) ** (1 / 3), 7.787 * xyz + 16 / 116)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def from_lab_t(lab: torch.Tensor) -> torch.Tensor:
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    inv = lambda t: torch.where(t ** 3 > 0.008856, t ** 3, (t - 16 / 116) / 7.787)
    xyz = torch.stack([inv(fx), inv(fy), inv(fz)], -1) * _WHITE.to(lab)
    x = (xyz @ _M_RGB.to(lab).T).clamp(0, 1)
    return torch.where(x > 0.0031308, 1.055 * x ** (1 / 2.4) - 0.055, 12.92 * x)


class DirectImg2Img:
    """Le img2img SANA-Sprint du mode `angle` (sana_deforum_test.Sana.img2img), appelé sans le pipeline diffusers.

    Mêmes calculs que SanaSprintImg2ImgPipeline (bruit SCM à l'angle θ = (1 - keep)·π/2, deux pas TrigFlow θ -> 0,83·θ -> 0,
    mêmes tirages aléatoires), sans ses contrôles, conversions et ré-échantillonnages à chaque appel : le pipeline passe
    environ 360 ms par appel hors modèle. Le pipeline travaille à la résolution native la plus proche (1024×576 ->
    1344×768, « resolution binning ») puis revient à la taille demandée ; `binning` reproduit ce détour, et la boucle
    peut aussi tourner directement à 1344×768 pour s'en passer."""

    def __init__(self, sana, W: int, H: int, binning: bool = True, guidance: float = 4.5):
        from diffusers.image_processor import PixArtImageProcessor
        from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_1024_BIN

        p = sana.i2i
        self.vae, self.tr = p.vae, p.transformer
        self.sd = float(p.scheduler.config.sigma_data)
        self.sf = float(p.vae.config.scaling_factor)
        self.W, self.H = W, H
        self.Hb, self.Wb = (PixArtImageProcessor.classify_height_width_bin(H, W, ratios=ASPECT_RATIO_1024_BIN)
                            if binning else (H, W))
        self.Hb, self.Wb = int(self.Hb), int(self.Wb)
        self.resize = PixArtImageProcessor.resize_and_crop_tensor
        self.guidance = torch.full([1], guidance, device=sana.dev, dtype=torch.float32)
        self.gscale = float(self.tr.config.guidance_embeds_scale)
        self.dev = sana.dev

    @torch.no_grad()
    def __call__(self, img_t, emb, keep: float, gen):
        """img_t : [1,3,H,W] dans [0,1] -> même sortie que Sana.img2img(…, mode "angle")."""
        from diffusers.utils.torch_utils import randn_tensor

        e, m = emb
        theta = float(np.clip(1.0 - keep, 0.02, 1.0)) * math.pi / 2
        ts = torch.tensor([theta, theta * (1.3 / 1.5708), 0.0], device=self.dev)
        x = img_t
        if x.shape[2] != self.Hb or x.shape[3] != self.Wb:
            x = torch.nn.functional.interpolate(x, size=(self.Hb, self.Wb), mode="bilinear", align_corners=False)
        x = (2 * x - 1).to(self.vae.dtype)
        lat0 = self.vae.encode(x).latent * self.sf * self.sd
        shape = lat0.shape
        noise = randn_tensor(shape, generator=gen, device=self.dev, dtype=torch.float32) * self.sd
        lat = torch.cos(ts[:1]) * lat0 + torch.sin(ts[:1]) * noise
        g = self.guidance.to(e.dtype) * self.gscale
        x0 = lat
        for i in range(2):
            s, t = ts[i], ts[i + 1]
            scm = (torch.sin(s) / (torch.cos(s) + torch.sin(s))).expand(1)
            k = scm.view(-1, 1, 1, 1)
            norm = torch.sqrt(k ** 2 + (1 - k) ** 2)
            inp = lat / self.sd * norm
            pred = self.tr(inp.to(self.tr.dtype), encoder_hidden_states=e.to(self.tr.dtype), encoder_attention_mask=m,
                           guidance=g, timestep=scm, return_dict=False)[0]
            pred = ((1 - 2 * k) * inp + (1 - 2 * k + 2 * k ** 2) * pred) / norm
            pred = pred.float() * self.sd
            x0 = torch.cos(s) * lat - torch.sin(s) * pred
            fresh = randn_tensor(shape, generator=gen, device=self.dev) * self.sd  # tiré comme le planificateur SCM
            lat = torch.cos(t) * x0 + torch.sin(t) * fresh
        out = self.vae.decode((x0 / self.sd).to(self.vae.dtype) / self.sf, return_dict=False)[0]
        if out.shape[2] != self.H or out.shape[3] != self.W:
            out = self.resize(out, self.W, self.H)
        return (out / 2 + 0.5).clamp(0, 1)


def match_lab_t(img_t: torch.Tensor, ref: tuple, w: float) -> torch.Tensor:
    """sana_deforum_test.match_lab sur le GPU : moyenne et écart-type L*a*b* ramenés vers la référence, dosés par w."""
    lab = to_lab_t(img_t[0].permute(1, 2, 0).float())
    m = (lab - lab.mean((0, 1))) / (lab.std((0, 1)) + 1e-6) * ref[1] + ref[0]
    return from_lab_t(lab * (1 - w) + m * w).permute(2, 0, 1)[None]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--settings", default=str(ROOT / "FovVid1CompSettings.txt"), help="réglage Deforum (JSON)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--W", type=int, default=1024)
    ap.add_argument("--H", type=int, default=576)
    ap.add_argument("--cadence", type=int, default=3,
                    help="diffusion toutes les N images (2 dans le réglage de référence ; 3 : 32 %% plus rapide, mêmes sauts à chaque diffusion, 5 par seconde au lieu de 7,5)")
    ap.add_argument("--strength", type=float, default=0.22,
                    help="part de l'image gardée à chaque diffusion (0.675 dans le réglage de référence : les prompts n'y passent pas)")
    ap.add_argument("--z-scale", type=float, default=1.0, help="vitesse d'avance (1 = translation_z / 200, comme Deforum)")
    ap.add_argument("--color", type=float, default=0.2, help="dosage du retour aux couleurs de référence (0 = libre)")
    ap.add_argument("--color-ref", choices=["slice", "first"], default="slice",
                    help="couleurs de référence : slice = l'image fixe de la tranche en cours (frames/, passe render), "
                         "fondue comme les prompts ; first = l'image 0 (réglage d'origine : toute la vidéo garde sa palette)")
    ap.add_argument("--prompt-blend", type=int, default=12, help="images de fondu entre les prompts de deux tranches (à 15 i/s)")
    ap.add_argument("--small", action="store_true", help="transformer 0.6B au lieu de 1.6B (un peu plus rapide, moins net)")
    ap.add_argument("--motion-fps", type=float, default=15.0,
                    help="cadence pour laquelle les courbes Deforum sont écrites : à un autre --fps, le mouvement par "
                         "seconde reste le même")
    ap.add_argument("--strength-mode", choices=["angle", "steps"], default="angle",
                    help="angle : bruit injecté à θ = (1 - strength)·π/2, 2 pas ; steps : img2img diffusers standard")
    ap.add_argument("--steps", type=int, default=4, help="pas de débruitage en mode steps")
    ap.add_argument("--seconds", type=float, default=None, help="ne rendre que le début (aperçu)")
    ap.add_argument("--audio", default="mixte", help="piste du montage à ajouter : voix, sonorisation, mixte (audio_<mode>.wav)")
    ap.add_argument("--tag", default="", help="suffixe des sorties, pour comparer des réglages (out_deforum_<tag>.mp4)")
    ap.add_argument("--depth-every", choices=["frame", "diffusion"], default="diffusion",
                    help="profondeur recalculée à chaque image, ou seulement quand une diffusion a renouvelé l'image "
                         "(entre deux, la caméra bouge à peine : on garde la carte précédente ; moitié moins d'appels)")
    ap.add_argument("--no-binning", action="store_true",
                    help="SANA à la taille de la boucle (1024x576) au lieu de sa taille native la plus proche (1344x768) : "
                         "~40 %% plus rapide, rendu différent")
    a = ap.parse_args(argv)
    sfx = f"_{a.tag}" if a.tag else ""

    run_dir = Path(a.run)
    cfg = run_config(run_dir)
    log = Log(run_dir, "deforum")
    dev, dtype = "cuda", torch.bfloat16  # pas de fp16 : image noire avec le 1.6B (sana_deforum/GUIDELINES.md)
    wins = json.loads((run_dir / "windows.json").read_text(encoding="utf-8"))
    durs = durations(run_dir, cfg, len(wins))
    total = sum(durs) + cfg["video"]["crossfade_s"]
    if a.seconds:
        total = min(total, a.seconds)
    N = int(round(total * a.fps))
    starts = np.cumsum([0.0] + durs)[:-1]
    # une image clé de prompt par tranche, à l'image vidéo où la tranche commence
    keys = sorted({int(round(s * a.fps)): k for k, s in enumerate(starts) if round(s * a.fps) < N}.items())
    pk = [(f0, make_prompt([x for x, _ in wins[k]["words"]], cfg, wins[k].get("state"))) for f0, k in keys]

    settings = DEFAULT_SETTINGS | json.loads(Path(a.settings).read_text())
    # les courbes sont écrites image par image pour --motion-fps : on les lit au même instant, et les déplacements
    # par image sont mis à l'échelle, pour que la caméra aille à la même vitesse par seconde quel que soit --fps
    ratio = a.motion_fps / a.fps
    n_ref = int(math.ceil(N * ratio)) + 2

    def sched(k, default):
        ref = parse_schedule(settings.get(k, default), n_ref)
        return np.interp(np.arange(N) * ratio, np.arange(n_ref), ref)

    tzs, rxs, rys, rzs = (sched(k, "0:(0)") * ratio for k in ("translation_z", "rotation_3d_x", "rotation_3d_y", "rotation_3d_z"))
    fovs, keeps = sched("fov_schedule", "0:(90)"), sched("strength_schedule", "0:(0.65)")
    if a.strength is not None:
        keeps[:] = a.strength
    noises, contrasts = sched("noise_schedule", "0:(0.0)"), sched("contrast_schedule", "0:(1.0)")
    cadence = max(1, int(a.cadence or float(settings.get("diffusion_cadence", 1))))  # 0 : celle du réglage
    W, H = a.W // 32 * 32, a.H // 32 * 32
    params = {"frames": N, "fps": a.fps, "W": W, "H": H, "cadence": cadence, "strength": round(float(keeps.mean()), 3),
              "z_scale": a.z_scale, "color": a.color, "color_ref": a.color_ref, "model": "0.6B" if a.small else "1.6B", "prompts": len(pk),
              "strength_mode": a.strength_mode, "depth_every": a.depth_every, "binning": not a.no_binning,
              "settings": Path(a.settings).name}
    log("réglages", json.dumps(params, ensure_ascii=False))

    T = {k: 0.0 for k in ("load", "encode", "depth", "warp", "diffuse", "color")}
    t0 = time.time()
    sana, depth, warp = Sana(not a.small, dev, dtype), Depth(dev), Warper(W, H, dev)
    direct = DirectImg2Img(sana, W, H, binning=not a.no_binning)  # mêmes images que Sana.img2img avec le binning
    sync(dev); T["load"] = time.time() - t0
    t0 = time.time()
    for _, p in pk:
        sana.embeds(p)
    sana.free_text_encoder()  # tous les prompts sont encodés : Gemma-2-2B (~5 Go) n'est plus utile
    sync(dev); T["encode"] = time.time() - t0
    log(f"{len(pk)} prompts encodés, modèles chargés ({T['load'] + T['encode']:.0f} s)")

    blend = max(0, round(a.prompt_blend / ratio))  # même durée de fondu en secondes à toute cadence

    def key_at(f):
        """Image clé en cours, la suivante pendant le fondu (sinon None) et le poids de la suivante."""
        i = max(k for k in range(len(pk)) if pk[k][0] <= f)
        if blend > 0 and i + 1 < len(pk) and f > pk[i + 1][0] - blend:
            return i, i + 1, 1 - (pk[i + 1][0] - f) / blend
        return i, None, 0.0

    # palette de chaque tranche : celle de son image fixe, qui suit déjà l'état (image.state_styles) ; avec l'image 0
    # comme seule référence, une drogue en plusieurs phases garde toute la vidéo la palette de la montée
    refs = None
    stills = [run_dir / "frames" / f"{k:04d}.png" for _, k in keys]
    if a.color_ref == "slice" and all(p.exists() for p in stills):
        refs = []
        for p in stills:
            lab = to_lab_t(torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((W // 4, H // 4)),
                                                       dtype=np.float32) / 255).to(dev))
            refs.append((lab.mean((0, 1)), lab.std((0, 1))))
    elif a.color_ref == "slice":
        log("pas d'images fixes dans frames/ (passe render) : couleurs de l'image 0")

    out = run_dir / f"deforum{sfx}"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    seed = int(cfg["image"].get("seed", 1))
    for attempt in range(1 + int(cfg["image"].get("retry_letterbox", 3))):
        gen = torch.Generator("cpu").manual_seed(seed + 1000 * attempt)
        img = sana.text2img(sana.embeds(pk[0][1]), W, H, gen).float().clamp(0, 1)
        # toute la plongée part de cette image : des bandes noires y resteraient jusqu'au bout
        if not letterboxed(Image.fromarray((img[0].permute(1, 2, 0).float() * 255).round().byte().cpu().numpy())):
            break
        log(f"première image : bandes noires, nouvel essai ({attempt + 1})")
    lab0 = to_lab_t(img[0].permute(1, 2, 0).float())
    ref_lab = (lab0.mean((0, 1)), lab0.std((0, 1)))
    from concurrent.futures import ThreadPoolExecutor

    writer = ThreadPoolExecutor(max_workers=2)  # l'encodage PNG se fait pendant que le GPU calcule l'image suivante
    pending = []

    def save(f, x):
        arr = (x[0].permute(1, 2, 0).float() * 255).round().byte().cpu().numpy()
        pending.append(writer.submit(lambda: Image.fromarray(arr).save(out / f"{f:05d}.png", compress_level=1)))

    save(0, img)
    n_diff, t_loop, d, fresh = 0, time.time(), None, True
    for f in range(1, N):
        t1 = time.time()
        if d is None or fresh or a.depth_every == "frame":
            d = depth(img, W, H, dev)
        fresh = False
        sync(dev); T["depth"] += time.time() - t1
        t1 = time.time()
        R = rot(math.radians(rxs[f]), math.radians(rys[f]), math.radians(rzs[f]), dev)
        img = warp(img.float(), d, tzs[f] / 200 * a.z_scale, R, float(fovs[f]))
        sync(dev); T["warp"] += time.time() - t1
        if f % cadence == 0:
            t1 = time.time()
            x = ((img - 0.5) * contrasts[f] + 0.5).clamp(0, 1)
            if noises[f] > 0:
                x = (x + torch.randn(x.shape, generator=gen).to(dev) * noises[f]).clamp(0, 1)
            i, j, w = key_at(f)
            pa, pb = pk[i][1], pk[j][1] if j is not None else None
            if a.strength_mode == "angle":
                x = direct(x.to(dtype), sana.mixed(pa, pb, w), float(keeps[f]), gen)
            else:
                x, _ = sana.img2img(x.to(dtype), sana.mixed(pa, pb, w), float(keeps[f]), W, H, gen, a.strength_mode, a.steps)
            img = x.float().clamp(0, 1)
            n_diff += 1
            fresh = True  # image renouvelée : nouvelle carte de profondeur à l'image suivante
            sync(dev); T["diffuse"] += time.time() - t1
            t1 = time.time()
            ref = ref_lab if refs is None else refs[i] if j is None else tuple(
                (1 - w) * x + w * y for x, y in zip(refs[i], refs[j]))
            img = match_lab_t(img, ref, a.color) if a.color > 0 else img
            T["color"] += time.time() - t1
        save(f, img)
        if f % max(1, N // 10) == 0:
            el = time.time() - t_loop
            log(f"image {f}/{N - 1} : {el:.0f} s, ~{el / f * (N - 1 - f):.0f} s restantes")
    for p in pending:
        p.result()
    writer.shutdown()
    loop = time.time() - t_loop
    timings = {k: round(v, 1) for k, v in T.items()} | {
        "loop_s": round(loop, 1), "diffusions": n_diff, "s_per_diffusion": round(T["diffuse"] / max(1, n_diff), 3),
        "s_per_frame": round(loop / max(1, N - 1), 3), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    (run_dir / f"deforum{sfx}.json").write_text(json.dumps({"params": params, "timings": timings, "prompts": pk},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    log("temps", json.dumps(timings))
    del sana, depth, direct
    torch.cuda.empty_cache()

    vc = cfg["video"]
    wav = run_dir / f"audio_{a.audio}.wav"
    mp4 = run_dir / f"out_deforum{sfx}.mp4"
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(out / "%05d.png")]
    cmd += ["-i", str(wav)] if wav.exists() else []
    cmd += ["-vf", f"scale={vc.get('width', 1920)}:{vc.get('height', 1080)}:flags=lanczos", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"] if wav.exists() else []
    subprocess.run(cmd + [str(mp4)], check=True)
    log(f"{mp4.name} : {N} images à {a.fps} i/s, {N / a.fps:.1f} s" + ("" if wav.exists() else " (sans son)"))
    log.done()


if __name__ == "__main__":
    main()
