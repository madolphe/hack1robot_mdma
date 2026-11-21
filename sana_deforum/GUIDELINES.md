# Consignes : tester la boucle Deforum + Sana-Sprint img2img

Ce fichier s'adresse à l'agent qui travaille sur l'autre machine.

## Le but

Projet hack1robo : produire des vidéos où l'on a l'impression de **s'enfoncer dans l'image**, comme dans Deforum (mode 3D). Le mécanisme visé est une **boucle de rétroaction** :
1. on avance la caméra dans l'image courante (profondeur, translation z, rotations, FOV) ;
2. on repasse l'image en img2img pour qu'elle se répare et invente du détail neuf.

Ce n'est pas une interpolation entre images fixes. Cette approche a été essayée sur Mac et rejetée : elle ne donne pas la sensation de s'enfoncer.

Ta mission : **faire tourner `sana_deforum_test.py`, mesurer, régler, et rapporter.** Le Mac de départ (M1, 16 Go) est trop lent pour cette boucle. Cette machine sert à savoir si l'approche donne le bon rendu, et à quel coût en temps.

## Fichiers

- `sana_deforum/sana_deforum_test.py` : script autonome, qui ne dépend pas du reste du dépôt.
- `FovVid1CompSettings.txt` (racine du dépôt) : réglage Deforum de référence. Le script lit `translation_z`, `rotation_3d_x/y/z`, `fov_schedule`, `strength_schedule`, `noise_schedule`, `contrast_schedule`, `diffusion_cadence` et `color_coherence`. Les autres clés (sampler, steps, cfg…) ne s'appliquent pas à Sana-Sprint.

## Installation

Python 3.12 recommandé. Sur Linux avec GPU NVIDIA, installe torch depuis l'index CUDA qui correspond au pilote.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch diffusers transformers accelerate safetensors pillow numpy scipy   # diffusers >= 0.43
# ffmpeg : paquet système (apt install ffmpeg / brew install ffmpeg)
```

Au premier lancement, environ 10 Go de poids sont téléchargés depuis Hugging Face, sans compte ni licence à accepter :
- Sana-Sprint 1.6B : encodeur de texte Gemma-2-2B, VAE DC-AE, transformer ;
- le transformer 0.6B seul (option par défaut) ;
- Depth Anything V2 Small.

## Commandes

Depuis `sana_deforum/` :

```bash
# 1. Test de fumée : 6 images, toute petite taille (vérifie que tout s'exécute)
python sana_deforum_test.py --settings ../FovVid1CompSettings.txt --frames 6 --W 512 --H 288 \
  --prompt "A surreal dreamlike oil painting, a visual metaphor of silence, memory, ocean"

# 2. Premier vrai rendu : 5 s à 15 i/s
python sana_deforum_test.py --settings ../FovVid1CompSettings.txt --frames 75 --W 768 --H 432 \
  --prompt "A surreal dreamlike oil painting, a visual metaphor of silence, memory, ocean"

# 3. Changement de prompt en cours de route (transition par mélange d'embeddings sur --prompt-blend images)
echo '{"0": "A surreal dreamlike oil painting, a visual metaphor of strobe, puppet, vibrations", "60": "A surreal dreamlike oil painting, a visual metaphor of silence, stillness, emptiness"}' > prompts.json
python sana_deforum_test.py --settings ../FovVid1CompSettings.txt --frames 120 --prompts-json prompts.json
```

Les sorties vont dans `out_sana_deforum/<horodatage>/` :
- `frames/` : une image PNG par image vidéo ;
- `video.mp4` ;
- `timings.json` : temps par étape, secondes par diffusion, force effective, pic mémoire ;
- `run.json` : les paramètres du run.

## Réglages à explorer, dans cet ordre

1. **`--strength`** : part de l'image **gardée** à chaque diffusion, avec la même convention que Deforum. La référence est 0,675. Essaie 0,55, 0,65 et 0,75. Trop haut, l'image ne se renouvelle pas et devient floue en avançant. Trop bas, elle clignote et perd toute continuité.
2. **`--cadence`** : diffusion toutes les N images (référence : 2). Une cadence de 1 est plus fluide mais coûte deux fois plus. Avec 3 ou 4, l'image risque de se dégrader entre deux diffusions.
3. **`--z-scale`** : vitesse d'avance (1,0 = valeurs du réglage ÷ 200, comme Deforum). Essaie 0,5, 1 et 2.
4. **`--strength-mode`** : `angle` (défaut) injecte le bruit à l'angle θ = (1 − strength)·π/2, ce qui donne une force continue avec 2 pas. `steps` utilise le mécanisme standard de diffusers, arrondi par paliers de 1/`--steps`. Compare-les à strength égale.
5. **`--big`** : transformer 1.6B au lieu de 0.6B (plus lent, peut-être plus net).
6. **`--color`** : 0 à 1, dosage de la correction des couleurs sur l'image 0. Mets 0 si les couleurs se figent trop.

## Ce qu'il faut rapporter

Pour chaque run, au minimum :
- **Machine :** GPU, VRAM, OS, version de torch.
- **Temps** (depuis `timings.json`) : `s_per_diffusion`, `s_per_frame`, `load`, `encode`, `peak_mem_gb`, et le temps total pour 5 s de vidéo.
- **Paramètres :** strength, cadence, z-scale, mode, modèle, taille.
- **Planche d'images :** une image toutes les 10 images vidéo, par exemple `ffmpeg -i video.mp4 -vf "select='not(mod(n\,10))',scale=320:-1,tile=6x3" -frames:v 1 planche.png`.
- **Ton jugement :** la sensation d'avancer dans l'image est-elle là ? Le détail se renouvelle-t-il ? Est-ce stable, ou ça clignote ? Les couleurs dérivent-elles ? Les bords en miroir se voient-ils ?
- **Ta recommandation :** le meilleur réglage trouvé, et le temps estimé pour 20 s et 60 s de vidéo en 768×432 et en 1024×576.

## Pièges connus

- **Téléchargement bloqué à 0 octet :** c'est le client Xet de Hugging Face. Le script force `HF_HUB_DISABLE_XET=1`. Garde-le si tu télécharges autrement.
- **N'utilise pas le fp16** pour Sana-Sprint : sur Mac, le 1.6B y donne une image noire. Le script prend du bf16 sur GPU et du fp32 sur CPU.
- **Mémoire :** l'encodeur de texte (~5 Go) est libéré une fois tous les prompts encodés. On ne peut donc pas ajouter de prompt pendant la boucle : il faut tout passer au départ.
- **Dimensions :** W et H sont arrondis au multiple de 32 inférieur (432 devient 416).
- **Le mode `angle` est une extrapolation.** Sana-Sprint est distillé pour partir de θ = π/2 (bruit pur). Démarrer plus bas devrait marcher, puisque c'est ce que fait l'img2img de diffusers, mais ce n'est validé que sur un test de 6 images. Si les images sortent délavées ou grises, compare avec `--strength-mode steps --steps 4`.
- **Mesures sur le Mac M1 16 Go**, à titre de repère, modèle 0.6B en 512×288, cadence 2 : **85 s par diffusion** avec l'encodeur de texte encore chargé (pic à 12,6 Go). L'effet de sa libération reste à confirmer. Sur un GPU, on s'attend à quelques secondes, voire moins d'une seconde, par diffusion.

## À ne pas faire

- Ne commite ni les poids, ni `out_sana_deforum/`, ni aucune clé ou jeton.
- Ne modifie pas `FovVid1CompSettings.txt`. Pour d'autres valeurs, copie-le sous un autre nom.
- Ne pousse pas sur `main`. La branche de travail est `visuels`.
