# hack1robo : les mots intérieurs d'un LLM

Ce projet artistique produit une vidéo des « pensées » d'un modèle de langage pendant qu'il disserte sur une question. À chaque token généré, une **Jacobian lens** lit les mots latents des couches intermédiaires de Gemma 3 12B. Ces mots deviennent ensuite des images (Z-Image-Turbo), une voix (Kokoro), un son (Stable Audio Open, synthèse paramétrique, effets pedalboard) et un montage (ffmpeg). Une « drogue » optionnelle oriente les activations du modèle vers un champ lexical qui évolue par phases. Une interface web Gradio protégée par mot de passe permet de lancer des vidéos depuis un téléphone.

Pour aller plus loin : `SPEC.md` et `SPEC_V2.md` (cahiers des charges), `DECISIONS.md` (journal des choix), `MORNING_REPORT.md` (prompt système, mécanisme de la drogue, calibration, mesures) et `PROGRESS.md`.

Les identités synthétiques et les sources des cinq substances principales sont regroupées dans [`PROFILS_SUBSTANCES.md`](PROFILS_SUBSTANCES.md).

---

## 1. Modèles utilisés

Tous les poids passent par le cache Hugging Face du projet, `weights/hub` (environ 80 Go). Ils ne sont pas inclus dans l'archive.

| Rôle | Dépôt Hugging Face | Détail | Licence / accès |
| --- | --- | --- | --- |
| LLM qui « pense » | `google/gemma-3-12b-it` | `Gemma3ForConditionalGeneration`, bf16, 48 couches, ~24 Go de VRAM | Licence Gemma, **accès à accepter sur HF** |
| Lecture des mots latents | `neuronpedia/jacobian-lens` | fichier `gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt` | public |
| Images | `Tongyi-MAI/Z-Image-Turbo` | `ZImagePipeline` (t2i) + `ZImageImg2ImgPipeline` (i2i, composants partagés), bf16, 1920×1080 | Apache-2.0 |
| Voix | `hexgrad/Kokoro-82M` | voix `ff_siwis` (fr), `af_heart`, `am_michael`, `bm_george` (en) en multi-voix | Apache-2.0 |
| Nappe sonore | `stabilityai/stable-audio-open-1.0` | `StableAudioPipeline`, fp16 | Stability Community License, **accès à accepter sur HF** |
| Traits sémantiques des mots | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | projection sur des axes d'ancrage | Apache-2.0 |

Bibliothèque de lentille : [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) (paquet `jlens`, commit `581d398`), clonée dans `vendor/jacobian-lens` puis installée en mode éditable.

Lexiques, placés dans `weights/lexicons/` et absents de l'archive :

| Lexique | Source | Fichier attendu |
| --- | --- | --- |
| NRC VAD (anglais + français) | saifmohammad.com (NRC-VAD-Lexicon) | `weights/lexicons/nrc/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt` et `.../OneFilePerLanguage/French-NRC-VAD-Lexicon.txt` |
| Lancaster Sensorimotor Norms | OSF `48wsc` | `weights/lexicons/lancaster.csv` |
| Concrétude, Brysbaert et al. | GitHub ArtsEngine | `weights/lexicons/Concreteness_ratings_Brysbaert_et_al_BRM.txt` |

Si un lexique manque, les traits se replient sur les embeddings seuls.

**Le code ne passe jamais `trust_remote_code=True`.**

---

## 2. Matériel et système

- GPU NVIDIA avec **48 Go de VRAM** (testé sur RTX A6000). Un seul modèle lourd occupe le GPU à la fois : chaque étape tourne dans son propre processus, et un verrou `runs/.gpu.lock` sérialise les chaînes.
- Linux, Python 3.12, CUDA 13.2 (torch 2.15.0+cu132).
- Paquets système : `ffmpeg` (testé en 8.1) et `espeak-ng` (phonétisation de Kokoro).
- Environ 100 Go de disque pour les poids et les runs.

---

## 3. Installation

```bash
git clone https://github.com/anthropics/jacobian-lens vendor/jacobian-lens
git -C vendor/jacobian-lens checkout 581d398

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt   # torch : choisir l'index CUDA adapté
uv pip install --python .venv/bin/python -e vendor/jacobian-lens
uv pip install --python .venv/bin/python \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Poids : accepter d'abord les licences de Gemma 3 et de Stable Audio Open sur huggingface.co
hf auth login                   # jeton en lecture seule, stocké dans ~/.cache/huggingface (jamais dans le projet)
./run.sh script scripts/download_weights.sh    # remplit weights/hub, journal dans runs/setup/
# Lexiques : à télécharger à la main dans weights/lexicons/ (voir le tableau ci-dessus)
```

`run.sh` force ensuite `HF_HUB_OFFLINE=1` : une fois les poids présents, plus rien n'est téléchargé.

Vérifications :

```bash
./run.sh test            # tests rapides sur CPU
./run.sh validate        # la lecture J-lens donne-t-elle des mots plausibles ?
./run.sh smoke image     # puis voice, audio
```

---

## 4. Utilisation

### En ligne de commande

```bash
./run.sh all --config configs/default.yaml                 # chaîne complète
./run.sh all --config runs/<id>/request.yaml               # requête écrite par l'interface (réutilise runs/<id>)
./run.sh all --config c.yaml --modes voix sonorisation mixte
```

Étapes séparées, toutes avec `--run runs/<id>` : `record`, `render`, `voice`, `sound`, `assemble [--mode voix|sonorisation|mixte]`.

Une config hérite de `configs/default.yaml` et n'écrit que ce qu'elle change. Exemple de requête v2 :

```yaml
input: {phrase: "Une machine peut-elle rêver ?"}
environment: foret          # clé de configs/environments.yaml, ou absent
drug: {name: datura, intensity: 1.0}   # clé de configs/drugs.yaml, ou "aucune"
voice: {n_voices: 2}        # 1 à 4
sound: {mode: mixte}        # voix | sonorisation | mixte
video: {target_duration_s: 60}
```

Résultats dans `runs/<id>/` : `trace.json` (mots par couche et par token), `windows.json`, `frames/`, `audio/`, `out_<mode>.mp4`, `out.mp4`, `config.yaml` (configuration résolue), `log.txt` et `timings.json`. Le rendu prend environ 7 min par minute de vidéo sur une A6000.

### Interface web

```bash
./run.sh web-install   # service systemd utilisateur + timer de renouvellement 48 h + linger (sans sudo)
./run.sh web-status    # état et lien public courant
./run.sh web-restart
```

- Lien public `*.gradio.live`, recopié dans `~/pCloudDrive/Transfert/hack1robo/lien_web.txt`.
- Identifiant `prise`. Le mot de passe est généré au premier lancement dans `~/.config/hack1robo/web_auth` (droits 600). Il n'est jamais versionné ni écrit dans un rapport.
- Onglet *Lancer* : question, environnement, drogue (ou « sans drogue »), intensité, durée, 1 à 4 voix, mode son. Les images s'affichent pendant le rendu.
- Onglet *Galerie* : toutes les vidéos de `runs/`, avec leurs paramètres.
- Sécurité : la question n'entre jamais dans une commande shell (elle est écrite par `yaml.safe_dump` et le sous-processus reçoit une liste d'arguments, sans `shell=True`). Les listes sont validées contre les catalogues. Seul `runs/` est servi. La file est limitée à 5 et l'API est fermée. Tests : `./run.sh py webtest`.

---

## 5. Fonctionnement

1. **Génération** (`record.py`). Gemma reçoit le prompt système *« Tu dissertes longuement et librement sur le sujet donné. »* puis la question (température 0.8, top-p 0.95, graine fixe : rendu déterministe). `max_new_tokens` est calculé à partir de la durée visée, et la fin de séquence est bloquée.
2. **Lecture J-lens** (`lensmodel.py`). Pour chaque couche l de la bande 0.38–0.92 (couches 19 à 44) : `logits_l = unembed(h_l @ J[l]ᵀ)`, où unembed vaut RMSNorm Gemma ×(1+w), puis lm_head et softcap, en fp32. Le top-50 est filtré (mots entiers, mots vides retirés, échos des 32 derniers tokens exclus).
3. **Fenêtres** (`windows.py`). Des fenêtres glissantes de tokens donnent une tranche d'environ 3 s chacune, avec les mots dominants et un prompt d'image *« A surreal dreamlike oil painting, a visual metaphor of {mots}. {style} »*.
4. **Images** (`render.py`). Z-Image t2i, puis i2i (strength 0.7) d'une tranche à l'autre, avec une remise à zéro toutes les 8 tranches.
5. **Voix** (`voice.py`). Kokoro prononce les nouveaux mots. Avec N voix, chaque voix est décalée de k·d/N.
6. **Son** (`features.py`, `sound.py`). Traits par mot (valence, éveil, concrétude, sensorialité) qui pilotent les effets de voix pedalboard et la nappe (Stable Audio en couche A, synthèse en couche B, ducking −6 dB).
7. **Montage** (`assemble.py`). Fondus de 0,8 s, encodage libx264 en 1080p30, durées étirées vers `target_duration_s`.

**Drogue** (`drugs.py`, `configs/drugs.yaml`). Le moteur ajoute `h += α·‖h‖·v` sur les couches 22 à 33, avec `v = J[l]ᵀ(g ⊙ W_U[mot])` normalisé. Toutes les 12 étapes, 3 mots sont tirés du champ lexical de la phase courante. Les phases suivent t/T. α vaut niveau × intensité, plafonné à 0.006, avec des transitions lissées. Le catalogue compte 11 drogues : alcool, cannabis, cocaïne, amphétamines, MDMA, LSD, champignons, DMT, kétamine, héroïne, datura. Le α=4 du cahier des charges détruisait le texte ; la calibration est détaillée dans `MORNING_REPORT.md`.

**Environnements** (`configs/environments.yaml`) : rave, mariage, enterrement, open space, forêt, urgences, plage, garde à vue, église, chambre. Chacun ajoute un contexte au prompt système, un style d'image et une texture sonore.

---

## 6. Arborescence

```
run.sh                 point d'entrée unique (toutes les étapes GPU passent par lui)
introspect/            code Python (cli, record, lensmodel, drugs, windows, render, voice,
                       features, lexicons, sound, assemble, webapp, webtest, validate, smoke,
                       sweep, rewindow, record_run, common, stopwords)
configs/               default.yaml, drugs.yaml, environments.yaml, variants/, tests/
scripts/               téléchargement des poids, lots de rendus, recette, unités systemd
requirements.txt       versions testées
vendor/                jacobian-lens (à cloner, non inclus)
weights/               poids et lexiques (non inclus)
runs/                  sorties (non incluses)
```
