# hack1robo : les mots intérieurs d'un LLM

Ce projet artistique produit une vidéo des « pensées » d'un modèle de langage pendant qu'il disserte sur une question. À chaque token généré, une **Jacobian lens** lit les mots latents des couches intermédiaires de Gemma 3 12B. Ces mots deviennent ensuite des images (Z-Image-Turbo), une voix (Kokoro), un son (Stable Audio Open, synthèse paramétrique, effets pedalboard) et un montage (ffmpeg). Une « drogue » optionnelle oriente les activations du modèle vers un champ lexical qui évolue par phases. Une interface web Gradio protégée par mot de passe permet de lancer des vidéos depuis un téléphone.

Pour aller plus loin : `SPEC.md` et `SPEC_V2.md` (cahiers des charges), `DECISIONS.md` (journal des choix), `MORNING_REPORT.md` (prompt système, mécanisme de la drogue, calibration, mesures) et `PROGRESS.md`.

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

---

## 7. Petits modèles (GPU de 16 Go)

`configs/small.yaml` fait tourner toute la chaîne sur un GPU de 16 Go (testé sur RTX 4090 Laptop, torch 2.14 cu130) :

| Rôle | Grand modèle (défaut) | Petit modèle (`small.yaml`) |
| --- | --- | --- |
| LLM | Gemma 3 12B-it, 48 couches | `Qwen/Qwen3.5-4B`, 32 couches, ~9 Go |
| Lentille | J-lens Neuronpedia | **J++ lens** (Ayonrinde & Lindsey, 2026), `qwen3.5-4b/jpp/...` chez Neuronpedia |
| Images | Z-Image-Turbo, 8 étapes | SANA-Sprint 1.6B, 2 étapes (~1 s par image) ; FLUX.2 [klein] 4B en variante |
| Sonorisation A | Stable Audio Open 1.0, 100 étapes | **Stable Audio 3 small-sfx**, 8 étapes (~0,2 s par clip, 2 Go de VRAM), dans son propre environnement `.venv-audio` ; sans ses poids, seule la couche B (synthèse) reste |

```bash
# environnement audio à part : la bibliothèque stable-audio-3 (pas sur PyPI) exige torch 2.7.1
git clone https://github.com/Stability-AI/stable-audio-3 vendor/stable-audio-3
git -C vendor/stable-audio-3 checkout 790793b
uv venv --python 3.12 .venv-audio
uv pip install --python .venv-audio/bin/python --torch-backend=cu126 torch==2.7.1 torchaudio==2.7.1 -e vendor/stable-audio-3

./run.sh script scripts/download_weights_small.sh     # ~25 Go (+16 Go pour FLUX.2 [klein])
./run.sh test configs/small.yaml                      # tests CPU (tokenizer, drogue, tranches)
./run.sh smoke image audio --config configs/small.yaml
./run.sh validate configs/small.yaml                  # lecture J++ contre lens.apply
./run.sh all --config configs/variants/small_lsd.yaml --modes voix mixte
./run.sh all --config configs/variants/img_flux2_klein.yaml
```

**Rythme** : `small.yaml` vise 40 s avec une image toutes les ~1,5 s (`video.target_duration_s: 40`, `slice_target_s: 1.5`, `min_duration_s: 1.0`, `crossfade_s: 0.5`) et tient la durée (`video.fit_target: true` : si les voix et leurs queues de réverbération dépassent, les tranches sont resserrées et le son déborde sur l'image suivante). Sur la prise `lsd_etats` : 30 images de 1,0 à 2,1 s, 40,5 s, contre 20 images de 2 à 4,1 s et 61,8 s avec les réglages de `default.yaml`. Pour recouper une vidéo sans relancer le LLM : `./run.sh py rewindow --from runs/<id> --run-id <nouveau> --config <config>`, puis `render`, `voice`, `sound`, `assemble`.

**Son** (réglages de `small.yaml`, ceux de `default.yaml` gardent l'ancien comportement) :

- `sound.sono.segments: phase` : la nappe n'est plus faite d'un clip Stable Audio et d'un drone par image (toutes les ~1,3 s), mais d'un segment continu par phase de drogue (20 s au plus, fondus de 1,5 s, une seule normalisation ; sans drogue, des blocs de 20 s). Le prompt d'un segment réunit les mots intérieurs de ses tranches. Avant : la nappe baissait de 4,5 dB à chaque changement d'image, des clips presque muets étaient remontés jusqu'à ×12,5, et le drone ajoutait 2,3 salves de bruit par seconde.
- `sound.drug_fx: 0.35` : force des effets « drogués » (modulation en anneau et saturation de la nappe ; phaser, distorsion et vibrato de la voix). Ils étaient à 100 % sur toute la prise par états (α 0,06 = niveau 4) : un son métallique de panne.
- `sound.duck_smooth_s: 0.5` (la nappe baisse sous la voix sans pomper) et `sound.loudness_db: -17` (niveau moyen tenu par un limiteur, crêtes à -1,5 dBFS pour l'encodage AAC).

Une vidéo d'une minute prend environ **2 min** (record 20 s, images 30 s, voix 10 s, son 27 s dont 4 s pour les 20 clips Stable Audio 3, montage 25 s par mode). Avec Stable Audio Open 1.0 (`layer_a: {engine: diffusers, model: stabilityai/stable-audio-open-1.0, steps: 100}`), ses clips prennent 4 min à eux seuls. Sur A6000 avec les grands modèles : ~7 min.

Ce qui change dans le code :

- **Tokenizer** : Qwen est en BPE au niveau des octets (`Ġmot`), Gemma en SentencePiece (`▁mot`). `lensmodel.word_starts` et `single_token_id` gèrent les deux ; la trace garde le texte décodé de chaque token (`text`) pour le filtre des échos.
- **Gabarit** : `input.chat_template_kwargs: {enable_thinking: false}`, sinon Qwen3.5 ouvre un bloc `<think>`.
- **Cache** : Qwen3.5 mêle attention et couches linéaires à état ; le modèle crée lui-même son cache.
- **Images** : `image.backend` (`zimage`, `sana_sprint`, `flux2_klein`, déduit du nom du modèle). FLUX.2 [klein] ne rebruite pas l'image précédente : il la reçoit en référence, avec une consigne d'édition (`image.edit_prefix`). Mais avec des mots abstraits il la recopie presque telle quelle et la vidéo se fige ; sa variante repart donc d'un text-to-image à chaque tranche (`reset_every: 1`, ~13 s par image avec `image.cpu_offload`, qui le fait tenir sur 16 Go). Sans « no text, no letters » dans le style, il écrit les mots en lettres.
- **Drogue** : Qwen demande des α environ **15 fois** plus grands que Gemma (`drug.alpha_scale: 15`). « ocean » seul affleure à 0,016, domine à 0,05–0,08 en restant cohérent et défait le texte vers 0,12 (runs `small-sweep-a*`). La norme du résidu de Gemma est gonflée par ses activations massives, d'où ses α minuscules.
- `model_path` : copie locale du LLM, utilisée si le dossier existe.
- **Lentille** : sur le même texte, J++ et la J-lens classique partagent environ la moitié des mots par tranche. J++ donne plus de noms concrets (*vocal, speech, sound*), la J-lens plus d'adverbes (*arguably, profoundly*) : `configs/variants/small_jlens.yaml` pour comparer. La direction de la drogue passe par la lentille, donc la calibration ×15 vaut pour J++.

**J-space pour l'analyse** (`readout.save_jspace`, actif dans `small.yaml`, coupé par défaut) : `record` écrit `runs/<id>/jspace.pt`, avec pour chaque token généré et chaque couche de la lentille (0 à 30) la sortie de bloc `h` (après la drogue) et `jh = J[l] @ h`, en fp16 (~64 Mo pour 200 tokens), plus les tokens, l'α de la drogue et la lentille utilisée. La génération ne change pas : la trace est identique à l'octet près avec ou sans. Format et vérification : `introspect/jspace.py`.

```python
from introspect.jspace import load
d = load("runs/small-lsd-sa3")          # d["h"], d["jh"] : [tokens, couches, 2560] ; d["layers"], d["text"], d["alpha"]
```

```bash
./run.sh py jspace runs/small-lsd-sa3 --check   # unembed(jh) redonne la trace (337/340 top-5, écarts dus au fp16)
```

**Page d'analyse** : `./run.sh py analyse runs/<id> --baseline runs/<témoin>` écrit `runs/<id>/analyse.html`, une page autonome (sans dépendance, ~2,5 Mo) : contrôles automatiques (valeurs finies, trace = J-space, injection mesurée contre le témoin, isolement des couches sous la drogue, auto-lecture de l'injection, filtre des échos…), carte pas × couche de 8 mesures avec la lecture de la lentille au survol, profils par couche drogué contre sobre, trajectoires PCA du J-space, tranches avec leurs images. Le témoin est facultatif.

Runs enregistrés : `small-lsd-sa3` (LSD, la vidéo) et son témoin sobre `small-sober-jspace` (même question, même graine, `configs/variants/small_sober.yaml`) ; les deux textes divergent après 16 tokens. La norme moyenne de `h` sur Qwen3.5-4B monte de 2 (couche 0) à 59 (couche 30), sans activations géantes : c'est pourquoi sa drogue demande des α plus grands que Gemma.

**Lecture sans l'injection** (`readout.subtract_injection`, actif par défaut, pour tous les modèles). Le flux résiduel additionne les sorties des blocs : le vecteur que la drogue ajoute aux couches 15..22 reste tel quel dans les couches suivantes, et sa direction est construite pour faire monter les mots tirés dans la lecture de la lentille. Sans correction, la lentille relisait donc la poussée elle-même : 7,7 % de la lecture des couches injectées (jusqu'à 18 %) et 14 % des mots de la vidéo étaient les mots injectés. Le hook garde maintenant la somme de ce qu'il a ajouté pour le token courant, et la lentille lit `h − injection` ; le modèle reçoit toujours `h` avec la drogue, donc le texte ne change pas (vérifié à l'octet près sur `small-lsd-fix` contre `small-lsd-sa3`). Après correction : 0,23 % dans les couches injectées, 1 % des mots de la vidéo (*sacred*, que le modèle a vraiment écrit). Ce qui reste de la drogue dans les mots est la réaction du modèle : leur ressemblance au vocabulaire LSD (embeddings multilingues) passe de 0,534 sobre à 0,569 corrigé, contre 0,629 en lecture brute, surtout au pic (0,613 contre 0,550). Pour un effet plus visible, il faut donc une dose plus forte, plus seulement un effet de lecture.

**Drogue par états** (`configs/states.yaml`, `introspect/states.py`, drogue `lsd_etats`). Au lieu de pousser des mots, une phase peut pousser un état, avec la méthode du projet mdma : vecteur d'un état = moyenne des activations sur 40 réponses écrites dans cet état (prompt « persona »), moins la même moyenne en état sobre. Trois états pour l'instant : `synesthesia` (pic sensoriel), `dissolution` (de l'ego), `afterglow` (rémanence).

```bash
./run.sh py states extract --config configs/small.yaml   # ~75 s -> vectors/qwen3.5-4b/trip_states.pt (0,5 Mo)
./run.sh py states sweep   --config configs/small.yaml   # ~85 s : dose α choisie avec le modèle comme juge (0 à 9)
./run.sh all --config configs/variants/small_lsd_states.yaml --modes mixte
```

- Balayage sur Qwen3.5-4B (8 questions jamais vues, réponses courtes en anglais) : à α 0,08, trait 2,7 → 6,6 (synesthésie), 3,0 → 6,8 (dissolution), 4,3 → 7,1 (rémanence), cohérence ~6,7 ; au-delà de 0,12 la synesthésie et la dissolution se défont. Dans une dissertation de 200 tokens en français, 0,08 fait boucler la dissolution : le catalogue prend `dose: 0.75` (α 0,06).
- Dans la phase, l'injection reste `h += α‖h‖u` aux couches 15..22, avec un mélange des deux états, pondéré par leurs doses, sur 5 % de la prise autour de chaque frontière.
- Effet (`small-lsd-states` contre le témoin sobre et la drogue par mots `small-lsd-fix`, même question, même graine) : le juge note chaque tranche du texte pour l'état de sa phase, synesthésie 5,6 (sobre 0,7, mots 1,0), dissolution 7,3 (2,4 ; 2,5), rémanence 4,8 (1,8 ; 3,8), cohérence inchangée. Les mots intérieurs suivent l'arc du texte (*silicon, burning, neural* → *universe, infinite, soul, love* → *prayer, eternity, instant*).
- Ce que la lentille lit dans chaque vecteur d'état confirme ce qu'il code : *violet, shimmer, neon, rainbow, crimson* ; *cosmic, infinite, universe, eternal, celestial* ; *loving, kindness, joyful, precious, gentle*. Ces vecteurs ne sont pas construits à partir de la lentille : elle les relit peu (1,6 % de la lecture des couches injectées, contre 7,7 % pour les mots).

Limites constatées : les mots intérieurs traduisent souvent en anglais ce que le texte français vient de dire (le filtre des échos ne compare que les mots identiques) ; le BPE de Qwen laisse passer quelques fragments (*philosoph*, *empt*) ; une image SANA sur 20 sort avec des bandes noires.
