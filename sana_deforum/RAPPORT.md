# Rapport : boucle Deforum + Sana-Sprint img2img (réponse à GUIDELINES.md)

21 novembre 2026, branche `small-models`. La méthode de `sana_deforum_test.py` est reprise telle quelle (classes `Sana`,
`Depth`, `Warper`, couleurs LAB, mode `angle`) par `introspect/deforum.py`, qui la cale sur un run du pipeline : le prompt
d'image de chaque tranche devient une image clé de prompt au moment où la tranche commence (fondu sur 12 images), et le
son du montage est ajouté.

```bash
./run.sh py deforum --run runs/<id> --strength 0.35 --color 0.2 --big    # -> runs/<id>/out_deforum.mp4
```

## Machine

RTX 4090 Laptop (16 Go), Ubuntu 24.04, pilote 580, torch 2.14.1+cu130, diffusers 0.43.0, transformers 5.23.0.

## Runs

Run du pipeline `photo-lumiere` (« Quel goût a la lumière du matin ? », drogue par états, prompts d'image « photo de
cinéma »), réglage `FovVid1CompSettings.txt`, cadence 2, 15 i/s, mode `angle` sauf mention.

| Réglage | Taille | Modèle | s/diffusion | s/image | Pic mémoire | Rendu |
|---|---|---|---|---|---|---|
| strength 0,675, couleur 1 (référence) | 1024×576 | 0.6B | 0,71 | 0,56 | 7,2 Go | s'enfonce, mais flou feutré dès 3 s, les prompts ne passent pas |
| strength 0,55, couleur 0,3 | 1024×576 | 0.6B | 0,72 | 0,58 | 7,2 Go | se renouvelle un peu, tourne au graphisme plat |
| strength 0,45, couleur 0 | 1024×576 | 0.6B | 0,74 | 0,53 | 7,2 Go | couleurs qui dérivent (rouge, cyan), motif figé |
| strength 0,55, couleur 0,3 | 1024×576 | 1.6B | 0,83 | 0,64 | 9,2 Go | plus net, vitrail |
| strength 0,5, couleur 0,3, `steps` 4 pas | 1024×576 | 1.6B | 0,81 | 0,62 | 9,2 Go | comme `angle`, s'aplatit pareil |
| **strength 0,35, couleur 0,2** | 1024×576 | 1.6B | 0,82 | 0,65 | 9,3 Go | **le seul où les prompts passent** (voir ci-dessous) |
| strength 0,35, couleur 0,2 | 768×416 | 1.6B | 0,75 | 0,50 | 9,2 Go | (mesure de temps seulement) |

Chargement 5 à 7 s, encodage des prompts 0,5 à 1,4 s (28 prompts). La diffusion fait ~65 % du temps, la profondeur ~10 %,
la correction de couleur ~10 %. Le coût par diffusion dépend peu de la taille : 1024×576 coûte à peine plus que 768×416.

## Jugement (vidéo complète : 40,5 s, 608 images, strength 0,35, 1.6B, 6 min 49 s)

- **Sensation d'avancer : oui.** La caméra entre dans l'image de façon continue, sans coupure.
- **Les prompts passent quand la force est basse.** Les citrons deviennent des tranches cristallines, puis des lèvres
  et une lune vers 16-20 s, puis des points, des vagues et un violon à la fin : ce sont les mots des tranches
  correspondantes. À 0,55 et au-dessus, la première image domine tout l'aperçu de 6 s.
- **Mais le détail se perd.** Entre ~10 et ~34 s, l'image devient de grandes surfaces plates kaki et olive : chaque passe
  img2img en 2 pas perd un peu de détail, et la boucle l'accumule. Les couleurs dérivent vers ce ton boueux même avec
  la correction sur l'image 0 (0,2).
- **Clignotement :** faible, la continuité est bonne. **Bords en miroir :** peu visibles avec la rotation lente.
- Le mode `angle` ne sort pas d'images délavées ni grises ; il n'est pas pire que `steps` à force égale.

## Moins de continuité : strength 0,30

À 0,30, l'image change ~8 % plus vite sur les 12 premières secondes (47 contre 44/255 d'écart par seconde). Les citrons
tiennent toujours ~4 s, comme les quatre premiers prompts (*citrus, lemon, tongue*), et la géométrie de tranche jusqu'à
~10 s : c'est la structure de l'image, que la reprojection garde d'une image à l'autre, plus que la force. En revanche, les
motifs des prompts reviennent plus souvent : le violon dès ~14 s, les lèvres vers 20 s et à la fin.

## Balayage de la force (cadence 3, 1.6B, couleur 0,2)

| strength | Changement / s | Saut à chaque diffusion | Couleur | Rendu |
|---|---|---|---|---|
| 0,10 | 226 | 4,1× | 19,3 | presque une photo neuve à chaque diffusion, fidèle aux prompts (lèvres, lavande, violon, rose sur les vagues, cœur au néon), mais l'image « pulse » 5 fois par seconde |
| 0,20 | 180 | 2,6× | 20,7 | entre les deux : objets photographiques, plus de continuité, le fond kaki commence |
| 0,25 | 183 | 2,2× | 23,5 | graphique, aplats kaki, motifs dessinés |
| 0,27 | 171 | 2,1× | 24,4 | idem |
| 0,30 | 159 | 2,0× | 24,7 | idem (défaut) |

Saut : mouvement de l'image à une image diffusée, divisé par celui d'une image où seule la caméra bouge. Au-dessus de
~0,25, la boucle perd la richesse photographique des prompts et tombe dans le style plat ; en dessous de ~0,2, elle
garde le style des prompts mais perd la continuité. Vidéos : `runs/photo-lumiere/out_deforum_k10.mp4` … `k27.mp4`,
grille synchronisée `runs/photo-lumiere/keep_compare/keep_grid.mp4`.

## Drogue en plusieurs phases : couleurs par tranche

Sur `runs/lumiere-bad-trip` (montée → pic sensoriel → bad trip → descente → rémanence, force 0,22), la correction de
couleur ramenait chaque diffusion vers l'image 0 (miel, lèvres) : toute la vidéo restait beige et dorée, bad trip compris
(`out_deforum_colorfirst.mp4`). `--color-ref slice` (défaut) prend comme référence la palette de l'image fixe de la
tranche en cours, celle de la passe `render`, qui suit déjà l'état, avec le même fondu que les prompts : violet au pic,
bleu-vert sombre au bad trip, gris à la descente, doré à la fin (`out_deforum.mp4`, même temps de rendu, 194 s).

Ce qui reste : la caméra avance vite (`translation_z` jusqu'à 6,75 vers 6 s, puis 2,97 par image après 10 s) et
la reprojection étire le bord de l'image en rayures qui convergent vers le point de fuite ; à 0,22, la diffusion les garde
et elles occupent une bonne part du pic sensoriel et du bad trip. Elles existaient déjà dans `photo-lumiere`, moins
présentes. `--z-scale` plus bas les réduirait (non essayé).

## Temps : pourquoi 7 min pour 40 s

Une vidéo du pipeline fait 28 images (une par tranche, ~40 s de rendu) ; la boucle en fait 608 (15 i/s) avec 303
diffusions, soit ~11 fois plus de génération. Répartition sur le premier rendu (394 s de boucle) : diffusion 250 s, sauvegarde
des PNG et copies 61 s, correction de couleur en numpy 46 s, profondeur 37 s. `introspect/deforum.py` fait maintenant la
correction de couleur sur le GPU (mêmes formules, écart < 1e-5, 0,9 ms contre 152 ms) et enregistre les PNG avec une
compression rapide (toujours sans perte) : 327 s au lieu de 409 s. Le reste est la diffusion elle-même ; pour aller plus
vite, il faut moins de diffusions (cadence 3, 12 i/s) ou le transformer 0.6B (0,71 s contre 0,82 s par diffusion).

## Accélérer : ce qui marche, ce qui ne marche pas

Mesures sur un appel img2img (1.6B, mode `angle`, entrée 1024×576) :

- **Le pipeline travaille à 1344×768.** SanaSprintImg2ImgPipeline ramène l'entrée à la taille native la plus proche
  (« resolution binning » : 1024×576 → 1344×768, 768×416 → 1344×704), y fait tout le calcul, puis revient à la taille
  demandée. C'est pourquoi la taille de la boucle change si peu le temps.
- **À 1344×768, le VAE fait l'essentiel :** encodage 266 ms, décodage 342 ms, transformer ~120 ms (2 pas), sur ~730 ms.
- **Appeler le VAE et le transformer sans le pipeline** (`DirectImg2Img`, `introspect/deforum.py`) donne des images
  identiques au bit près, mais pas plus vite (722 contre 733 ms) : le surcoût du pipeline est négligeable.
- **`torch.compile`** : impossible ici, Triton compile un module C et l'en-tête `Python.h` de Python 3.12 n'est pas
  installé (paquet `python3.12-dev`).
- **`channels_last` sur le VAE** : 4 % seulement, et la sortie change ; écarté.
- **Profondeur seulement après une diffusion** (entre deux, la caméra bouge à peine, on garde la carte) : 35 → 18 s.
- **PNG écrits dans un fil à part**, pendant que le GPU calcule : ~28 s → ~0 s.
- **`--no-binning`** : SANA à 1024×576 directement, 0,50 s par diffusion au lieu de 0,83 (-40 %). Le rendu change :
  plus d'objets en volume (boîtes, disques, marches), moins de grands aplats kaki.

| Version (40,5 s, 608 images, strength 0,30) | Temps total |
|---|---|
| Premier rendu | 6 min 49 s |
| Couleur sur le GPU, PNG rapides | 5 min 27 s |
| + profondeur après diffusion, écriture en parallèle (même réglage) | 4 min 46 s |
| + `--no-binning` | 3 min 07 s |

Moins d'images diffusées (`fast` = profondeur après diffusion, écriture en parallèle) :

| Version | Images | Diffusions | Temps total | Sauts à chaque diffusion |
|---|---|---|---|---|
| 15 i/s, cadence 2 (`fast`) | 608 | 303 | 4 min 46 s | 2,0 fois le mouvement d'une image caméra seule, 7,5 par seconde |
| 12 i/s, cadence 2 (`--fps 12`, même vitesse de caméra par seconde) | 486 | 242 | 3 min 51 s | 1,7 fois |
| **15 i/s, cadence 3** (défaut de `introspect/deforum.py`) | 608 | 202 | **3 min 15 s** | 2,0 fois, 5 par seconde |

La cadence 3 ne fait pas plus « pulser » l'image : chaque diffusion la déplace autant qu'à la cadence 2, simplement
moins souvent, et le mouvement de caméra reste à 15 i/s.

Une boucle à rétroaction amplifie le moindre écart : avec la profondeur réutilisée, les motifs ne sont plus les mêmes
(lune, lèvres, rose, danseur au lieu du violon), dans le même style. `--depth-every frame` rend la boucle d'origine.

## Recommandation

- Réglage retenu (défauts de `introspect/deforum.py`) : strength 0,30, couleur 0,2, 1.6B, cadence 3, 15 i/s, 1024×576.
- Temps estimés à 15 i/s, chargement compris, avant l'accélération : 20 s de vidéo ≈ 2,5 min en 768×432, 3,4 min en
  1024×576 ; 60 s ≈ 7,6 min en 768×432, 9,9 min en 1024×576. Après les accélérations ci-dessus : 60 s ≈ 7 min avec le
  binning, ≈ 4,6 min avec `--no-binning`.
- Pour garder du détail sur des vidéos longues, la boucle seule ne suffit pas. Pistes non essayées : réancrer à chaque
  début de tranche en mêlant une image t2i fraîche de la tranche (ou l'image clé déjà rendue par `render`) à l'image
  courante, ajouter un léger renforcement de netteté (anti-flou de Deforum) avant chaque diffusion, et prendre la
  référence de couleur au début de chaque phase plutôt qu'à l'image 0.
