#!/usr/bin/env bash
# Poids des petits modèles (configs/small.yaml) dans le cache HF du projet (weights/hub), ~25 Go
# (+16 Go pour FLUX.2 [klein]). Licences à accepter d'abord sur huggingface.co : stable-audio-3-small-sfx.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_CACHE="$PWD/weights/hub" HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"  # le client Xet (Rust) se fait couper derrière certains proxys : HTTP simple
HF=.venv/bin/hf
$HF download neuronpedia/jacobian-lens qwen3.5-4b/jpp/Salesforce-wikitext/Qwen3.5-4B_jpp_lens.pt qwen3.5-4b/jpp/Salesforce-wikitext/config.yaml qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt
[ -d /home/flowers/work/hf/Qwen3.5-4B ] || $HF download Qwen/Qwen3.5-4B   # copie locale : model_path dans configs/small.yaml
# un motif par option : après le premier, `hf download` lirait les suivants comme des noms de fichiers
$HF download hexgrad/Kokoro-82M --include "*.json" --include "kokoro-v1_0.pth" --include "voices/ff_siwis.pt" \
  --include "voices/af_heart.pt" --include "voices/am_michael.pt" --include "voices/bm_george.pt"
$HF download sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --exclude "onnx/*" --exclude "openvino/*" \
  --exclude "pytorch_model.bin" --exclude "tf_model.h5"
$HF download Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers
$HF download black-forest-labs/FLUX.2-klein-4B --exclude "flux-2-klein-4b.safetensors"
# Stable Audio 3 small-sfx (~3,5 Go), lu par .venv-audio (voir README, section 7)
$HF download stabilityai/stable-audio-3-small-sfx --include "model_config.json" --include "model.safetensors" \
  --include "t5gemma-b-b-ul2/*" || echo "Stable Audio 3 indisponible (licence non acceptée ?) : couche A désactivée"
# Stable Audio Open 1.0 (engine: diffusers), seulement les dossiers de diffusers (~5,3 Go sur 15,7) :
# $HF download stabilityai/stable-audio-open-1.0 --include "model_index.json" --include "projection_model/*" \
#   --include "scheduler/*" --include "text_encoder/*" --include "tokenizer/*" --include "transformer/*" --include "vae/*"
echo ALL-DONE
