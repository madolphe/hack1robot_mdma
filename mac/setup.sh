#!/usr/bin/env bash
# Installe un venv local pour txt2img.py (macOS, Apple Silicon).
set -euo pipefail
cd "$(dirname "$0")"
echo "Mac : $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) Go de RAM, macOS $(sw_vers -productVersion)"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch diffusers transformers accelerate safetensors gguf
# HF_HUB_DISABLE_XET : le client Xet bloque sur cette machine (géré dans txt2img.py)
.venv/bin/python -c "import torch, diffusers, transformers; print('torch', torch.__version__, '| diffusers', diffusers.__version__, '| transformers', transformers.__version__, '| mps', torch.backends.mps.is_available())"
