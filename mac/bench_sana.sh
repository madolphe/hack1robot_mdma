#!/usr/bin/env bash
# Compare les variantes de Sana-Sprint sur le même prompt et la même seed.
cd "$(dirname "$0")"
P="A surreal dreamlike oil painting, a visual metaphor of silence, memory, ocean"
for v in "1.6B bf16" "1.6B fp16" "0.6B bf16" "0.6B fp16"; do
  set -- $v
  extra=(); [ "$1" = "0.6B" ] && extra=(--small)
  echo "===== $1 $2"
  .venv/bin/python txt2img_sana.py "$P" --dtype "$2" "${extra[@]}" -o "out/bench-$1-$2.png" 2>&1 \
    | grep -v -E "Warning|warn|FutureWarning|deprecate|redirects"
done 2>&1 | tee bench_sana.log
echo "===== fini"
