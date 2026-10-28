#!/usr/bin/env bash
# E16 — score del regressore ufficiale sui 174 problemi pubblici, per i 5 modelli di gara.
# Uso: bash /work/aimo/vm/e16_all.sh <gpu> "<modelli>" [N]
# Dentro l'immagine ufficiale, con i flag deterministici trovati in E14 (costano il 5%).
set -u
GPU="${1:-0}"; MODELLI="${2:-}"; N="${3:-174}"
IMG="ghcr.io/aimo-interp/aimo-codabench-runtime:torch2.12.1-cuda13.2-cudnn9-transformers5.13.0-sklearn1.8.0-kernels0.15.2"
mkdir -p /work/aimo/e16/log
echo "=== E16 gpu=$GPU N=$N inizio $(date -u) ==="
for m in $MODELLI; do
  t=${m//\//__}
  if [ -s "/work/aimo/e16/e16_${t}.jsonl" ]; then echo "salto $m (esiste)"; continue; fi
  echo "[$(date -u +%H:%M)] $m su gpu$GPU"
  docker run --rm --gpus "device=$GPU" -e HF_HOME=/work/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e PYTHONUNBUFFERED=1 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e E14_DETERMINISTIC=1 \
    -v /work:/work -w /work/aimo "$IMG" \
    python /work/aimo/vm/e16_estrai.py --model "$m" --gpu "$GPU" --n "$N" > "/work/aimo/e16/log/${t}.log" 2>&1
  echo "[$(date -u +%H:%M)] $(grep -h 'scritto' /work/aimo/e16/log/${t}.log | cut -c1-220)"
  grep -q Traceback "/work/aimo/e16/log/${t}.log" && echo "!!! errore su $m"
done
echo "=== E16 gpu$GPU fine $(date -u) ==="
