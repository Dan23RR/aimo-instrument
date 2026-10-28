#!/usr/bin/env bash
# E16-bis — la stessa estrazione, SENZA i flag deterministici, sullo stesso hardware.
# Serve a separare tre spiegazioni della caduta di AUROC (0,649 su A100 -> 0,513 qui):
#   (a) i flag deterministici,  (b) l'hardware,  (c) lo stato numerico casuale.
# Uso: bash /work/aimo/vm/e16_nodet.sh <gpu> "<modelli>" [N] [ripetizioni]
set -u
GPU="${1:-0}"; MODELLI="${2:-}"; N="${3:-174}"; REP="${4:-2}"
IMG="ghcr.io/aimo-interp/aimo-codabench-runtime:torch2.12.1-cuda13.2-cudnn9-transformers5.13.0-sklearn1.8.0-kernels0.15.2"
mkdir -p /work/aimo/e16nodet/log
echo "=== E16-bis (senza flag) gpu=$GPU N=$N rep=$REP inizio $(date -u) ==="
for m in $MODELLI; do
  t=${m//\//__}
  for r in $(seq 0 $((REP-1))); do
    out="/work/aimo/e16nodet/rep$r"
    mkdir -p "$out"
    if [ -s "$out/e16_${t}.jsonl" ]; then echo "salto $m rep$r"; continue; fi
    echo "[$(date -u +%H:%M)] $m rep$r su gpu$GPU (nessun flag deterministico)"
    docker run --rm --gpus "device=$GPU" -e HF_HOME=/work/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      -e PYTHONUNBUFFERED=1 -v /work:/work -w /work/aimo "$IMG" \
      python /work/aimo/vm/e16_estrai.py --model "$m" --gpu "$GPU" --n "$N" --out "$out" > "/work/aimo/e16nodet/log/${t}_rep$r.log" 2>&1
    echo "[$(date -u +%H:%M)] $(grep -h 'scritto' /work/aimo/e16nodet/log/${t}_rep$r.log | cut -c1-200)"
  done
done
echo "=== E16-bis gpu$GPU fine $(date -u) ==="
