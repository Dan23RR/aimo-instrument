#!/usr/bin/env bash
# E14 — orchestratore. Esegue le condizioni di PREREG_E14.md dentro l'immagine ufficiale.
# Uso:  bash /work/aimo/vm/e14_all.sh [N]      (N problemi, default 40; usare 5 per la prova)
# Log: /work/aimo/e14/log/*.log ; output: /work/aimo/e14/out/*.jsonl
set -u
N="${1:-40}"
CONDS="${2:-C0 C1 C2 C3 C4}"
ha() { case " $CONDS " in *" $1 "*) return 0;; *) return 1;; esac; }
IMG="ghcr.io/aimo-interp/aimo-codabench-runtime:torch2.12.1-cuda13.2-cudnn9-transformers5.13.0-sklearn1.8.0-kernels0.15.2"
mkdir -p /work/aimo/e14/log /work/aimo/e14/out
cd /work/aimo
PY="${E14_PY:-python}"

run() {  # cond rep gpu tag [extra docker args...]
  local cond=$1 rep=$2 gpu=$3 tag=$4; shift 4
  local log="/work/aimo/e14/log/${cond}_rep${rep}_gpu${gpu}_${tag}.log"
  if [ -s "/work/aimo/e14/out/e14_${cond}_rep${rep}_gpu${gpu}_${tag}.jsonl" ]; then echo "salto $cond rep$rep $tag (esiste)"; return; fi
  echo "[$(date -u +%H:%M:%S)] avvio $cond rep$rep gpu$gpu $tag"
  docker run --rm --gpus "\"device=$gpu\"" -e HF_HOME=/work/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e PYTHONUNBUFFERED=1 "$@" -v /work:/work -w /work/aimo "$IMG" \
    $PY /work/aimo/vm/e14_run.py --cond "$cond" --rep "$rep" --gpu "$gpu" --tag "$tag" --n "$N" > "$log" 2>&1
  echo "[$(date -u +%H:%M:%S)] fine  $cond rep$rep $tag -> $(grep -o 'gen [0-9]*s = [0-9.]* s/item.*' "$log" | tail -1)"
}

echo "=== E14 N=$N condizioni [$CONDS] inizio $(date -u) ==="
# C0: default, 5 ripetizioni, scheda 0
if ha C0; then for r in 0 1 2 3 4; do run C0 $r 0 a; done; fi
# C1: batch 1, 3 ripetizioni
if ha C1; then for r in 0 1 2; do run C1 $r 0 a; done; fi
# C2: flag deterministici, 3 ripetizioni
if ha C2; then for r in 0 1 2; do run C2 $r 0 a -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e E14_DETERMINISTIC=1; done; fi
# C3: contesa, due processi identici sulla stessa scheda, 3 coppie
if ha C3; then for r in 0 1 2; do
  run C3 $r 0 a & p1=$!
  run C3 $r 0 b & p2=$!
  wait $p1 $p2
done; fi
# C4: default sulla seconda scheda, 3 ripetizioni
if ha C4; then for r in 0 1 2; do run C4 $r 1 a; done; fi
echo "=== E14 fine $(date -u) ==="
ls -la /work/aimo/e14/out | tail -25
