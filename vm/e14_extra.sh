#!/usr/bin/env bash
# E14 — condizioni aggiuntive su una scheda scelta: C5 (senza compilazione) e C4 (seconda scheda).
# Uso: bash /work/aimo/vm/e14_extra.sh <gpu> "C5 C4" [N]
# Non modifica e14_all.sh mentre gira. Stessa immagine, stesso e14_run.py.
set -u
GPU="${1:-1}"
CONDS="${2:-C5 C4}"
N="${3:-40}"
IMG="ghcr.io/aimo-interp/aimo-codabench-runtime:torch2.12.1-cuda13.2-cudnn9-transformers5.13.0-sklearn1.8.0-kernels0.15.2"
mkdir -p /work/aimo/e14/log /work/aimo/e14/out
cd /work/aimo
ha() { case " $CONDS " in *" $1 "*) return 0;; *) return 1;; esac; }
run() {  # cond rep gpu tag [extra docker args...]
  local cond=$1 rep=$2 gpu=$3 tag=$4; shift 4
  local log="/work/aimo/e14/log/${cond}_rep${rep}_gpu${gpu}_${tag}.log"
  if [ -s "/work/aimo/e14/out/e14_${cond}_rep${rep}_gpu${gpu}_${tag}.jsonl" ]; then echo "salto $cond rep$rep $tag (esiste)"; return; fi
  echo "[$(date -u +%H:%M:%S)] avvio $cond rep$rep gpu$gpu $tag"
  docker run --rm --gpus "device=$gpu" -e HF_HOME=/work/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e PYTHONUNBUFFERED=1 "$@" -v /work:/work -w /work/aimo "$IMG" \
    python /work/aimo/vm/e14_run.py --cond "$cond" --rep "$rep" --gpu "$gpu" --tag "$tag" --n "$N" > "$log" 2>&1
  echo "[$(date -u +%H:%M:%S)] fine  $cond rep$rep $tag -> $(grep -o 'gen [0-9]*s = [0-9.]* s/item.*' "$log" | tail -1)"
}
echo "=== E14 extra gpu=$GPU [$CONDS] N=$N inizio $(date -u) ==="
if ha C5; then for r in 0 1 2; do run C5 $r $GPU a -e TORCH_COMPILE_DISABLE=1; done; fi
if ha C4; then for r in 0 1 2; do run C4 $r $GPU a; done; fi
echo "=== E14 extra fine $(date -u) ==="
