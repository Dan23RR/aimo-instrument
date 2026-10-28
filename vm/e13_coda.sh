#!/usr/bin/env bash
# E13 — coda dell'etichettatura per scheda. Aspetta che E14 liberi la scheda e che le varianti D2 esistano.
# Uso: bash /work/aimo/vm/e13_coda.sh <gpu> "<modelli separati da spazio>" [N]
#   gpu0: aspetta "=== E14 fine" in e14_all.log ;  gpu1: aspetta "=== E14 extra fine" in e14_extra.log
set -u
GPU="$1"; MODELLI="$2"; N="${3:-164}"; CONDS="${4:-A B Ap}"
vuole() { case " $CONDS " in *" $1 "*) return 0;; *) return 1;; esac; }
source /work/venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda HF_HOME=/work/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
L=/work/aimo/e13/log; mkdir -p $L; cd /work/aimo/vm
if [ "$GPU" = "0" ]; then SEGNALE="=== E14 fine"; FILE=/work/aimo/e14_all.log; else SEGNALE="=== E14 extra fine"; FILE=/work/aimo/e14_extra.log; fi
echo "[$(date -u +%H:%M)] gpu$GPU: aspetto '$SEGNALE' e varianti_D2.parquet"
until grep -q "$SEGNALE" "$FILE" 2>/dev/null && [ -f /work/aimo/e13/varianti_D2.parquet ]; do sleep 60; done
echo "[$(date -u +%H:%M)] gpu$GPU: via. modelli: $MODELLI"
for m in $MODELLI; do
  t=${m//\//__}
  for c in "A D1 20260921" "B D2 20260922" "Ap D1 20260923"; do
    set -- $c
    vuole $1 || { echo "escluso $m $1 (non in [$CONDS])"; continue; }
    if [ -f /work/aimo/e13/etich_${t}_$1.parquet ]; then echo "salto $m $1"; continue; fi
    # DeepSeek-R1 chiude da solo con \boxed e ha gia' la condizione A col prompt nudo: si resta
    # coerenti dentro il modello. Gli altri richiedono il formato esplicito (misurato il 9/9).
    case "$m" in *DeepSeek-R1*) P=nudo;; *) P=esplicito;; esac
    echo "[$(date -u +%H:%M)] gpu$GPU: $m cond $1 ($2, seme $3, prompt $P)"
    python e13_etichetta.py --model "$m" --cond $1 --draw $2 --seed $3 --gpu $GPU --n $N --prompt $P > $L/etich_${t}_$1.log 2>&1
    echo "[$(date -u +%H:%M)] gpu$GPU: $(grep -h 'scritto' $L/etich_${t}_$1.log | cut -c1-220)"
    grep -q "Traceback" $L/etich_${t}_$1.log && echo "!!! errore in $m $1, vedi $L/etich_${t}_$1.log"
  done
done
echo "=== E13 coda gpu$GPU fine $(date -u) ==="
