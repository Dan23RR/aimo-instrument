#!/usr/bin/env bash
# E13 — orchestratore (PREREG_E13.md). Uso: bash /work/aimo/vm/e13_all.sh [N]   (N problemi; 5 per la prova)
set -u
N="${1:-164}"
K="${E13_K:-10}"
source /work/venv/bin/activate
export HF_HOME=/work/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p /work/aimo/e13/log
cd /work/aimo/vm
L=/work/aimo/e13/log

echo "=== E13 N=$N K=$K inizio $(date -u) ==="
# varianti: due estrazioni indipendenti, una per scheda
[ -f /work/aimo/e13/varianti_D1.parquet ] || python e13_varianti.py --draw D1 --seed 20260913 --n $N --k $K --gpu 0 > $L/var_D1.log 2>&1 &
p1=$!
[ -f /work/aimo/e13/varianti_D2.parquet ] || python e13_varianti.py --draw D2 --seed 20260914 --n $N --k $K --gpu 1 > $L/var_D2.log 2>&1 &
p2=$!
wait $p1 $p2
echo "[$(date -u +%H:%M)] varianti pronte: $(grep -h 'scritto' $L/var_D1.log $L/var_D2.log | cut -c1-160)"

# etichettatura: per modello, tre condizioni; due modelli alla volta sulle due schede
MODELLI="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think openai/gpt-oss-120b"
etich() {  # model gpu
  local m=$1 g=$2 t=${1//\//__}
  for c in "A D1 20260921" "B D2 20260922" "Ap D1 20260923"; do
    set -- $c
    [ -f /work/aimo/e13/etich_${t}_$1.parquet ] && { echo "salto $m $1"; continue; }
    echo "[$(date -u +%H:%M)] $m cond $1 su gpu$g"
    python e13_etichetta.py --model "$m" --cond $1 --draw $2 --seed $3 --gpu $g --n $N > $L/etich_${t}_$1.log 2>&1
    echo "[$(date -u +%H:%M)] $(grep -h 'scritto' $L/etich_${t}_$1.log | cut -c1-200)"
  done
}
i=0
for m in $MODELLI; do
  etich "$m" $((i % 2)) &
  pids[$i]=$!
  i=$((i + 1))
  if [ $((i % 2)) -eq 0 ]; then wait; fi
done
wait
echo "=== E13 fine $(date -u) ==="
ls -la /work/aimo/e13/*.parquet
