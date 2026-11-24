#!/usr/bin/env bash
# E10 stadio 2 — da lanciare SOLO se lo stadio 1 (H22 e H23) e' sopravvissuto.
# Per ciascun modello nuovo: etichette robust_trad (vLLM) + punteggi del regressore (HF,
# identico al container). Poi calibra le soglie e scrive soluzione/soglie_per_modello.json.
# Uso: bash e10_stadio2.sh   (in background: run_in_background)
set -u
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"
T=$(cat out/e10_traduttore.txt)
MODELLI=("Qwen/Qwen3.5-4B" "Skywork/Skywork-OR1-Math-7B" "allenai/Olmo-3-7B-Think" "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
TAG=stadio2
echo "=== STADIO 2: traduttore=$T modelli=${#MODELLI[@]} ==="
for M in "${MODELLI[@]}"; do
  S=${M//\//__}
  echo "--- [$M] etichette robust_trad (vLLM) ---"
  python -m modal run --detach e10_modal.py --model "$M" --tag "$TAG" --translator "$T" 2>&1 | grep -E "modello=|vLLM|generazioni=|scritto|Error|Traceback|\{'file'" | tail -6
  echo "--- [$M] punteggi regressore (HF, come il container) ---"
  python -m modal run --detach e9_feature_modal.py --model "$M" --tag "$TAG" 2>&1 | grep -E "modello=|caricato|scritto|Error|Traceback|\{'file'" | tail -6
  python -m modal volume get aimo-e9 "e10_${TAG}_${S}.parquet" "out/e10_${TAG}_${S}.parquet" --force 2>&1 | tail -1
  python -m modal volume get aimo-e9 "e9feat_${TAG}_${S}.parquet" "out/e9feat_${TAG}_${S}.parquet" --force 2>&1 | tail -1
done
echo "=== CALIBRAZIONE ==="
python e9_calibra.py "$TAG" "${MODELLI[@]}" --prefisso e10 --colonna robust_trad
echo "=== FINE STADIO 2 ==="
