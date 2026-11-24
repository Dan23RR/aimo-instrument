#!/usr/bin/env bash
# Prova di E13 su 3 problemi, k=2, scheda 1: varianti con gpt-oss-120b, poi etichettatura DeepSeek con 3 campioni base.
set -u
source /work/venv/bin/activate
export HF_HOME=/work/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p /work/aimo/e13/smoke
cd /work/aimo/vm
echo "--- varianti D1 (3 problemi, k=2) $(date -u) ---"
python e13_varianti.py --draw D1 --seed 20260913 --n 3 --k 2 --gpu 1 --out /work/aimo/e13/smoke 2>&1 | grep -E "E13|vLLM caricato|passo|scritto|Error|Traceback|error" | tail -12
echo "--- etichetta DeepSeek cond A (3 problemi, 3 campioni base) $(date -u) ---"
python e13_etichetta.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --cond A --draw D1 --seed 20260921 --gpu 1 --n 3 --n-base 3 --out /work/aimo/e13/smoke 2>&1 | grep -E "E13|vLLM caricato|blocco|scritto|Error|Traceback|error" | tail -12
echo "SMOKE13_FINE $(date -u)"
