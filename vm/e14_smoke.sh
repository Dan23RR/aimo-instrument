#!/usr/bin/env bash
# Prova di E14 su 5 problemi: aspetta l'immagine, verifica lo stack dentro il container, esegue C0 una volta.
set -u
IMG="ghcr.io/aimo-interp/aimo-codabench-runtime:torch2.12.1-cuda13.2-cudnn9-transformers5.13.0-sklearn1.8.0-kernels0.15.2"
mkdir -p /work/aimo/e14/smoke /work/aimo/e14/log /work/aimo/e14/out
echo "attendo l'immagine... $(date -u)"
until docker image inspect "$IMG" >/dev/null 2>&1; do sleep 20; done
echo "immagine pronta $(date -u)"
echo "--- stack nel container ---"
docker run --rm --gpus device=0 "$IMG" sh -c 'which python python3; python -c "import sys,torch,transformers,sklearn,pandas;print(sys.executable, \"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"tf\", transformers.__version__, \"sk\", sklearn.__version__, \"gpu\", torch.cuda.get_device_name(0))"' 2>&1 | tail -3
echo "--- prova C0 su 5 problemi ---"
docker run --rm --gpus device=0 -e HF_HOME=/work/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -v /work:/work -w /work/aimo "$IMG" python /work/aimo/vm/e14_run.py --cond C0 --rep 0 --gpu 0 --tag smoke --n 5 --out /work/aimo/e14/smoke 2>&1 | tail -15
echo "SMOKE_FINE $(date -u)"
