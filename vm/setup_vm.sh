#!/usr/bin/env bash
# Primo accesso alla VM AIMO (2x RTX PRO 6000 Blackwell). Idempotente: si puo' rilanciare.
# Uso, da root sulla VM:   bash setup_vm.sh 2>&1 | tee /root/setup_vm.log
set -u
echo "=== [0] stato iniziale ==="
date -u; hostname; cat /etc/os-release | head -2
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || echo "!!! nvidia-smi assente"
nvcc --version 2>/dev/null | tail -1 || echo "(nvcc assente: non serve per vLLM in wheel)"
python3 --version; df -h / | tail -1; free -g | head -2; nproc

echo "=== [1] pacchetti di base ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq tmux git rsync htop curl ca-certificates gnupg lsb-release build-essential python3-venv python3-pip >/dev/null && echo ok

echo "=== [2] uv ==="
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; fi
export PATH="$HOME/.local/bin:$PATH"; uv --version

echo "=== [3] Docker + NVIDIA Container Toolkit (per l'immagine ufficiale, E14/E16) ==="
if ! command -v docker >/dev/null; then curl -fsSL https://get.docker.com | sh >/dev/null 2>&1 && echo "docker installato"; fi
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit >/dev/null && nvidia-ctk runtime configure --runtime=docker >/dev/null && systemctl restart docker && echo "toolkit ok"
fi
docker --version; docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi -L 2>&1 | tail -3

echo "=== [4] ambiente Python per vLLM (E13/E15) ==="
mkdir -p /work && cd /work
if [ ! -d /work/venv ]; then uv venv /work/venv --python 3.12 >/dev/null; fi
source /work/venv/bin/activate
uv pip install -q vllm pandas pyarrow "huggingface_hub[cli]" transformers openai-harmony accelerate 2>&1 | tail -2
python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| vllm', vllm.__version__, '| gpu', torch.cuda.device_count())"

echo "=== [5] immagine ufficiale del runtime (in background) ==="
nohup docker pull ghcr.io/aimo-interp/aimo-codabench-runtime:torch2.12.1-cuda13.2-cudnn9-transformers5.13.0-sklearn1.8.0-kernels0.15.2 > /root/pull_runtime.log 2>&1 &

echo "=== [6] modelli nella cache HF (in background, ~210 GB) ==="
export HF_HOME=/work/hf; mkdir -p "$HF_HOME"
for m in deepseek-ai/DeepSeek-R1-0528-Qwen3-8B Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think openai/gpt-oss-120b; do
  nohup hf download "$m" > "/root/dl_$(echo $m | tr '/' '_').log" 2>&1 &
done
echo "download avviati; controlla con: tail -n2 /root/dl_*.log ; du -sh /work/hf"

echo "=== [7] riepilogo ==="
echo "HF_HOME=/work/hf  venv=/work/venv  immagine: vedi /root/pull_runtime.log"
echo "prossimo passo: rsync di dati/ gpu/ soluzione/ in /work/aimo dal PC, poi prova E14 su 5 problemi"
date -u
