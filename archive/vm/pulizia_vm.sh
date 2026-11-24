#!/usr/bin/env bash
# pulizia_vm.sh: restituzione della VM kinit-event-gpu01 agli organizzatori AIMO (17 settembre 2026).
#
# Senza argomenti: SOLO ELENCO di cio' che verrebbe rimosso. Nessuna modifica.
# Con l'argomento CONFERMO: rimuove SOLO cio' che abbiamo creato noi e, come ultimo passo, toglie
# le NOSTRE due chiavi SSH da authorized_keys lasciando quella degli organizzatori.
# La cancellazione la lancia Daniel, non l'assistente.
#
# Backup verificato prima di scrivere questo file: AIMO/vm_backup/backup_aimo_2026-09-17.tar.gz,
# sha256 66b9fc87ec7b4d4e128baf022d34f331d5797f77b5a358f98cd8b6d5b62bfea5, 214 file, hash identico
# sulla VM e in locale. Ambiente e revisioni dei modelli in AIMO/vm_backup/AMBIENTE_VM_2026-09-17.txt.
#
# Attribuzione per data di creazione (stat), controllata il 17/9:
#   NOSTRO: /work (mkdir di setup_vm.sh, 8/9 12:27); cache di uv, vllm, flashinfer; .triton .nv
#   .humming .tilelang (primi run di vLLM, 8/9 12:34-12:40); .config/uv .config/fish e le righe
#   "env" in .bashrc .profile .zshrc (installer di uv, 8/9 12:26:59); .config/vllm; uv in
#   .local/bin e .local/share/uv (dentro c'e' solo un lock vuoto, nessuna credenziale); le due
#   immagini docker; i log di setup e di download in /root.
#   NON NOSTRO, NON TOCCATO: nvim e htop (8/9 11:04-11:31, prima del nostro accesso),
#   .python_history, cuda-keyring, .bash_history (contiene anche le sessioni degli organizzatori).
#   Lasciati apposta: i pacchetti di sistema installati (docker, nvidia-container-toolkit, apt).
set -euo pipefail

MODO="${1:-ELENCO}"
CHIAVE_DANIEL="AAAAC3NzaC1lZDI1NTE5AAAAILm7oyYDqhEWbQpdCby43gGDQUy2fxdqIcgb+DcVr0Kr"
CHIAVE_AUTOMAZIONE="AAAAC3NzaC1lZDI1NTE5AAAAICEUFRL7w2zTSj4CcUyPhcZmghTf3f3hcAeP47Thyajw"
AK=/root/.ssh/authorized_keys
RIGA_UV='. "$HOME/.local/bin/env"'

PERCORSI=(
  /work
  /root/.cache/uv /root/.cache/vllm /root/.cache/flashinfer
  /root/.triton /root/.nv /root/.humming /root/.tilelang
  /root/.config/uv /root/.config/fish /root/.config/vllm
  /root/.local/bin/uv /root/.local/bin/uvx /root/.local/bin/env /root/.local/bin/env.fish
  /root/.local/share/uv
  /root/setup_vm.sh /root/setup_vm.log /root/pull_runtime.log
  /root/dl_Qwen_Qwen3.5-4B.log /root/dl_Skywork_Skywork-OR1-Math-7B.log
  /root/dl_allenai_Olmo-3-7B-Think.log /root/dl_deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.log
  /root/dl_openai_gpt-oss-120b.log
)
IMMAGINI=(
  ghcr.io/aimo-interp/aimo-codabench-runtime:torch2.12.1-cuda13.2-cudnn9-transformers5.13.0-sklearn1.8.0-kernels0.15.2
  nvidia/cuda:12.8.1-base-ubuntu22.04
)

echo "== controlli di sicurezza"
for k in "$CHIAVE_DANIEL" "$CHIAVE_AUTOMAZIONE"; do
  case "$k" in
    AAAAC3NzaC1lZDI1NTE5*) ;;
    *) echo "STOP: chiavi pubbliche non inserite nello script."; exit 1 ;;
  esac
done
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "STOP: ci sono processi sulle GPU, la VM potrebbe essere gia' in uso da altri."; exit 1
fi
ATTESI="aimo backup_aimo_2026-09-17.tar.gz hf venv"
TROVATI="$( (ls -A /work 2>/dev/null || true) | LC_ALL=C sort | tr '\n' ' ' | sed 's/ $//')"
if [ -d /work ] && [ "$TROVATI" != "$ATTESI" ]; then
  echo "STOP: /work contiene voci inattese ($TROVATI): potrebbero non essere nostre."; exit 1
fi
N_TOTALI=$(grep -c . "$AK" || true)
N_NOSTRE=$(grep -c -F -e "$CHIAVE_DANIEL" -e "$CHIAVE_AUTOMAZIONE" "$AK" || true)
echo "  nessun processo sulle GPU"
echo "  /work contiene: ${TROVATI:-(non esiste)}"
echo "  authorized_keys: $N_TOTALI chiavi, di cui $N_NOSTRE nostre"

echo "== cosa verrebbe rimosso"
ESISTENTI=()
for p in "${PERCORSI[@]}"; do
  if [ -e "$p" ]; then
    ESISTENTI+=("$p")
    printf '  %8s  %s\n' "$(du -sh "$p" 2>/dev/null | cut -f1)" "$p"
  fi
done
if [ "${#ESISTENTI[@]}" -gt 0 ]; then
  echo "  totale file: $(du -sch "${ESISTENTI[@]}" 2>/dev/null | tail -1 | cut -f1)"
fi
for i in "${IMMAGINI[@]}"; do
  if docker image inspect "$i" >/dev/null 2>&1; then
    echo "  immagine docker: $i ($(docker image inspect --format '{{.Size}}' "$i" | awk '{printf "%.1f GB", $1/1e9}'))"
  fi
done
for f in /root/.bashrc /root/.profile /root/.zshrc; do
  if [ -f "$f" ] && grep -qxF "$RIGA_UV" "$f"; then echo "  riga dell'installer di uv in $f"; fi
done
echo "  ultimo passo: le $N_NOSTRE chiavi SSH nostre (quella degli organizzatori resta)"

if [ "$MODO" != "CONFERMO" ]; then
  echo "== SOLO ELENCO: nulla e' stato modificato. Per eseguire: bash /root/pulizia_vm.sh CONFERMO"
  exit 0
fi

echo "== rimozione"
for i in "${IMMAGINI[@]}"; do
  if docker image inspect "$i" >/dev/null 2>&1; then
    docker rmi "$i" >/dev/null 2>&1 && echo "  rimossa immagine $i" || echo "  NON rimossa immagine $i"
  fi
done
for p in "${ESISTENTI[@]}"; do
  rm -rf -- "$p" && echo "  rimosso $p"
done
rmdir /root/.local/bin 2>/dev/null && echo "  rimossa /root/.local/bin (vuota)" || true
for f in /root/.bashrc /root/.profile /root/.zshrc; do
  if [ -f "$f" ] && grep -qxF "$RIGA_UV" "$f"; then
    grep -vxF "$RIGA_UV" "$f" > "$f.nuovo" || true
    cat "$f.nuovo" > "$f"
    rm -f "$f.nuovo"
    echo "  tolta la riga di uv da $f"
  fi
done
if [ -f /root/.zshrc ] && ! grep -q '[^[:space:]]' /root/.zshrc; then
  rm -f /root/.zshrc && echo "  rimosso /root/.zshrc (conteneva solo la riga di uv)"
fi

echo "== chiavi SSH, ultimo passo"
grep -v -F -e "$CHIAVE_DANIEL" -e "$CHIAVE_AUTOMAZIONE" "$AK" > "$AK.nuovo" || true
N_DOPO=$(grep -c . "$AK.nuovo" || true)
if [ "$N_DOPO" -ge 1 ] && [ $((N_TOTALI - N_DOPO)) -eq "$N_NOSTRE" ]; then
  cat "$AK.nuovo" > "$AK"
  rm -f "$AK.nuovo"
  echo "  tolte $N_NOSTRE chiavi nostre; restano:"
  ssh-keygen -lf "$AK" | sed 's/^/    /'
else
  rm -f "$AK.nuovo"
  echo "  STOP: authorized_keys NON modificato (prima $N_TOTALI, dopo $N_DOPO, nostre $N_NOSTRE)"
  exit 1
fi

echo "== stato finale"
df -h / | tail -1 | awk '{print "  disco: " $3 " usati, " $4 " liberi"}'
[ -d /work ] && echo "  ATTENZIONE: /work esiste ancora" || echo "  /work non esiste piu'"
echo "  immagini docker rimaste: $(docker images -q | wc -l)"
rm -f -- "$0"
echo "== FATTO. Lo script si e' cancellato da solo. Da questo momento non abbiamo piu' accesso."
