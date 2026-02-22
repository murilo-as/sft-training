#!/usr/bin/env bash

#============================== Slurm ==========================================
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --nodelist=dgx-H100-02
#SBATCH --job-name=sft-train
#SBATCH --output=jobs/job.%j.out
#SBATCH --error=jobs/job.%j.err

set -euo pipefail

log(){ echo "[HOST $(date +'%F %T')] $*"; }
die(){ echo "[HOST $(date +'%F %T')] ERRO: $*" >&2; exit 1; }

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

if [ -f .env ]; then set -a; . ./.env; set +a; fi
CONFIG_FILE="${CONFIG_FILE:-config/config.yaml}"
[ -f "$CONFIG_FILE" ] || die "config nao encontrada: $CONFIG_FILE"
[ -f "data/train.jsonl" ] || die "data/train.jsonl nao encontrado"

OUT_DIR_HOST="$PROJECT_DIR/out"
mkdir -p "$OUT_DIR_HOST" jobs

# Apptainer image (ajuste o caminho se necessario)
APPT_IMAGE="${APPT_IMAGE:-/raid/user_muriloalves/images/pytorch-2.8.0-cu128-devel.sif}"
[[ -f "$APPT_IMAGE" ]] || die "Imagem SIF nao encontrada: $APPT_IMAGE"

# Caches para nao usar /tmp do no
export HF_HOME="${HF_HOME:-$OUT_DIR_HOST/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$OUT_DIR_HOST/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$OUT_DIR_HOST/pip}"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"

log "Projeto: $PROJECT_DIR"
log "Config: $CONFIG_FILE"
log "Dataset: data/train.jsonl (ok)"
log "OUT: $OUT_DIR_HOST"
log "Imagem (Apptainer): $APPT_IMAGE"
log "Subindo container com Apptainer..."

export CUDA_VISIBLE_DEVICES=0

srun apptainer exec --nv \
	-B "$PROJECT_DIR":/data \
	-B "$OUT_DIR_HOST":/data/out \
	"$APPT_IMAGE" \
	bash -lc "bash /data/scripts/in_container.sh /data/$CONFIG_FILE"
