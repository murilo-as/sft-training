#!/usr/bin/env bash
set -euo pipefail

log(){ echo "[CONTAINER $(date +'%F %T')] $*"; }
die(){ echo "[CONTAINER $(date +'%F %T')] ERRO: $*" >&2; exit 1; }

CFG="${1:-/data/config/config.yaml}"
[ -f "$CFG" ] || die "config nao encontrada no container: $CFG"

cd /data

OUT_ROOT=/data/out
mkdir -p "$OUT_ROOT"

export HF_HOME="$OUT_ROOT/cache/hf"
export TRITON_CACHE_DIR="$OUT_ROOT/cache/triton"
export OFFLOAD_DIR="$OUT_ROOT/offload"
export WANDB_DIR="$OUT_ROOT/logs/wandb"
export PIP_CACHE_DIR="$OUT_ROOT/cache/pip"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$OFFLOAD_DIR" "$WANDB_DIR" "$PIP_CACHE_DIR"

export PYTHONPATH="/data:${PYTHONPATH:-}"
export WANDB_MODE=online
[ -z "${WANDB_BASE_URL:-}" ] && unset WANDB_BASE_URL

if [ -n "${SSL_CERT_FILE:-}" ] && [ ! -f "$SSL_CERT_FILE" ]; then
  log "Aviso: SSL_CERT_FILE definido mas arquivo nao existe: $SSL_CERT_FILE - unsetando"
  unset SSL_CERT_FILE
fi
if [ -n "${SSL_CERT_DIR:-}" ] && [ ! -d "$SSL_CERT_DIR" ]; then
  log "Aviso: SSL_CERT_DIR definido mas diretorio nao existe: $SSL_CERT_DIR - unsetando"
  unset SSL_CERT_DIR
fi
if [ -n "${REQUESTS_CA_BUNDLE:-}" ] && [ ! -f "$REQUESTS_CA_BUNDLE" ]; then
  log "Aviso: REQUESTS_CA_BUNDLE definido mas arquivo nao existe: $REQUESTS_CA_BUNDLE - unsetando"
  unset REQUESTS_CA_BUNDLE
fi

# Fallback para bundle de CA do sistema, se existir
if [ -z "${SSL_CERT_FILE:-}" ] && [ -f /etc/ssl/certs/ca-certificates.crt ]; then
  export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
fi

VENV="$OUT_ROOT/env/.venv"
if [ ! -f "$VENV/bin/activate" ]; then
  log "Criando venv: $VENV"
  python -m venv --system-site-packages "$VENV" || true
fi
. "$VENV/bin/activate"
python -m pip install -U pip wheel setuptools

REQ=/data/requirements.txt
META="$OUT_ROOT/env/.req.sha256"
if [ -f "$REQ" ]; then
  CUR=$(sha256sum "$REQ" | awk '{print $1}')
  if [ ! -f "$META" ] || [ "$(cat "$META")" != "$CUR" ]; then
    log "Instalando requirements (hash mudou)"
    pip install -r "$REQ"
    echo -n "$CUR" > "$META"
  else
    log "Requirements sem mudancas - pulando pip install"
  fi
else
  log "requirements.txt nao encontrado - seguindo sem instalar deps"
fi

python - <<'PY' || true
def ver(m):
    try:
        mod = __import__(m)
        print(f"{m}={getattr(mod,'__version__','n/a')}")
    except Exception:
        print(f"{m}=MISSING")
for m in ("torch","transformers","trl","peft","datasets","accelerate"):
    ver(m)
PY

# Preprocessamento dos dados
TRAIN_JSON="data/train.json"
TRAIN_PROCESSED="data/train_processed.jsonl"

if [ -f "$TRAIN_JSON" ]; then
  # Verifica se precisa reprocessar (se o processed não existe ou se o JSON é mais novo)
  if [ ! -f "$TRAIN_PROCESSED" ] || [ "$TRAIN_JSON" -nt "$TRAIN_PROCESSED" ]; then
    log "Executando preprocessamento dos dados..."
    python -u preprocess.py || die "Falha no preprocessamento"
  else
    log "Dados já preprocessados - pulando etapa de preprocessamento"
  fi
else
  log "AVISO: $TRAIN_JSON não encontrado - tentando usar dados já processados"
fi

log "Treino iniciado"
log "CFG=$CFG"
log "OUT_ROOT=$OUT_ROOT | HF_HOME=$HF_HOME | WANDB_DIR=$WANDB_DIR | OFFLOAD_DIR=$OFFLOAD_DIR"
exec python -u train_sft.py
