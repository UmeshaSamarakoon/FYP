#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH=.
export CFN_USE_EMBEDDINGS=true
export CFN_W2V2_MODEL="${CFN_W2V2_MODEL:-WAV2VEC2_BASE}"
export CFN_VISUAL_TCN_PATH="${CFN_VISUAL_TCN_PATH:-$ROOT_DIR/models/visual_tcn.pth}"
export MEDIAPIPE_DISABLE_GPU=1
export CUDA_VISIBLE_DEVICES=-1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

DATA_PATH="${1:-data/processed/causal_multimodal_embeddings_train.csv}"
TRAIN_SOURCE="${TRAIN_SOURCE:-fakeavceleb}"
OUT_DIR="$ROOT_DIR/models/matrix_runs"
mkdir -p "$OUT_DIR"

run_one() {
  local run_id="$1"
  local epochs="$2"
  local patience="$3"
  local causal_weight="$4"
  local batch_size="$5"

  echo "========== RUN ${run_id} =========="
  echo "epochs=${epochs} patience=${patience} causal_weight=${causal_weight} batch=${batch_size}"

  .venv/bin/python -m src.training.train_cfn \
    --data "$DATA_PATH" \
    --train-source "$TRAIN_SOURCE" \
    --use-scaler \
    --use-embeddings \
    --epochs "$epochs" \
    --patience "$patience" \
    --batch-size "$batch_size" \
    --causal-weight "$causal_weight" \
    2>&1 | tee "$OUT_DIR/train_${run_id}.log"

  cp models/cfn_emb.pth "$OUT_DIR/cfn_emb_${run_id}.pth"
  cp models/cfn_scaler.pkl "$OUT_DIR/cfn_scaler_${run_id}.pkl"
  echo "Saved: $OUT_DIR/cfn_emb_${run_id}.pth"
  echo "Saved: $OUT_DIR/cfn_scaler_${run_id}.pkl"
}

run_one A 25 6 0.05 128
run_one B 30 8 0.10 128
run_one C 35 10 0.20 128
run_one D 40 10 0.30 128

echo "Matrix complete. Artifacts in: $OUT_DIR"
