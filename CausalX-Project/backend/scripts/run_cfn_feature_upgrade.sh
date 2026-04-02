#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH=.
export MEDIAPIPE_DISABLE_GPU=1
export CUDA_VISIBLE_DEVICES=-1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

if [[ -z "${DATA_PATH:-}" ]]; then
  if [[ -f "data/processed/causal_multimodal_dataset_fakeav_fulltrain_train.csv" ]]; then
    DATA_PATH="data/processed/causal_multimodal_dataset_fakeav_fulltrain_train.csv"
  else
    DATA_PATH="data/processed/causal_multimodal_dataset.csv"
  fi
fi
HARD_NEG_FILE="${HARD_NEG_FILE:-data/processed/hard_negatives_fakeav.tsv}"
RUN_TAG="${RUN_TAG:-featup_$(date '+%Y%m%d_%H%M%S')}"
MODEL_DIR="${MODEL_DIR:-models/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-models/experiment_logs}"
LOG_PATH="${LOG_DIR}/train_${RUN_TAG}.log"

EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-42}"
CAUSAL_WEIGHT="${CAUSAL_WEIGHT:-0.15}"
HARD_NEG_WEIGHT="${HARD_NEG_WEIGHT:-6.0}"
FOCAL_ALPHA="${FOCAL_ALPHA:-0.75}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
TRAIN_SOURCE="${TRAIN_SOURCE:-fakeavceleb}"
FEATURE_PROFILE="${FEATURE_PROFILE:-auto}"
SELECTION_METRIC="${SELECTION_METRIC:-hybrid_robust}"
SELECTION_THRESHOLD_MODE="${SELECTION_THRESHOLD_MODE:-best_f1}"
SELECTION_THRESHOLD="${SELECTION_THRESHOLD:-0.5}"
CLASS_BALANCE_MODE="${CLASS_BALANCE_MODE:-none}"
WEIGHT_APPLICATION="${WEIGHT_APPLICATION:-auto}"
MIN_DOMAIN_SPEC="${MIN_DOMAIN_SPEC:-0.05}"
MIN_DOMAIN_REC="${MIN_DOMAIN_REC:-0.20}"
TARGET_ACC="${TARGET_ACC:-}"
TARGET_PRECISION="${TARGET_PRECISION:-}"
TARGET_RECALL="${TARGET_RECALL:-}"
TARGET_F1="${TARGET_F1:-}"
TARGET_PRIORITY="${TARGET_PRIORITY:-f1}"

mkdir -p "$MODEL_DIR" "$LOG_DIR"

cmd=(
  .venv/bin/python -m src.training.train_cfn
  --data "$DATA_PATH"
  --model-dir "$MODEL_DIR"
  --train-source "$TRAIN_SOURCE"
  --feature-profile "$FEATURE_PROFILE"
  --class-balance-mode "$CLASS_BALANCE_MODE"
  --weight-application "$WEIGHT_APPLICATION"
  --use-embeddings
  --use-scaler
  --group-balance
  --use-weighted-sampler
  --loss focal
  --focal-alpha "$FOCAL_ALPHA"
  --focal-gamma "$FOCAL_GAMMA"
  --causal-weight "$CAUSAL_WEIGHT"
  --scheduler cosine
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --seed "$SEED"
  --selection-metric "$SELECTION_METRIC"
  --selection-threshold-mode "$SELECTION_THRESHOLD_MODE"
  --selection-threshold "$SELECTION_THRESHOLD"
  --min-domain-spec "$MIN_DOMAIN_SPEC"
  --min-domain-rec "$MIN_DOMAIN_REC"
)

if [[ -f "$HARD_NEG_FILE" ]]; then
  cmd+=(
    --hard-negative-file "$HARD_NEG_FILE"
    --hard-negative-weight "$HARD_NEG_WEIGHT"
  )
else
  echo "Warning: hard-negative file not found: $HARD_NEG_FILE"
fi

if [[ -n "$TARGET_ACC" ]]; then
  cmd+=(--target-acc "$TARGET_ACC")
fi
if [[ -n "$TARGET_PRECISION" ]]; then
  cmd+=(--target-precision "$TARGET_PRECISION")
fi
if [[ -n "$TARGET_RECALL" ]]; then
  cmd+=(--target-recall "$TARGET_RECALL")
fi
if [[ -n "$TARGET_F1" ]]; then
  cmd+=(--target-f1 "$TARGET_F1")
fi
if [[ "$SELECTION_THRESHOLD_MODE" == "target" ]]; then
  cmd+=(--target-priority "$TARGET_PRIORITY")
fi

echo "Run tag: $RUN_TAG"
echo "Model dir: $MODEL_DIR"
echo "Log file: $LOG_PATH"
echo "Data: $DATA_PATH"
echo "Feature profile: $FEATURE_PROFILE"

"${cmd[@]}" 2>&1 | tee "$LOG_PATH"

best_auc="$(rg -o 'Best AUC: [0-9.]+' "$LOG_PATH" | tail -n1 | awk '{print $3}' || true)"
best_sel="$(rg -o 'Best selection score: [-0-9.]+' "$LOG_PATH" | tail -n1 | awk '{print $4}' || true)"

RUN_CSV="$LOG_DIR/feature_upgrade_runs.csv"
if [[ ! -f "$RUN_CSV" ]]; then
  echo "timestamp_utc,run_tag,model_dir,log_path,best_auc,best_selection_score,data_path,feature_profile" > "$RUN_CSV"
fi
echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'),$RUN_TAG,$MODEL_DIR,$LOG_PATH,${best_auc:-},${best_sel:-},$DATA_PATH,$FEATURE_PROFILE" >> "$RUN_CSV"

echo "Run complete."
echo "Artifacts:"
echo "  $MODEL_DIR/cfn_emb.pth"
echo "  $MODEL_DIR/cfn_scaler.pkl"
echo "  $MODEL_DIR/cfn_threshold_report.json"
echo "Tracking CSV: $RUN_CSV"
