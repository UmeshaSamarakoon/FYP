#!/usr/bin/env bash
set -euo pipefail

DATASET_CSV=${DATASET_CSV:-data/processed/causal_multimodal_dataset.csv}
BALANCED_CSV=${BALANCED_CSV:-data/processed/causal_multimodal_dataset_balanced.csv}
TRIM_FAKEAV_AUDIO_HEAD=${TRIM_FAKEAV_AUDIO_HEAD:-false}
FAKEAV_ROOT=${FAKEAV_ROOT:-data/raw/fakeavceleb}
FAKEAV_TRIM_SECONDS=${FAKEAV_TRIM_SECONDS:-0.10}
FAKEAV_TRIM_MANIFEST=${FAKEAV_TRIM_MANIFEST:-data/processed/fakeav_audio_trim_manifest.csv}

if [[ "${TRIM_FAKEAV_AUDIO_HEAD}" == "true" ]]; then
  echo "==> Trimming first ${FAKEAV_TRIM_SECONDS}s of FakeAVCeleb audio (in-place) from ${FAKEAV_ROOT}"
  python scripts/trim_fakeav_audio_head.py \
    --input-root "${FAKEAV_ROOT}" \
    --in-place \
    --trim-seconds "${FAKEAV_TRIM_SECONDS}" \
    --manifest-csv "${FAKEAV_TRIM_MANIFEST}"
fi

echo "==> Extracting causal features to ${DATASET_CSV}"
echo "==> Dataset root: FAKEAV_ROOT=${FAKEAV_ROOT}"
python -m src.preprocessing.batch_feature_extractor

echo "==> (Optional) Balancing dataset to ${BALANCED_CSV}"
python -m src.preprocessing.balance_dataset \
  --input-csv "${DATASET_CSV}" \
  --output-csv "${BALANCED_CSV}"

echo "==> Training CFN model on balanced dataset"
python -m src.training.train_cfn \
  --dataset-csv "${BALANCED_CSV}"
