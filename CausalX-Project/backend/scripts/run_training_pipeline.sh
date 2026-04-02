#!/usr/bin/env bash
set -euo pipefail

DATASET_CSV=${DATASET_CSV:-data/processed/causal_multimodal_dataset.csv}
BALANCED_CSV=${BALANCED_CSV:-data/processed/causal_multimodal_dataset_balanced.csv}

echo "==> Extracting causal features to ${DATASET_CSV}"
python -m src.preprocessing.batch_feature_extractor

echo "==> (Optional) Balancing dataset to ${BALANCED_CSV}"
python -m src.preprocessing.balance_dataset \
  --input-csv "${DATASET_CSV}" \
  --output-csv "${BALANCED_CSV}"

echo "==> Training CFN model on balanced dataset"
python -m src.training.train_cfn \
  --dataset-csv "${BALANCED_CSV}"
