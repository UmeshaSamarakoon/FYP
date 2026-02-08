#!/usr/bin/env bash
set -euo pipefail

DATASET_CSV=${DATASET_CSV:-data/processed/causal_multimodal_dataset.csv}

echo "==> Training embedding-aware CFN model"
python -m src.training.train_cfn \
  --dataset-csv "${DATASET_CSV}" \
  --use-embeddings
