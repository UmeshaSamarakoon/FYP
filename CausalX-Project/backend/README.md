# CausalX

## Retraining the CFN model (recommended for accuracy)

This project ships with a pretrained CFN model (`models/cfn.pth`). To improve accuracy on your own dataset, retrain using the processed CSV features.

### 1) Prepare the dataset
Generate the feature CSV using the preprocessing pipeline (adjust paths as needed):

```bash
python -m src.preprocessing.batch_feature_extractor
```

This produces a CSV similar to:

```
data/processed/causal_multimodal_dataset.csv
```

### 2) Retrain with efficient defaults
Run the training script with early stopping, stratified split, scaling, and class imbalance handling:

```bash
python -m src.training.train_cfn \
  --data data/processed/causal_multimodal_dataset.csv \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 5 \
  --val-split 0.2 \
  --use-scaler
```

### 3) Use the new model in inference
The training script writes:
- `models/cfn.pth` (trained weights)
- `models/cfn_scaler.pkl` (feature scaler, used automatically if present)

No code changes are required; the inference pipeline automatically loads the scaler if found.
