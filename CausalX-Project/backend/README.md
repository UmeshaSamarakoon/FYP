# CausalX

## Retraining the CFN model (recommended for higher accuracy)

The inference pipeline uses a pretrained model from `backend/models/cfn.pth`. For better accuracy on your dataset, retrain using the feature CSV produced by the preprocessing scripts.

### 1) Prepare the dataset
Generate the feature dataset:

```
python -m src.preprocessing.batch_feature_extractor
```

This writes `data/processed/causal_multimodal_dataset.csv`.

If you update feature extraction (e.g., new lip/audio statistics), rerun this step to refresh the CSV.

### 2) Train with efficient techniques
Run the training script (uses mixed precision on GPU, cosine LR, early stopping, feature standardization, and class imbalance weighting):

```
python -m src.training.train_cfn \
  --dataset-csv data/processed/causal_multimodal_dataset.csv \
  --epochs 30 \
  --batch-size 128 \
  --lr 1e-3 \
  --weight-decay 1e-4
```

The best checkpoint is saved to `backend/models/cfn.pth` by default.

> Note: `--data` is supported as a deprecated alias for `--dataset-csv` if you see older docs or scripts.

### 2a) (Optional) Balance the dataset (Option A)
If your dataset is heavily imbalanced, downsample to the smallest class before training:

```
python -m src.preprocessing.balance_dataset \
  --input-csv data/processed/causal_multimodal_dataset.csv \
  --output-csv data/processed/causal_multimodal_dataset_balanced.csv
```

Then train using the balanced CSV:

```
python -m src.training.train_cfn \
  --dataset-csv data/processed/causal_multimodal_dataset_balanced.csv
```

### 3) Use the new model
Restart the API server so it reloads the updated weights.
