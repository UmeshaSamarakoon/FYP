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

## Embedding-aware training (TCN + Wav2Vec2)
If you have generated embedding columns (`tcn_visual_emb`, `wav2vec_audio_emb`) in the CSV,
train the V2 CFN model with:

```
python -m src.training.train_cfn \
  --dataset-csv data/processed/causal_multimodal_dataset.csv \
  --use-embeddings \
  --epochs 30 \
  --batch-size 128 \
  --lr 1e-3 \
  --weight-decay 1e-4
```

This writes `models/cfn_emb.pth`. Enable it during inference by setting:
- `CFN_USE_EMBEDDINGS=true`
- `CFN_EMB_MODEL_PATH=models/cfn_emb.pth`

Optional SCM dependency score:
- `CFN_ENABLE_SCM_CHECKS=true`

## Deployment note (Render)
If you deploy the backend as a Render service, keep `runtime.txt` in this `backend/` directory so Render pins the Python version for the service root.


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
