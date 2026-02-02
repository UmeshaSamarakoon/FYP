import argparse
import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.modules.causal_fusion import CausalFusionNetwork
from sklearn.metrics import roc_auc_score, accuracy_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize(train, val):
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-6
    return (train - mean) / std, (val - mean) / std


def build_loaders(av_feats, phys_feats, labels, batch_size, val_split):
    n = len(labels)
    indices = np.random.permutation(n)
    split = int(n * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]

    X_av_train, X_av_val = av_feats[train_idx], av_feats[val_idx]
    X_phys_train, X_phys_val = phys_feats[train_idx], phys_feats[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    X_av_train, X_av_val = standardize(X_av_train, X_av_val)
    X_phys_train, X_phys_val = standardize(X_phys_train, X_phys_val)

    X_av_train = torch.tensor(X_av_train, dtype=torch.float32)
    X_phys_train = torch.tensor(X_phys_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_av_val = torch.tensor(X_av_val, dtype=torch.float32)
    X_phys_val = torch.tensor(X_phys_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_av_train, X_phys_train, y_train)
    val_ds = TensorDataset(X_av_val, X_phys_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def compute_pos_weight(labels):
    labels = labels.astype(np.float32)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 1.0
    return float(neg / pos)


def resolve_dataset_csv(args):
    if args.dataset_csv:
        return args.dataset_csv
    if args.data:
        print("⚠️  --data is deprecated; use --dataset-csv instead.")
        return args.data
    raise ValueError("No dataset CSV provided. Use --dataset-csv path/to.csv")


AV_FEATURES = [
    "lip_variance",
    "av_correlation",
    "av_lag_frames",
    "lip_mean",
    "lip_std",
    "lip_range",
    "lip_velocity_mean",
    "lip_velocity_std",
    "audio_rms_mean",
    "audio_rms_std",
    "mouth_flow_mean",
    "mouth_flow_std",
    "mouth_flow_max"
]

PHYS_FEATURES = [
    "jitter_mean",
    "jitter_std"
]


def build_feature_matrix(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = 0.0
        print(f"⚠️  Missing columns {missing}; filled with 0.0. Regenerate features for best accuracy.")
    return df[columns].values


def train(args):
    set_seed(args.seed)

    dataset_csv = resolve_dataset_csv(args)
    df = pd.read_csv(dataset_csv)
    av_feats = build_feature_matrix(df, AV_FEATURES)
    phys_feats = build_feature_matrix(df, PHYS_FEATURES)
    labels = df["label"].values.astype(np.float32)

    train_loader, val_loader = build_loaders(
        av_feats,
        phys_feats,
        labels,
        args.batch_size,
        args.val_split
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalFusionNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    pos_weight = compute_pos_weight(labels)

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for av, phys, label in train_loader:
            av = av.to(device)
            phys = phys.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                preds = model(av, phys)
                batch_weights = torch.where(label == 1, pos_weight, 1.0).to(device)
                loss = torch.nn.functional.binary_cross_entropy(preds, label, weight=batch_weights)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for av, phys, label in val_loader:
                av = av.to(device)
                phys = phys.to(device)
                label = label.to(device)
                preds = model(av, phys)
                loss = torch.nn.functional.binary_cross_entropy(preds, label)
                val_loss += loss.item()
                val_probs.append(preds.detach().cpu().numpy())
                val_targets.append(label.detach().cpu().numpy())

        avg_train = total_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))
        val_probs = np.concatenate(val_probs, axis=0).reshape(-1)
        val_targets = np.concatenate(val_targets, axis=0).reshape(-1)
        val_preds = (val_probs >= 0.5).astype(np.float32)
        try:
            val_auc = roc_auc_score(val_targets, val_probs)
        except ValueError:
            val_auc = float("nan")
        val_acc = accuracy_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1:02d} | train_loss={avg_train:.4f} "
            f"val_loss={avg_val:.4f} val_acc={val_acc:.3f} val_auc={val_auc:.3f}"
        )

        if avg_val < best_val - args.min_delta:
            best_val = avg_val
            patience_counter = 0
            os.makedirs(args.model_dir, exist_ok=True)
            model_path = os.path.join(args.model_dir, args.model_name)
            torch.save(model.state_dict(), model_path)
            print(f"✔ Saved best model to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    print("Learned alpha (AV causal weight):", model.alpha.item())
    print("Learned beta (Physical causal weight):", model.beta.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CFN model with efficient techniques.")
    parser.add_argument("--dataset-csv", default="data/processed/causal_multimodal_dataset.csv")
    parser.add_argument("--data", default=None, help="Deprecated alias for --dataset-csv.")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--model-name", default="cfn.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
