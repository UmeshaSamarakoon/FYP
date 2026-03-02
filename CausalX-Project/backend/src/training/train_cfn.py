import argparse
import os
import random

import joblib
import numpy as np
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.modules.causal_fusion import CausalFusionNetwork, CausalFusionNetworkV2
from sklearn.metrics import roc_auc_score, accuracy_score

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(X_av, X_phys, y, weights, batch_size, shuffle=True):
    if weights is None:
        dataset = TensorDataset(X_av, X_phys, y)
    else:
        dataset = TensorDataset(X_av, X_phys, y, weights)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def focal_loss_probs(preds, labels, alpha=0.75, gamma=2.0, eps=1e-7):
    preds = torch.clamp(preds, eps, 1.0 - eps)
    pt = torch.where(labels == 1, preds, 1.0 - preds)
    alpha_t = torch.where(labels == 1, torch.full_like(labels, alpha), torch.full_like(labels, 1.0 - alpha))
    return -alpha_t * ((1.0 - pt) ** gamma) * torch.log(pt)


def _compute_cls_loss(preds, label, criterion, loss_type="bce", focal_alpha=0.75, focal_gamma=2.0):
    if loss_type == "focal":
        return focal_loss_probs(preds, label, alpha=focal_alpha, gamma=focal_gamma)
    return criterion(preds, label)


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    use_weights=False,
    use_causal=False,
    causal_weight=0.0,
    loss_type="bce",
    focal_alpha=0.75,
    focal_gamma=2.0,
):
    model.train()
    total_loss = 0.0
    for batch in loader:
        if use_weights:
            av, phys, label, weight = batch
            weight = weight.to(device)
        else:
            av, phys, label = batch
        av = av.to(device)
        phys = phys.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        if use_causal:
            probs, av_h, phys_h = model.branch_outputs(av, phys)
            cls_loss = _compute_cls_loss(
                probs,
                label,
                criterion,
                loss_type=loss_type,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
            )
            if use_weights:
                cls_loss = (cls_loss * weight).mean()
            elif cls_loss.dim() > 0:
                cls_loss = cls_loss.mean()
            causal_loss = model.causal_penalty(av_h, phys_h)
            loss = cls_loss + causal_weight * causal_loss
        else:
            probs = model(av, phys)
            if use_weights:
                raw_loss = _compute_cls_loss(
                    probs,
                    label,
                    criterion,
                    loss_type=loss_type,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                )
                loss = (raw_loss * weight).mean()
            else:
                loss = _compute_cls_loss(
                    probs,
                    label,
                    criterion,
                    loss_type=loss_type,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                )
                if loss.dim() > 0:
                    loss = loss.mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def eval_epoch(
    model,
    loader,
    criterion,
    device,
    use_causal=False,
    loss_type="bce",
    focal_alpha=0.75,
    focal_gamma=2.0,
):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for av, phys, label in loader:
            av = av.to(device)
            phys = phys.to(device)
            label = label.to(device)
            if use_causal:
                preds, _, _ = model.branch_outputs(av, phys)
            else:
                preds = model(av, phys)
            loss = _compute_cls_loss(
                preds,
                label,
                criterion,
                loss_type=loss_type,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
            )
            if loss.dim() > 0:
                loss = loss.mean()
            total_loss += loss.item()
            all_preds.extend(preds.squeeze(1).cpu().numpy().tolist())
            all_labels.extend(label.squeeze(1).cpu().numpy().tolist())

    if not all_labels:
        return 0.0, 0.0, 0.0, [], []

    acc = accuracy_score(all_labels, [1 if p >= 0.5 else 0 for p in all_preds])
    try:
        auc_val = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc_val = 0.0
    return total_loss / max(len(loader), 1), acc, auc_val, all_labels, all_preds


def threshold_sweep(labels, probs):
    """
    Sweep thresholds to find best F1 and report PR AUC.
    """
    labels = np.array(labels)
    probs = np.array(probs)
    if probs.size == 0 or labels.size == 0:
        return {"pr_auc": 0.0, "best_f1": 0.0, "best_thr": 0.5}

    prec, rec, thr = precision_recall_curve(labels, probs)
    pr_auc = auc(rec, prec)

    best_f1, best_thr = 0.0, 0.5
    thresholds = np.linspace(probs.min(), probs.max(), 50) if probs.size else [0.5]
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    return {"pr_auc": pr_auc, "best_f1": best_f1, "best_thr": best_thr}


def filter_training_sources(df, train_source):
    """
    Keep only requested data source(s) for training.
    """
    source = (train_source or "all").strip().lower()
    if source == "all":
        return df

    if source != "fakeavceleb":
        raise ValueError(f"Unsupported --train-source value: {train_source}")

    if "dataset" in df.columns:
        mask = df["dataset"].astype(str).str.lower().eq("fakeavceleb")
        return df.loc[mask].copy()

    if "path" in df.columns:
        mask = df["path"].astype(str).str.lower().str.contains("fakeavceleb", regex=False)
        return df.loc[mask].copy()

    if "video_id" in df.columns:
        # Conservative fallback: keep rows whose video_id explicitly mentions fakeavceleb.
        mask = df["video_id"].astype(str).str.lower().str.contains("fakeavceleb", regex=False)
        filtered = df.loc[mask].copy()
        if not filtered.empty:
            return filtered

    raise RuntimeError(
        "Could not isolate FakeAVCeleb rows: missing 'dataset' or 'path' hints in input CSV."
    )


def main():
    parser = argparse.ArgumentParser(description="Train Causal Fusion Network (CFN).")
    parser.add_argument(
        "--data",
        "--dataset-csv",
        dest="data",
        default="data/processed/causal_multimodal_dataset.csv",
        help="Path to the dataset CSV (default: data/processed/causal_multimodal_dataset.csv)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--use-scaler", action="store_true")
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Include TCN/Wav2Vec2 embedding columns and train CFN V2.",
    )
    parser.add_argument(
        "--causal-weight",
        type=float,
        default=0.0,
        help="Weight for SCM-inspired consistency penalty (0 to disable).",
    )
    parser.add_argument(
        "--scheduler",
        choices=["plateau", "cosine"],
        default="cosine",
        help="Learning rate scheduler to use (default: cosine).",
    )
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--train-source",
        choices=["fakeavceleb", "all"],
        default="fakeavceleb",
        help="Training source filter. Default keeps only FakeAVCeleb rows.",
    )
    args = parser.parse_args()

    set_seed(SEED)

    df = pd.read_csv(args.data)
    before_rows = len(df)
    df = filter_training_sources(df, args.train_source)
    after_rows = len(df)
    if after_rows == 0:
        raise RuntimeError("No rows available after source filtering.")
    print(
        f"Loaded {before_rows} rows from {args.data}; "
        f"using {after_rows} rows after --train-source={args.train_source} filter."
    )

    av_feature_cols = ["lip_variance", "av_correlation", "av_lag_frames"]
    if args.use_embeddings:
        av_feature_cols.extend(["tcn_visual_emb", "wav2vec_audio_emb"])
    av_feats = df[av_feature_cols].values
    phys_feats = df[["jitter_mean", "jitter_std"]].values
    labels = df["label"].values

    X_av_train, X_av_val, X_phys_train, X_phys_val, y_train, y_val = train_test_split(
        av_feats,
        phys_feats,
        labels,
        test_size=args.val_split,
        random_state=SEED,
        stratify=labels if len(np.unique(labels)) > 1 else None
    )

    scaler = None
    if args.use_scaler:
        scaler = {
            "av": StandardScaler().fit(X_av_train),
            "phys": StandardScaler().fit(X_phys_train)
        }
        X_av_train = scaler["av"].transform(X_av_train)
        X_av_val = scaler["av"].transform(X_av_val)
        X_phys_train = scaler["phys"].transform(X_phys_train)
        X_phys_val = scaler["phys"].transform(X_phys_val)

    X_av_train = torch.tensor(X_av_train, dtype=torch.float32)
    X_phys_train = torch.tensor(X_phys_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_av_val = torch.tensor(X_av_val, dtype=torch.float32)
    X_phys_val = torch.tensor(X_phys_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    sample_weights = None
    use_weights = False
    if len(np.unique(labels)) > 1:
        class_counts = np.bincount(labels.astype(int))
        class_weights = class_counts.sum() / np.maximum(class_counts, 1)
        sample_weights = torch.tensor(
            [class_weights[int(l)] for l in y_train.squeeze(1).numpy()],
            dtype=torch.float32
        ).unsqueeze(1)
        use_weights = True

    train_loader = build_loaders(
        X_av_train,
        X_phys_train,
        y_train,
        sample_weights,
        args.batch_size,
        shuffle=True
    )
    val_loader = build_loaders(
        X_av_val,
        X_phys_val,
        y_val,
        None,
        args.batch_size,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_embeddings:
        model = CausalFusionNetworkV2(av_dim=X_av_train.shape[1], phys_dim=X_phys_train.shape[1]).to(device)
    else:
        model = CausalFusionNetwork().to(device)

    if use_weights:
        criterion = torch.nn.BCELoss(reduction="none")
    else:
        criterion = torch.nn.BCELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_auc = -1.0
    epochs_no_improve = 0

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "cfn_emb.pth" if args.use_embeddings else "cfn.pth")
    scaler_path = os.path.join(args.model_dir, "cfn_scaler.pkl")

    for epoch in range(args.epochs):
        use_causal = args.causal_weight > 0
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_weights=use_weights,
            use_causal=use_causal,
            causal_weight=args.causal_weight,
            loss_type=args.loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
        )
        val_loss, val_acc, val_auc, val_labels, val_probs = eval_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_causal=use_causal,
            loss_type=args.loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
        )
        sweep = threshold_sweep(val_labels, val_probs)
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_auc)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} "
            f"val_auc={val_auc:.3f} "
            f"pr_auc={sweep['pr_auc']:.3f} "
            f"best_f1={sweep['best_f1']:.3f} "
            f"best_thr={sweep['best_thr']:.3f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            if scaler is not None:
                joblib.dump(scaler, scaler_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    print("Best AUC:", best_auc)
    print("Learned alpha (AV causal weight):", model.alpha.item())
    print("Learned beta (Physical causal weight):", model.beta.item())
    if args.causal_weight > 0:
        # Report average causal penalty on the validation set
        model.eval()
        with torch.no_grad():
            _, av_h, phys_h = model.branch_outputs(X_av_val.to(device), X_phys_val.to(device))
            causal_pen = model.causal_penalty(av_h, phys_h).item()
        print("Validation causal penalty:", causal_pen)
    print(f"✔ CFN model saved to {model_path}")
    if scaler is not None:
        print(f"✔ Scaler saved to {scaler_path}")


if __name__ == "__main__":
    main()
