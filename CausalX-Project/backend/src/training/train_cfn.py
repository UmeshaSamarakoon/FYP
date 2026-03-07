import argparse
import os
import random
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.modules.causal_fusion import CausalFusionNetwork, CausalFusionNetworkV2

DEFAULT_SEED = 42


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


def infer_domain_labels(df: pd.DataFrame):
    if "dataset" in df.columns:
        return df["dataset"].astype(str).str.lower().fillna("unknown").to_numpy()
    if "path" in df.columns:
        path_s = df["path"].astype(str).str.lower()
        dom = np.full(len(df), "unknown", dtype=object)
        dom[path_s.str.contains("fakeavceleb", regex=False).to_numpy()] = "fakeavceleb"
        dom[path_s.str.contains("dfdc", regex=False).to_numpy()] = "dfdc"
        return dom
    return np.array(["unknown"] * len(df), dtype=object)


def _binary_confusion(labels, preds):
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    rec = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    bal = 0.5 * (rec + spec)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acc": acc,
        "rec": rec,
        "spec": spec,
        "prec": prec,
        "bal_acc": bal,
        "f1": f1,
    }


def compute_domain_metrics(labels, probs, domains, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    domains = np.asarray(domains, dtype=object)
    preds = (probs >= float(threshold)).astype(int)

    overall = _binary_confusion(labels, preds)
    per_domain = {}
    for d in sorted(set(domains.tolist())):
        mask = domains == d
        if not np.any(mask):
            continue
        per_domain[str(d)] = _binary_confusion(labels[mask], preds[mask])

    if per_domain:
        bal_list = [m["bal_acc"] for m in per_domain.values()]
        spec_list = [m["spec"] for m in per_domain.values()]
        rec_list = [m["rec"] for m in per_domain.values()]
        macro_bal = float(np.mean(bal_list))
        worst_bal = float(np.min(bal_list))
        worst_spec = float(np.min(spec_list))
        worst_rec = float(np.min(rec_list))
    else:
        macro_bal = worst_bal = overall["bal_acc"]
        worst_spec = overall["spec"]
        worst_rec = overall["rec"]

    return {
        "overall": overall,
        "per_domain": per_domain,
        "macro_bal_acc": macro_bal,
        "worst_bal_acc": worst_bal,
        "worst_spec": worst_spec,
        "worst_rec": worst_rec,
    }


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


def _norm_path(v):
    try:
        return Path(str(v)).expanduser().as_posix().lower()
    except Exception:
        return str(v).strip().lower()


def _norm_name(v):
    try:
        return Path(str(v)).name.lower()
    except Exception:
        return str(v).strip().lower()


def load_hard_negative_paths(path: str):
    """
    Load hard negatives from jsonl/csv/tsv and return a normalized path set.
    Expected rows typically contain at least: path, label, pred.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Hard-negative file not found: {path}")

    paths = set()
    raw_lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if p.suffix.lower() == ".jsonl":
        for raw in p.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            row_path = row.get("path")
            if not row_path:
                continue
            label = row.get("label")
            pred = row.get("pred")
            # Prefer true hard negatives: real label predicted fake.
            if label is not None and pred is not None:
                if int(label) == 0 and int(pred) == 1:
                    paths.add(_norm_path(row_path))
            else:
                paths.add(_norm_path(row_path))
    else:
        # Common notebook output: plain text file with one absolute path per line.
        # Avoid delimiter sniffing because video paths can contain punctuation/spaces.
        if raw_lines and all(("\t" not in ln and "," not in ln) for ln in raw_lines[:32]):
            return set(_norm_path(ln) for ln in raw_lines)

        # First try regular CSV/TSV with headers.
        try:
            df = pd.read_csv(p, sep=None, engine="python", on_bad_lines="skip")
        except Exception:
            df = None
        if df is not None and "path" in df.columns:
            if "label" in df.columns and "pred" in df.columns:
                df = df[(df["label"].astype(int) == 0) & (df["pred"].astype(int) == 1)]
            paths = set(df["path"].astype(str).map(_norm_path).tolist())
            return paths

        # Fallback for headerless TSV/CSV produced by notebook scripts:
        # path<TAB>label<TAB>pred  (or path,label,pred)
        for line in raw_lines:
            if "\t" in line:
                parts = line.rsplit("\t", 2)
            else:
                parts = line.rsplit(",", 2)

            if not parts:
                continue
            row_path = parts[0].strip()
            if not row_path or row_path.lower() == "path":
                continue

            if len(parts) >= 3:
                try:
                    label = int(parts[1].strip())
                    pred = int(parts[2].strip())
                except Exception:
                    paths.add(_norm_path(row_path))
                    continue
                if label == 0 and pred == 1:
                    paths.add(_norm_path(row_path))
            else:
                paths.add(_norm_path(row_path))

    return paths


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
    parser.add_argument(
        "--group-balance",
        action="store_true",
        help="Apply domain+label inverse-frequency sample weighting when dataset column is available.",
    )
    parser.add_argument(
        "--use-weighted-sampler",
        action="store_true",
        help="Use WeightedRandomSampler for train batches (uses computed sample weights).",
    )
    parser.add_argument(
        "--hard-negative-file",
        type=str,
        default=None,
        help="Optional jsonl/csv/tsv with hard negatives to upweight (real predicted fake).",
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=2.0,
        help="Multiplicative weight for matched hard-negative training rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for split and training reproducibility.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="robust_bal",
        choices=["auc", "robust_bal", "hybrid_robust"],
        help="Metric used for best-checkpoint selection (default: robust_bal).",
    )
    parser.add_argument(
        "--selection-threshold",
        type=float,
        default=0.5,
        help="Decision threshold used for robustness metrics during selection.",
    )
    parser.add_argument(
        "--selection-threshold-mode",
        type=str,
        default="fixed",
        choices=["fixed", "best_f1"],
        help=(
            "Threshold mode for robustness metrics/checkpoint constraints: "
            "'fixed' uses --selection-threshold, 'best_f1' uses per-epoch swept best_thr."
        ),
    )
    parser.add_argument(
        "--min-domain-spec",
        type=float,
        default=0.0,
        help="Minimum worst-domain specificity required to accept a checkpoint.",
    )
    parser.add_argument(
        "--min-domain-rec",
        type=float,
        default=0.0,
        help="Minimum worst-domain recall required to accept a checkpoint.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Using seed: {args.seed}")

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
    domain_labels = infer_domain_labels(df)
    pos_rate = float(np.mean(labels.astype(float)))
    print(f"Label prevalence: pos={pos_rate:.4f}, neg={1.0 - pos_rate:.4f}")
    if args.selection_threshold_mode == "best_f1" and (pos_rate > 0.90 or pos_rate < 0.10):
        print(
            "Warning: --selection-threshold-mode=best_f1 with extreme class skew can "
            "collapse specificity/recall trade-offs. Consider mode='fixed'."
        )

    all_idx = np.arange(len(df))
    stratify_target = labels if len(np.unique(labels)) > 1 else None
    domain_label_combo = np.array([f"{d}:{int(y)}" for d, y in zip(domain_labels, labels)], dtype=object)
    combo_counts = pd.Series(domain_label_combo).value_counts()
    if len(combo_counts) > 1 and int(combo_counts.min()) >= 2:
        stratify_target = domain_label_combo
        print(f"Using dataset+label stratified split across {len(combo_counts)} groups.")
    elif stratify_target is not None:
        print("Using label-only stratified split.")
    else:
        print("Split has no stratification (single class).")

    idx_train, idx_val = train_test_split(
        all_idx,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=stratify_target,
    )
    train_df = df.iloc[idx_train].copy()
    val_df = df.iloc[idx_val].copy()

    X_av_train = av_feats[idx_train]
    X_av_val = av_feats[idx_val]
    X_phys_train = phys_feats[idx_train]
    X_phys_val = phys_feats[idx_val]
    y_train_arr = labels[idx_train]
    y_val_arr = labels[idx_val]
    val_domains = infer_domain_labels(val_df)

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
    y_train = torch.tensor(y_train_arr, dtype=torch.float32).unsqueeze(1)

    X_av_val = torch.tensor(X_av_val, dtype=torch.float32)
    X_phys_val = torch.tensor(X_phys_val, dtype=torch.float32)
    y_val = torch.tensor(y_val_arr, dtype=torch.float32).unsqueeze(1)

    sample_weights = None
    use_weights = False
    sampler = None
    if len(np.unique(y_train_arr)) > 1:
        # Base class balancing
        class_counts = np.bincount(y_train_arr.astype(int))
        class_weights = class_counts.sum() / np.maximum(class_counts, 1)
        sample_weights_np = np.array([class_weights[int(l)] for l in y_train_arr], dtype=np.float32)

        # Optional domain+label balancing
        if args.group_balance and "dataset" in train_df.columns:
            grp = (
                train_df["dataset"].astype(str).str.lower().fillna("unknown")
                + ":"
                + pd.Series(y_train_arr, index=train_df.index).astype(int).astype(str)
            )
            grp_counts = grp.value_counts().to_dict()
            grp_weights = {k: len(grp) / max(v, 1) for k, v in grp_counts.items()}
            sample_weights_np *= grp.map(grp_weights).to_numpy(dtype=np.float32)
            print(f"Applied group balancing across {len(grp_weights)} dataset-label groups.")

        # Optional hard-negative upweighting
        if args.hard_negative_file:
            hn_paths = load_hard_negative_paths(args.hard_negative_file)
            if "path" in train_df.columns:
                train_keys = train_df["path"].astype(str).map(_norm_path)
                hn_lookup = hn_paths
            elif "video_id" in train_df.columns:
                # Support datasets that only keep clip filename (e.g., DFDC video_id).
                train_keys = train_df["video_id"].astype(str).map(_norm_name)
                hn_lookup = {Path(p).name.lower() for p in hn_paths}
            else:
                train_keys = pd.Series([""] * len(train_df), index=train_df.index)
                hn_lookup = set()

            hn_mask = train_keys.isin(hn_lookup).to_numpy() & (y_train_arr.astype(int) == 0)
            hn_count = int(np.sum(hn_mask))
            if hn_count > 0:
                sample_weights_np[hn_mask] *= float(args.hard_negative_weight)
            print(f"Applied hard-negative weight to {hn_count} training rows.")

        # Normalize mean weight for stable loss scale
        sample_weights_np = sample_weights_np / max(float(np.mean(sample_weights_np)), 1e-6)
        sample_weights = torch.tensor(sample_weights_np, dtype=torch.float32).unsqueeze(1)
        use_weights = True

        if args.use_weighted_sampler:
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights_np, dtype=torch.double),
                num_samples=len(sample_weights_np),
                replacement=True,
            )
            print("Using WeightedRandomSampler for train batches.")

    if sampler is not None:
        if sample_weights is not None:
            dataset = TensorDataset(X_av_train, X_phys_train, y_train, sample_weights)
        else:
            dataset = TensorDataset(X_av_train, X_phys_train, y_train)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    else:
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
    best_selection_score = -1.0
    saved_checkpoint = False
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
        selection_threshold = float(args.selection_threshold)
        if args.selection_threshold_mode == "best_f1":
            selection_threshold = float(sweep["best_thr"])
        domain_metrics = compute_domain_metrics(
            val_labels,
            val_probs,
            val_domains,
            threshold=selection_threshold,
        )
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_auc)

        if args.selection_metric == "auc":
            selection_score = float(val_auc)
        elif args.selection_metric == "hybrid_robust":
            selection_score = (
                0.50 * float(domain_metrics["worst_bal_acc"])
                + 0.30 * float(domain_metrics["macro_bal_acc"])
                + 0.20 * float(domain_metrics["overall"]["f1"])
            )
        else:
            selection_score = float(domain_metrics["worst_bal_acc"])

        meets_domain_constraints = (
            float(domain_metrics["worst_spec"]) >= float(args.min_domain_spec)
            and float(domain_metrics["worst_rec"]) >= float(args.min_domain_rec)
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} "
            f"val_auc={val_auc:.3f} "
            f"pr_auc={sweep['pr_auc']:.3f} "
            f"best_f1={sweep['best_f1']:.3f} "
            f"best_thr={sweep['best_thr']:.3f} "
            f"sel_thr={selection_threshold:.3f} "
            f"worst_bal={domain_metrics['worst_bal_acc']:.3f} "
            f"worst_spec={domain_metrics['worst_spec']:.3f} "
            f"worst_rec={domain_metrics['worst_rec']:.3f} "
            f"sel_score={selection_score:.3f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc

        improved = selection_score > best_selection_score
        if improved and meets_domain_constraints:
            best_selection_score = selection_score
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            saved_checkpoint = True
            if scaler is not None:
                joblib.dump(scaler, scaler_path)
        elif improved:
            epochs_no_improve += 1
            failed = []
            if float(domain_metrics["worst_spec"]) < float(args.min_domain_spec):
                failed.append(
                    f"worst_spec={domain_metrics['worst_spec']:.3f}<{args.min_domain_spec:.3f}"
                )
            if float(domain_metrics["worst_rec"]) < float(args.min_domain_rec):
                failed.append(
                    f"worst_rec={domain_metrics['worst_rec']:.3f}<{args.min_domain_rec:.3f}"
                )
            print(
                f"Checkpoint skipped (constraints): "
                + (" and ".join(failed) if failed else "constraint check failed")
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    print("Best AUC:", best_auc)
    print("Best selection score:", best_selection_score)
    if not saved_checkpoint:
        print("Warning: no checkpoint met selection constraints; saving last epoch weights.")
        torch.save(model.state_dict(), model_path)
        if scaler is not None:
            joblib.dump(scaler, scaler_path)
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
