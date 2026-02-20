from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from btc_direction.config import PipelineConfig
from btc_direction.features import make_feature_frame
from btc_direction.model import compute_metrics


def parse_args() -> argparse.Namespace:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description="Train CUDA LSTM multi-horizon BTC direction model.")
    parser.add_argument("--features", default=str(cfg.features_path), type=str)
    parser.add_argument("--data", default=str(cfg.raw_data_path), type=str)
    parser.add_argument("--models-dir", default=str(cfg.models_dir), type=str)
    parser.add_argument("--reports-dir", default=str(cfg.reports_dir), type=str)
    parser.add_argument("--horizons", default="1,3,6,12", type=str)
    parser.add_argument("--weights", default="0.4,0.25,0.2,0.15", type=str)
    parser.add_argument("--seq-len", default=96, type=int)
    parser.add_argument("--batch-size", default=1024, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--hidden-size", default=160, type=int)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.20, type=float)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--holdout-fraction", default=cfg.holdout_fraction, type=float)
    parser.add_argument("--val-fraction-within-dev", default=0.10, type=float)
    parser.add_argument("--patience", default=3, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=cfg.random_seed, type=int)
    return parser.parse_args()


def _parse_int_list(s: str) -> list[int]:
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def _parse_float_list(s: str) -> list[float]:
    out = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one float.")
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        open_times: np.ndarray,
        indices: np.ndarray,
        seq_len: int,
    ) -> None:
        self.features = features
        self.targets = targets
        self.open_times = open_times
        self.indices = indices
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        end_idx = int(self.indices[idx])
        start_idx = end_idx - self.seq_len + 1
        x = self.features[start_idx : end_idx + 1]
        y = self.targets[end_idx]
        t = self.open_times[end_idx]
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(t, dtype=torch.long)


class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, out_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, out_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


@dataclass(slots=True)
class PredBundle:
    probs: np.ndarray
    targets: np.ndarray
    open_times: np.ndarray


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> PredBundle:
    model.eval()
    prob_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []

    with torch.no_grad():
        for x, y, t in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            prob_parts.append(probs)
            target_parts.append(y.numpy().astype(np.int8))
            time_parts.append(t.numpy().astype(np.int64))

    return PredBundle(
        probs=np.concatenate(prob_parts, axis=0),
        targets=np.concatenate(target_parts, axis=0),
        open_times=np.concatenate(time_parts, axis=0),
    )


def weighted_auc(probs: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    aucs = []
    for i in range(targets.shape[1]):
        y = targets[:, i]
        p = probs[:, i]
        if len(np.unique(y)) < 2:
            aucs.append(0.5)
        else:
            aucs.append(float(roc_auc_score(y, p)))
    aucs_arr = np.array(aucs, dtype=np.float32)
    return float((aucs_arr * weights).sum())


def _build_features_if_missing(features_path: Path, raw_data_path: Path, horizons: list[int]) -> None:
    if features_path.exists():
        return
    if not raw_data_path.exists():
        raise FileNotFoundError("Neither features nor raw data file exists.")
    raw_df = pd.read_parquet(raw_data_path)
    features_df, _, _ = make_feature_frame(raw_df, horizons=horizons)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(features_path, index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    horizons = _parse_int_list(args.horizons)
    weights = _parse_float_list(args.weights)
    if len(horizons) != len(weights):
        raise ValueError("horizons and weights length mismatch.")
    weights_arr = np.array(weights, dtype=np.float32)
    weights_arr = weights_arr / weights_arr.sum()

    features_path = Path(args.features)
    raw_data_path = Path(args.data)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)

    _build_features_if_missing(features_path, raw_data_path, horizons)
    df = pd.read_parquet(features_path)

    feature_cols = [c for c in df.columns if c.startswith("f_")]
    target_cols = [f"target_h{h}" for h in horizons]
    for col in target_cols:
        if col not in df.columns:
            raise ValueError(f"Missing target column {col}. Rebuild features with matching horizons.")

    features = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[target_cols].to_numpy(dtype=np.float32)
    open_times = df["open_time"].to_numpy(dtype=np.int64)

    n_rows = len(df)
    seq_len = args.seq_len
    min_idx = seq_len - 1
    if n_rows <= min_idx + 1000:
        raise ValueError("Not enough rows for requested sequence length.")

    holdout_size = max(20_000, int(n_rows * args.holdout_fraction))
    holdout_start = n_rows - holdout_size
    dev_end = holdout_start
    val_start = int(dev_end * (1.0 - args.val_fraction_within_dev))
    val_start = max(val_start, min_idx + 10_000)

    train_idx = np.arange(min_idx, val_start, dtype=np.int64)
    val_idx = np.arange(val_start, dev_end, dtype=np.int64)
    holdout_idx = np.arange(max(holdout_start, min_idx), n_rows, dtype=np.int64)

    train_ds = SequenceDataset(features, targets, open_times, train_idx, seq_len=seq_len)
    val_ds = SequenceDataset(features, targets, open_times, val_idx, seq_len=seq_len)
    holdout_ds = SequenceDataset(features, targets, open_times, holdout_idx, seq_len=seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHorizonLSTM(
        input_size=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        out_size=len(horizons),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    loss_weights = torch.tensor(weights_arr, device=device, dtype=torch.float32).view(1, -1)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_val_score = -np.inf
    best_state = None
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y, _ in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss_matrix = bce(logits, y)
                loss = (loss_matrix * loss_weights).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_size = x.shape[0]
            running_loss += float(loss.detach().cpu()) * batch_size
            seen += batch_size

        train_loss = running_loss / max(1, seen)
        val_preds = evaluate_model(model, val_loader, device=device)
        val_score = weighted_auc(val_preds.probs, val_preds.targets, weights_arr)
        scheduler.step(val_score)

        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_weighted_auc": val_score})
        print(f"Epoch {epoch}/{args.epochs} - train_loss={train_loss:.6f} val_weighted_auc={val_score:.6f}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "lstm_multihorizon.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "horizons": horizons,
            "weights": [float(x) for x in weights_arr],
            "seq_len": seq_len,
            "feature_cols": feature_cols,
        },
        model_path,
    )

    val_bundle = evaluate_model(model, val_loader, device=device)
    holdout_bundle = evaluate_model(model, holdout_loader, device=device)

    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history_rows).to_csv(reports_dir / "lstm_training_history.csv", index=False)

    val_metrics_rows = []
    hold_metrics_rows = []
    for i, h in enumerate(horizons):
        vm = compute_metrics(val_bundle.targets[:, i].astype(np.int8), val_bundle.probs[:, i])
        vm["horizon"] = h
        val_metrics_rows.append(vm)

        hm = compute_metrics(holdout_bundle.targets[:, i].astype(np.int8), holdout_bundle.probs[:, i])
        hm["horizon"] = h
        hold_metrics_rows.append(hm)

    val_df = pd.DataFrame(val_metrics_rows).sort_values("horizon")
    hold_df = pd.DataFrame(hold_metrics_rows).sort_values("horizon")
    val_df.to_csv(reports_dir / "lstm_validation_metrics.csv", index=False)
    hold_df.to_csv(reports_dir / "lstm_holdout_metrics.csv", index=False)

    weighted_val_proba = (val_bundle.probs * weights_arr.reshape(1, -1)).sum(axis=1)
    weighted_val_truth = (val_bundle.targets * weights_arr.reshape(1, -1)).sum(axis=1)
    weighted_val_target = (weighted_val_truth >= 0.5).astype(np.int8)

    weighted_holdout_proba = (holdout_bundle.probs * weights_arr.reshape(1, -1)).sum(axis=1)
    weighted_holdout_truth = (holdout_bundle.targets * weights_arr.reshape(1, -1)).sum(axis=1)
    weighted_holdout_target = (weighted_holdout_truth >= 0.5).astype(np.int8)

    weighted_metrics = {
        "validation": compute_metrics(weighted_val_target, weighted_val_proba),
        "holdout": compute_metrics(weighted_holdout_target, weighted_holdout_proba),
        "horizons": horizons,
        "weights": [float(x) for x in weights_arr],
        "seq_len": seq_len,
    }
    (reports_dir / "lstm_weighted_metrics.json").write_text(json.dumps(weighted_metrics, indent=2), encoding="utf-8")

    val_pred_df = pd.DataFrame({"open_time": val_bundle.open_times, "split": "validation"})
    hold_pred_df = pd.DataFrame({"open_time": holdout_bundle.open_times, "split": "holdout"})
    for i, h in enumerate(horizons):
        val_pred_df[f"proba_h{h}"] = val_bundle.probs[:, i]
        val_pred_df[f"target_h{h}"] = val_bundle.targets[:, i].astype(np.int8)
        hold_pred_df[f"proba_h{h}"] = holdout_bundle.probs[:, i]
        hold_pred_df[f"target_h{h}"] = holdout_bundle.targets[:, i].astype(np.int8)

    preds_df = pd.concat([val_pred_df, hold_pred_df], ignore_index=True)
    preds_df.to_parquet(reports_dir / "lstm_predictions.parquet", index=False)

    print("LSTM training complete.")
    print("Holdout weighted metrics:", weighted_metrics["holdout"])


if __name__ == "__main__":
    main()
