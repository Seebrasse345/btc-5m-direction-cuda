from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def xgb_default_params(seed: int = 42) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "tree_method": "hist",
        "device": "cuda",
        "eta": 0.03,
        "max_depth": 8,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 8.0,
        "lambda": 1.5,
        "alpha": 0.1,
        "max_bin": 512,
        "grow_policy": "lossguide",
        "seed": seed,
    }


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(np.int8)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
        "logloss": float(log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }
    if len(np.unique(y_true)) == 2:
        metrics["auc"] = float(roc_auc_score(y_true, proba))
    else:
        metrics["auc"] = float("nan")
    return metrics


@dataclass(slots=True)
class FoldResult:
    fold_metrics: pd.DataFrame
    oof_pred: np.ndarray


def cross_validate_xgb(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits,
    params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> FoldResult:
    oof_pred = np.full(shape=len(y), fill_value=np.nan, dtype=np.float32)
    rows: list[dict[str, float | int]] = []
    max_bin = int(params.get("max_bin", 256))

    for fold_id, (train_idx, val_idx) in enumerate(cv_splits, start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        dtrain = xgb.QuantileDMatrix(X_train, y_train, max_bin=max_bin)
        dval = xgb.QuantileDMatrix(X_val, y_val, ref=dtrain, max_bin=max_bin)
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        val_proba = booster.predict(xgb.DMatrix(X_val)).astype(np.float32)
        oof_pred[val_idx] = val_proba
        fold_metrics = compute_metrics(y_val, val_proba)
        fold_metrics["fold"] = fold_id
        fold_metrics["best_iteration"] = int(booster.best_iteration)
        rows.append(fold_metrics)

    return FoldResult(fold_metrics=pd.DataFrame(rows), oof_pred=oof_pred)


def train_final_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> xgb.Booster:
    n_rows = len(X_train)
    val_size = max(10_000, int(n_rows * 0.1))
    val_size = min(val_size, n_rows // 3)
    split = n_rows - val_size
    if split <= 0:
        raise ValueError("Not enough training rows to create internal validation split.")

    max_bin = int(params.get("max_bin", 256))
    dtrain = xgb.QuantileDMatrix(X_train[:split], y_train[:split], max_bin=max_bin)
    dval = xgb.QuantileDMatrix(X_train[split:], y_train[split:], ref=dtrain, max_bin=max_bin)

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    return booster


def feature_importance_from_booster(
    booster: xgb.Booster,
    feature_names: list[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    raw = booster.get_score(importance_type=importance_type)
    rows = [{"feature": feature_names[int(k[1:])], "importance": v} for k, v in raw.items()]
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names))})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)
