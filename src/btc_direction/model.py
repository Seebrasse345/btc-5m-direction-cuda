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


def sample_xgb_params(seed: int, trial: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed + trial * 17)
    return {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "tree_method": "hist",
        "device": "cuda",
        "eta": float(np.exp(rng.uniform(np.log(0.008), np.log(0.08)))),
        "max_depth": int(rng.integers(4, 13)),
        "subsample": float(rng.uniform(0.60, 1.00)),
        "colsample_bytree": float(rng.uniform(0.60, 1.00)),
        "min_child_weight": float(np.exp(rng.uniform(np.log(1.0), np.log(24.0)))),
        "lambda": float(np.exp(rng.uniform(np.log(1e-3), np.log(15.0)))),
        "alpha": float(np.exp(rng.uniform(np.log(1e-4), np.log(8.0)))),
        "gamma": float(rng.uniform(0.0, 6.0)),
        "max_bin": int(rng.choice([256, 384, 512])),
        "grow_policy": str(rng.choice(["depthwise", "lossguide"])),
        "seed": seed + trial,
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


@dataclass(slots=True)
class TuneResult:
    best_params: dict[str, Any]
    best_score: float
    trials_df: pd.DataFrame


def _train_eval_auc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[float, int]:
    max_bin = int(params.get("max_bin", 256))
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
    pred = booster.predict(xgb.DMatrix(X_val)).astype(np.float32)
    auc = compute_metrics(y_val, pred)["auc"]
    return float(auc), int(booster.best_iteration)


def tune_xgb_params(
    X: np.ndarray,
    y: np.ndarray,
    tune_splits,
    base_params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
    n_trials: int,
    seed: int,
) -> TuneResult:
    if n_trials <= 0:
        return TuneResult(best_params=base_params, best_score=float("nan"), trials_df=pd.DataFrame())

    rows: list[dict[str, float | int | str]] = []
    best_score = float("-inf")
    best_params = dict(base_params)

    for trial in range(1, n_trials + 1):
        params = sample_xgb_params(seed=seed, trial=trial)
        aucs = []
        iters = []
        for split_id, (train_idx, val_idx) in enumerate(tune_splits, start=1):
            auc, best_iteration = _train_eval_auc(
                X_train=X[train_idx],
                y_train=y[train_idx],
                X_val=X[val_idx],
                y_val=y[val_idx],
                params=params,
                num_boost_round=max(300, num_boost_round // 2),
                early_stopping_rounds=max(40, early_stopping_rounds // 2),
            )
            aucs.append(auc)
            iters.append(best_iteration)

        score = float(np.nanmean(aucs))
        row = {
            "trial": trial,
            "score_auc": score,
            "eta": params["eta"],
            "max_depth": params["max_depth"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "min_child_weight": params["min_child_weight"],
            "lambda": params["lambda"],
            "alpha": params["alpha"],
            "gamma": params["gamma"],
            "max_bin": params["max_bin"],
            "grow_policy": params["grow_policy"],
            "avg_best_iteration": float(np.mean(iters)) if iters else float("nan"),
        }
        rows.append(row)

        if score > best_score:
            best_score = score
            best_params = dict(params)

    # Keep objective/eval config stable.
    best_params["objective"] = base_params.get("objective", "binary:logistic")
    best_params["eval_metric"] = base_params.get("eval_metric", ["logloss", "auc"])
    best_params["tree_method"] = base_params.get("tree_method", "hist")
    best_params["device"] = base_params.get("device", "cuda")
    best_params["seed"] = seed

    return TuneResult(
        best_params=best_params,
        best_score=best_score,
        trials_df=pd.DataFrame(rows).sort_values("score_auc", ascending=False).reset_index(drop=True),
    )


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
