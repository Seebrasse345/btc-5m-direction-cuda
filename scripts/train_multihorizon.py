from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from btc_direction.config import PipelineConfig
from btc_direction.features import make_feature_frame
from btc_direction.model import (
    compute_metrics,
    cross_validate_xgb,
    feature_importance_from_booster,
    tune_xgb_params,
    train_final_xgb,
    xgb_default_params,
)
from btc_direction.signals import make_directional_signals, optimize_signal_thresholds, signal_metrics
from btc_direction.validation import build_time_series_cv


def parse_args() -> argparse.Namespace:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description="Train leakage-safe multi-horizon BTC direction models.")
    parser.add_argument("--data", default=str(cfg.raw_data_path), type=str)
    parser.add_argument("--features-out", default=str(cfg.features_path), type=str)
    parser.add_argument("--models-dir", default=str(cfg.models_dir), type=str)
    parser.add_argument("--reports-dir", default=str(cfg.reports_dir), type=str)
    parser.add_argument("--horizons", default="1,3,6,12", type=str)
    parser.add_argument("--weights", default="0.4,0.25,0.2,0.15", type=str)
    parser.add_argument("--n-splits", default=cfg.n_splits, type=int)
    parser.add_argument("--test-size", default=cfg.test_size, type=int)
    parser.add_argument("--holdout-fraction", default=cfg.holdout_fraction, type=float)
    parser.add_argument("--num-boost-round", default=cfg.num_boost_round, type=int)
    parser.add_argument("--early-stopping-rounds", default=cfg.early_stopping_rounds, type=int)
    parser.add_argument("--tune-trials", default=12, type=int)
    parser.add_argument("--tune-folds", default=2, type=int)
    parser.add_argument("--min-signal-coverage", default=0.20, type=float)
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


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    data_path = Path(args.data)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    features_out = Path(args.features_out)

    horizons = _parse_int_list(args.horizons)
    weights = _parse_float_list(args.weights)
    if len(horizons) != len(weights):
        raise ValueError("horizons and weights must have the same length.")
    if any(h <= 0 for h in horizons):
        raise ValueError("All horizons must be positive.")

    weights_arr = np.array(weights, dtype=np.float32)
    weights_arr = weights_arr / weights_arr.sum()

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    raw_df = pd.read_parquet(data_path)
    features_df, feature_cols, target_cols = make_feature_frame(raw_df, horizons=horizons)
    features_out.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(features_out, index=False)

    X = features_df[feature_cols].to_numpy(dtype=np.float32)
    open_times = features_df["open_time"].to_numpy(dtype=np.int64)
    n_rows = len(features_df)
    holdout_size = int(n_rows * args.holdout_fraction)
    holdout_size = max(20_000, holdout_size)
    if holdout_size >= n_rows:
        raise ValueError("Holdout size is too large for available rows.")
    dev_end = n_rows - holdout_size

    X_dev = X[:dev_end]
    X_holdout = X[dev_end:]
    holdout_times = open_times[dev_end:]

    max_h = max(horizons)
    tscv = build_time_series_cv(
        n_rows=len(X_dev),
        n_splits=args.n_splits,
        test_size=args.test_size,
        gap=max_h,
    )
    split_indices = list(tscv.split(X_dev))

    base_params = xgb_default_params(seed=args.seed)
    cv_rows: list[pd.DataFrame] = []
    holdout_rows: list[dict[str, float | str]] = []
    dummy_rows: list[dict[str, float | str]] = []
    signal_rows: list[dict[str, float | str]] = []
    signal_threshold_rows: list[dict[str, float | str]] = []
    horizon_holdout_pred: dict[int, np.ndarray] = {}
    horizon_holdout_true: dict[int, np.ndarray] = {}
    horizon_oof_pred: dict[int, np.ndarray] = {}
    horizon_oof_true: dict[int, np.ndarray] = {}
    horizon_params: dict[int, dict] = {}

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for horizon, target_col in zip(horizons, target_cols):
        print(f"Training horizon h={horizon} ({horizon * 5} minutes)")
        y_all = features_df[target_col].to_numpy(dtype=np.int8)
        y_dev = y_all[:dev_end]
        y_holdout = y_all[dev_end:]

        if args.tune_trials > 0:
            tune_fold_count = max(1, min(args.tune_folds, len(split_indices)))
            tune_splits = split_indices[-tune_fold_count:]
            tune_result = tune_xgb_params(
                X=X_dev,
                y=y_dev,
                tune_splits=tune_splits,
                base_params=base_params,
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=args.early_stopping_rounds,
                n_trials=args.tune_trials,
                seed=args.seed + horizon * 101,
            )
            params = tune_result.best_params
            if not tune_result.trials_df.empty:
                tune_result.trials_df.to_csv(reports_dir / f"tuning_trials_h{horizon}.csv", index=False)
        else:
            params = dict(base_params)
        horizon_params[horizon] = dict(params)

        cv_result = cross_validate_xgb(
            X=X_dev,
            y=y_dev,
            cv_splits=split_indices,
            params=params,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        fold_metrics = cv_result.fold_metrics.copy()
        fold_metrics["horizon"] = horizon
        cv_rows.append(fold_metrics)
        horizon_oof_pred[horizon] = cv_result.oof_pred
        horizon_oof_true[horizon] = y_dev

        final_model = train_final_xgb(
            X_train=X_dev,
            y_train=y_dev,
            params=params,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        model_path = models_dir / f"xgb_h{horizon}.json"
        final_model.save_model(model_path)

        holdout_proba = final_model.predict(xgb.DMatrix(X_holdout)).astype(np.float32)
        horizon_holdout_pred[horizon] = holdout_proba
        horizon_holdout_true[horizon] = y_holdout

        hold_metrics = compute_metrics(y_holdout, holdout_proba)
        hold_metrics["horizon"] = str(horizon)
        holdout_rows.append(hold_metrics)

        class_prior = float(y_dev.mean())
        dummy_proba = np.full_like(holdout_proba, fill_value=class_prior, dtype=np.float32)
        dummy_metrics = compute_metrics(y_holdout, dummy_proba)
        dummy_metrics["horizon"] = str(horizon)
        dummy_rows.append(dummy_metrics)

        oof_proba = cv_result.oof_pred
        threshold_cfg = optimize_signal_thresholds(
            y_true=y_dev,
            proba=oof_proba,
            min_coverage=args.min_signal_coverage,
        )
        oof_signal = make_directional_signals(oof_proba[~np.isnan(oof_proba)], threshold_cfg.lower, threshold_cfg.upper)
        oof_signal_metrics = signal_metrics(y_dev[~np.isnan(oof_proba)], oof_signal)
        holdout_signal = make_directional_signals(holdout_proba, threshold_cfg.lower, threshold_cfg.upper)
        holdout_signal_metrics = signal_metrics(y_holdout, holdout_signal)

        signal_threshold_rows.append(
            {
                "horizon": horizon,
                "lower": threshold_cfg.lower,
                "upper": threshold_cfg.upper,
                "objective_score": threshold_cfg.objective_score,
            }
        )
        signal_rows.append(
            {
                "horizon": horizon,
                "split": "oof_dev",
                **oof_signal_metrics,
            }
        )
        signal_rows.append(
            {
                "horizon": horizon,
                "split": "holdout",
                **holdout_signal_metrics,
            }
        )

        importance_df = feature_importance_from_booster(final_model, feature_cols)
        importance_df.to_csv(reports_dir / f"feature_importance_h{horizon}.csv", index=False)

    cv_df = pd.concat(cv_rows, ignore_index=True)
    cv_df.to_csv(reports_dir / "cv_metrics_by_fold.csv", index=False)

    holdout_df = pd.DataFrame(holdout_rows).sort_values("horizon")
    holdout_df.to_csv(reports_dir / "holdout_metrics.csv", index=False)
    dummy_df = pd.DataFrame(dummy_rows).sort_values("horizon")
    dummy_df.to_csv(reports_dir / "holdout_dummy_baseline_metrics.csv", index=False)
    signal_df = pd.DataFrame(signal_rows).sort_values(["split", "horizon"])
    signal_df.to_csv(reports_dir / "signal_policy_metrics.csv", index=False)
    threshold_df = pd.DataFrame(signal_threshold_rows).sort_values("horizon")
    threshold_df.to_csv(reports_dir / "signal_policy_thresholds.csv", index=False)

    weighted_proba = np.zeros(len(X_holdout), dtype=np.float32)
    weighted_truth = np.zeros(len(X_holdout), dtype=np.float32)
    weighted_oof_proba = np.zeros(len(X_dev), dtype=np.float32)
    weighted_oof_truth = np.zeros(len(X_dev), dtype=np.float32)
    for w, h in zip(weights_arr, horizons):
        weighted_proba += w * horizon_holdout_pred[h]
        weighted_truth += w * horizon_holdout_true[h].astype(np.float32)
        weighted_oof_proba += w * np.nan_to_num(horizon_oof_pred[h], nan=0.5)
        weighted_oof_truth += w * horizon_oof_true[h].astype(np.float32)

    weighted_target = (weighted_truth >= 0.5).astype(np.int8)
    weighted_metrics = compute_metrics(weighted_target, weighted_proba)
    weighted_metrics["horizons"] = ",".join(str(h) for h in horizons)
    weighted_metrics["weights"] = ",".join(f"{float(w):.4f}" for w in weights_arr)

    weighted_oof_target = (weighted_oof_truth >= 0.5).astype(np.int8)
    oof_mask = np.ones(len(weighted_oof_proba), dtype=bool)
    for h in horizons:
        oof_mask &= ~np.isnan(horizon_oof_pred[h])
    weighted_signal_threshold = optimize_signal_thresholds(
        y_true=weighted_oof_target[oof_mask],
        proba=weighted_oof_proba[oof_mask],
        min_coverage=args.min_signal_coverage,
    )
    weighted_holdout_signal = make_directional_signals(
        weighted_proba,
        lower=weighted_signal_threshold.lower,
        upper=weighted_signal_threshold.upper,
    )
    weighted_signal_metrics = signal_metrics(weighted_target, weighted_holdout_signal)
    weighted_signal_payload = {
        "mode": "weighted_multi_horizon",
        "lower": weighted_signal_threshold.lower,
        "upper": weighted_signal_threshold.upper,
        "objective_score": weighted_signal_threshold.objective_score,
        **weighted_signal_metrics,
    }
    _save_json(reports_dir / "weighted_multi_horizon_metrics.json", weighted_metrics)
    _save_json(reports_dir / "weighted_signal_policy_metrics.json", weighted_signal_payload)
    _save_json(reports_dir / "best_signal_policy.json", weighted_signal_payload)

    pred_cols = {"open_time": holdout_times}
    for h in horizons:
        pred_cols[f"proba_h{h}"] = horizon_holdout_pred[h]
        pred_cols[f"target_h{h}"] = horizon_holdout_true[h]
    pred_cols["weighted_proba"] = weighted_proba
    pred_cols["weighted_target"] = weighted_target
    pd.DataFrame(pred_cols).to_parquet(reports_dir / "holdout_predictions.parquet", index=False)

    oof_cols = {"open_time": open_times[:dev_end]}
    for h in horizons:
        oof_cols[f"proba_h{h}"] = horizon_oof_pred[h]
        oof_cols[f"target_h{h}"] = horizon_oof_true[h]
    oof_cols["weighted_proba"] = weighted_oof_proba
    oof_cols["weighted_target"] = weighted_oof_target
    pd.DataFrame(oof_cols).to_parquet(reports_dir / "oof_predictions.parquet", index=False)

    run_config = {
        "data_path": str(data_path),
        "features_out": str(features_out),
        "models_dir": str(models_dir),
        "reports_dir": str(reports_dir),
        "horizons": horizons,
        "weights": [float(w) for w in weights_arr],
        "n_splits": args.n_splits,
        "test_size": args.test_size,
        "holdout_fraction": args.holdout_fraction,
        "num_boost_round": args.num_boost_round,
        "early_stopping_rounds": args.early_stopping_rounds,
        "tune_trials": args.tune_trials,
        "tune_folds": args.tune_folds,
        "min_signal_coverage": args.min_signal_coverage,
        "seed": args.seed,
        "n_rows_total": int(n_rows),
        "n_rows_dev": int(len(X_dev)),
        "n_rows_holdout": int(len(X_holdout)),
        "feature_count": len(feature_cols),
        "xgb_base_params": base_params,
        "xgb_params_by_horizon": {str(k): v for k, v in horizon_params.items()},
    }
    _save_json(reports_dir / "run_config.json", run_config)

    summary = {
        "cv_auc_mean_by_horizon": cv_df.groupby("horizon")["auc"].mean().to_dict(),
        "holdout_auc_by_horizon": holdout_df.set_index("horizon")["auc"].to_dict(),
        "weighted_holdout_auc": float(weighted_metrics.get("auc", float("nan"))),
        "weighted_holdout_f1": float(weighted_metrics.get("f1", float("nan"))),
        "weighted_signal_holdout_win_rate": float(weighted_signal_metrics.get("win_rate", float("nan"))),
        "weighted_signal_holdout_coverage": float(weighted_signal_metrics.get("coverage", float("nan"))),
        "best_signal_mode": str(weighted_signal_payload["mode"]),
        "best_signal_holdout_win_rate": float(weighted_signal_payload["win_rate"]),
        "best_signal_holdout_coverage": float(weighted_signal_payload["coverage"]),
    }
    _save_json(reports_dir / "summary.json", summary)

    print("Training complete.")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
