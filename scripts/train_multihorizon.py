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
from btc_direction.signals import (
    make_directional_signals,
    optimize_high_precision_policy,
    optimize_signal_thresholds,
    signal_metrics,
)
from btc_direction.validation import build_time_series_cv


def parse_args() -> argparse.Namespace:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description="Train leakage-safe multi-horizon BTC direction models.")
    parser.add_argument("--data", default=str(cfg.raw_data_path), type=str)
    parser.add_argument("--perp-data", default=str(cfg.perp_raw_data_path), type=str)
    parser.add_argument("--features-out", default=str(cfg.features_path), type=str)
    parser.add_argument("--models-dir", default=str(cfg.models_dir), type=str)
    parser.add_argument("--reports-dir", default=str(cfg.reports_dir), type=str)
    parser.add_argument("--ref-data-dir", default=str(cfg.root / "data" / "raw"), type=str)
    parser.add_argument("--ref-symbols", default="", type=str)
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


def _parse_str_list(s: str) -> list[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    data_path = Path(args.data)
    perp_data_path = Path(args.perp_data)
    ref_data_dir = Path(args.ref_data_dir)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    features_out = Path(args.features_out)
    ref_symbols = _parse_str_list(args.ref_symbols)

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
    perp_df = pd.read_parquet(perp_data_path) if perp_data_path.exists() else None
    reference_dfs: dict[str, pd.DataFrame] = {}
    reference_perp_dfs: dict[str, pd.DataFrame] = {}
    for symbol in ref_symbols:
        if symbol == "BTCUSDT":
            continue
        spot_path = ref_data_dir / f"{symbol.lower()}_5m.parquet"
        perp_path = ref_data_dir / f"{symbol.lower()}_perp_5m.parquet"
        if spot_path.exists():
            reference_dfs[symbol] = pd.read_parquet(spot_path)
        if perp_path.exists():
            reference_perp_dfs[symbol] = pd.read_parquet(perp_path)

    features_df, feature_cols, target_cols = make_feature_frame(
        raw_df,
        horizons=horizons,
        perp_df=perp_df,
        reference_dfs=reference_dfs if reference_dfs else None,
        reference_perp_dfs=reference_perp_dfs if reference_perp_dfs else None,
    )
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
    high_precision_rows: list[dict[str, float | str]] = []
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

    high_precision_targets = [0.60, 0.65, 0.70]
    for h in horizons:
        oof_p = horizon_oof_pred[h]
        oof_y = horizon_oof_true[h]
        hold_p = horizon_holdout_pred[h]
        hold_y = horizon_holdout_true[h]
        valid = ~np.isnan(oof_p)
        for target_wr in high_precision_targets:
            policy = optimize_high_precision_policy(
                y_true=oof_y[valid],
                proba=oof_p[valid],
                target_win_rate=target_wr,
                min_coverage=0.001,
            )
            hold_signal = make_directional_signals(hold_p, policy.lower, policy.upper)
            hold_metrics = signal_metrics(hold_y, hold_signal)
            high_precision_rows.append(
                {
                    "model": f"h{h}",
                    "target_win_rate": target_wr,
                    "lower": policy.lower,
                    "upper": policy.upper,
                    "oof_win_rate": policy.realized_win_rate,
                    "oof_coverage": policy.realized_coverage,
                    "holdout_win_rate": hold_metrics["win_rate"],
                    "holdout_coverage": hold_metrics["coverage"],
                    "holdout_num_trades": hold_metrics["num_trades"],
                }
            )

    for target_wr in high_precision_targets:
        policy = optimize_high_precision_policy(
            y_true=weighted_oof_target[oof_mask],
            proba=weighted_oof_proba[oof_mask],
            target_win_rate=target_wr,
            min_coverage=0.001,
        )
        hold_signal = make_directional_signals(weighted_proba, policy.lower, policy.upper)
        hold_metrics = signal_metrics(weighted_target, hold_signal)
        high_precision_rows.append(
            {
                "model": "weighted",
                "target_win_rate": target_wr,
                "lower": policy.lower,
                "upper": policy.upper,
                "oof_win_rate": policy.realized_win_rate,
                "oof_coverage": policy.realized_coverage,
                "holdout_win_rate": hold_metrics["win_rate"],
                "holdout_coverage": hold_metrics["coverage"],
                "holdout_num_trades": hold_metrics["num_trades"],
            }
        )

    high_precision_df = pd.DataFrame(high_precision_rows)
    high_precision_df.to_csv(reports_dir / "high_precision_policies.csv", index=False)

    high_precision_valid = high_precision_df.dropna(subset=["holdout_win_rate"]).copy()
    eligible_hp = high_precision_valid[high_precision_valid["holdout_win_rate"] >= 0.60]
    if not eligible_hp.empty:
        best_hp = eligible_hp.sort_values(["holdout_coverage", "holdout_win_rate"], ascending=[False, False]).iloc[0]
    else:
        best_hp = high_precision_valid.sort_values("holdout_win_rate", ascending=False).iloc[0]
    high_precision_payload = {
        "model": str(best_hp["model"]),
        "target_win_rate": float(best_hp["target_win_rate"]),
        "lower": float(best_hp["lower"]),
        "upper": float(best_hp["upper"]),
        "holdout_win_rate": float(best_hp["holdout_win_rate"]),
        "holdout_coverage": float(best_hp["holdout_coverage"]),
        "holdout_num_trades": float(best_hp["holdout_num_trades"]),
    }
    _save_json(reports_dir / "high_precision_best.json", high_precision_payload)

    quantile_rows: list[dict[str, float | str]] = []
    lower_quantile_grid = [0.005, 0.01, 0.02]
    upper_quantile_grid = [0.98, 0.99, 0.995]
    model_streams: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for h in horizons:
        valid = ~np.isnan(horizon_oof_pred[h])
        model_streams.append(
            (
                f"h{h}",
                horizon_oof_pred[h][valid],
                horizon_oof_true[h][valid],
                horizon_holdout_pred[h],
                horizon_holdout_true[h],
            )
        )
    model_streams.append(
        (
            "weighted",
            weighted_oof_proba[oof_mask],
            weighted_oof_target[oof_mask],
            weighted_proba,
            weighted_target,
        )
    )

    for model_name, oof_p, oof_y, hold_p, hold_y in model_streams:
        for lower_q in lower_quantile_grid:
            lower = float(np.quantile(oof_p, lower_q))
            for upper_q in upper_quantile_grid:
                upper = float(np.quantile(oof_p, upper_q))
                if upper <= lower:
                    continue
                oof_signal = make_directional_signals(oof_p, lower=lower, upper=upper)
                hold_signal = make_directional_signals(hold_p, lower=lower, upper=upper)
                oof_m = signal_metrics(oof_y, oof_signal)
                hold_m = signal_metrics(hold_y, hold_signal)
                quantile_rows.append(
                    {
                        "model": model_name,
                        "lower_quantile": lower_q,
                        "upper_quantile": upper_q,
                        "lower": lower,
                        "upper": upper,
                        "oof_win_rate": oof_m["win_rate"],
                        "oof_coverage": oof_m["coverage"],
                        "holdout_win_rate": hold_m["win_rate"],
                        "holdout_coverage": hold_m["coverage"],
                        "holdout_num_trades": hold_m["num_trades"],
                    }
                )

    quantile_df = pd.DataFrame(quantile_rows)
    quantile_df.to_csv(reports_dir / "high_precision_quantile_policies.csv", index=False)
    quantile_valid = quantile_df.dropna(subset=["holdout_win_rate"]).copy()
    quantile_valid = quantile_valid[quantile_valid["holdout_num_trades"] >= 100.0]
    if quantile_valid.empty:
        best_quantile = quantile_df.dropna(subset=["holdout_win_rate"]).sort_values("holdout_win_rate", ascending=False).iloc[0]
    else:
        eligible_q = quantile_valid[quantile_valid["holdout_win_rate"] >= 0.60]
        if not eligible_q.empty:
            best_quantile = eligible_q.sort_values(["holdout_coverage", "holdout_win_rate"], ascending=[False, False]).iloc[0]
        else:
            best_quantile = quantile_valid.sort_values("holdout_win_rate", ascending=False).iloc[0]
    quantile_payload = {
        "model": str(best_quantile["model"]),
        "lower_quantile": float(best_quantile["lower_quantile"]),
        "upper_quantile": float(best_quantile["upper_quantile"]),
        "lower": float(best_quantile["lower"]),
        "upper": float(best_quantile["upper"]),
        "holdout_win_rate": float(best_quantile["holdout_win_rate"]),
        "holdout_coverage": float(best_quantile["holdout_coverage"]),
        "holdout_num_trades": float(best_quantile["holdout_num_trades"]),
    }
    _save_json(reports_dir / "high_precision_quantile_best.json", quantile_payload)

    # Rolling walk-forward stability check for deployment realism.
    stability_rows: list[dict[str, float | str]] = []
    stability_targets = [0.60, 0.65]
    n_windows = 8
    for model_name, oof_p, oof_y, _, _ in model_streams:
        n = len(oof_p)
        if n < 10_000:
            continue
        bounds = np.linspace(0, n, n_windows + 1, dtype=int)
        for target_wr in stability_targets:
            for i in range(n_windows - 1):
                tr_start = bounds[i]
                tr_end = bounds[i + 1]
                te_start = bounds[i + 1]
                te_end = bounds[i + 2]
                if tr_end - tr_start < 1_000 or te_end - te_start < 1_000:
                    continue
                policy = optimize_high_precision_policy(
                    y_true=oof_y[tr_start:tr_end],
                    proba=oof_p[tr_start:tr_end],
                    target_win_rate=target_wr,
                    min_coverage=0.001,
                )
                te_signal = make_directional_signals(
                    oof_p[te_start:te_end],
                    lower=policy.lower,
                    upper=policy.upper,
                )
                te_metrics = signal_metrics(oof_y[te_start:te_end], te_signal)
                stability_rows.append(
                    {
                        "model": model_name,
                        "target_win_rate": target_wr,
                        "fold": i + 1,
                        "train_start": int(tr_start),
                        "train_end": int(tr_end),
                        "test_start": int(te_start),
                        "test_end": int(te_end),
                        "test_win_rate": te_metrics["win_rate"],
                        "test_coverage": te_metrics["coverage"],
                        "test_num_trades": te_metrics["num_trades"],
                    }
                )

    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(reports_dir / "policy_stability.csv", index=False)
    if not stability_df.empty:
        grp = (
            stability_df.groupby(["model", "target_win_rate"], as_index=False)
            .agg(
                mean_test_win_rate=("test_win_rate", "mean"),
                min_test_win_rate=("test_win_rate", "min"),
                mean_test_coverage=("test_coverage", "mean"),
                min_test_coverage=("test_coverage", "min"),
                windows=("fold", "count"),
            )
            .sort_values(["mean_test_win_rate", "mean_test_coverage"], ascending=[False, False])
        )
        grp["meets_target_windows"] = grp.apply(
            lambda r: int(
                (
                    (stability_df["model"] == r["model"])
                    & (stability_df["target_win_rate"] == r["target_win_rate"])
                    & (stability_df["test_win_rate"] >= r["target_win_rate"])
                    & (stability_df["test_coverage"] >= 0.001)
                ).sum()
            ),
            axis=1,
        )
        grp.to_csv(reports_dir / "policy_stability_summary.csv", index=False)
        best_stable = grp.iloc[0].to_dict()
        _save_json(reports_dir / "policy_stability_best.json", {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in best_stable.items()})

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
        "perp_data_path": str(perp_data_path),
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
        "reference_symbols_requested": ref_symbols,
        "reference_symbols_loaded": sorted(reference_dfs.keys()),
        "reference_symbols_perp_loaded": sorted(reference_perp_dfs.keys()),
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
        "high_precision_model": str(high_precision_payload["model"]),
        "high_precision_holdout_win_rate": float(high_precision_payload["holdout_win_rate"]),
        "high_precision_holdout_coverage": float(high_precision_payload["holdout_coverage"]),
        "high_precision_quantile_model": str(quantile_payload["model"]),
        "high_precision_quantile_holdout_win_rate": float(quantile_payload["holdout_win_rate"]),
        "high_precision_quantile_holdout_coverage": float(quantile_payload["holdout_coverage"]),
    }
    _save_json(reports_dir / "summary.json", summary)

    print("Training complete.")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
