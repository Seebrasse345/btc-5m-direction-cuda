from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from btc_direction.config import PipelineConfig
from btc_direction.data import load_or_update_btc_data
from btc_direction.features import make_feature_frame
from btc_direction.model import compute_metrics, xgb_default_params


def parse_args() -> argparse.Namespace:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description="Incrementally update XGBoost BTC direction models with new candles.")
    parser.add_argument("--data", default=str(cfg.raw_data_path), type=str)
    parser.add_argument("--features-out", default=str(cfg.features_path), type=str)
    parser.add_argument("--models-dir", default=str(cfg.models_dir), type=str)
    parser.add_argument("--reports-dir", default=str(cfg.reports_dir), type=str)
    parser.add_argument("--state-file", default="artifacts/online_state.json", type=str)
    parser.add_argument("--run-config", default="reports/run_config.json", type=str)
    parser.add_argument("--horizons", default="1,3,6,12", type=str)
    parser.add_argument("--lookback-rows", default=180_000, type=int)
    parser.add_argument("--eval-size", default=20_000, type=int)
    parser.add_argument("--update-rounds", default=220, type=int)
    parser.add_argument("--early-stopping-rounds", default=40, type=int)
    parser.add_argument("--min-new-rows", default=24, type=int)
    parser.add_argument("--seed", default=cfg.random_seed, type=int)
    return parser.parse_args()


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_params_by_horizon(run_config_path: Path, horizons: list[int], seed: int) -> dict[int, dict]:
    base = xgb_default_params(seed=seed)
    payload = _load_json(run_config_path)
    raw = payload.get("xgb_params_by_horizon", {})
    out: dict[int, dict] = {}
    for h in horizons:
        params = raw.get(str(h), None)
        if isinstance(params, dict):
            merged = dict(base)
            merged.update(params)
            merged["seed"] = seed + h
            out[h] = merged
        else:
            p = dict(base)
            p["seed"] = seed + h
            out[h] = p
    return out


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    features_out = Path(args.features_out)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    state_path = Path(args.state_file)
    run_config_path = Path(args.run_config)
    horizons = _parse_int_list(args.horizons)
    if not horizons:
        raise ValueError("At least one horizon is required.")

    raw_df = load_or_update_btc_data(
        output_path=data_path,
        symbol="BTCUSDT",
        interval="5m",
        start_date="2017-08-17",
        force_full=False,
    )
    features_df, feature_cols, target_cols = make_feature_frame(raw_df, horizons=horizons)
    features_out.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(features_out, index=False)

    X = features_df[feature_cols].to_numpy(dtype=np.float32)
    open_times = features_df["open_time"].to_numpy(dtype=np.int64)

    state = _load_json(state_path)
    last_seen_open_time = int(state.get("last_open_time", 0))
    if last_seen_open_time > 0:
        new_mask = open_times > last_seen_open_time
        new_rows = int(new_mask.sum())
    else:
        new_rows = len(open_times)

    if new_rows < args.min_new_rows:
        print(f"Only {new_rows} new rows since last update. Skipping model update.")
        return

    first_new_idx = max(0, len(open_times) - new_rows)
    start_idx = max(0, first_new_idx - args.lookback_rows)
    params_by_horizon = _load_params_by_horizon(run_config_path, horizons, seed=args.seed)

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for horizon, target_col in zip(horizons, target_cols):
        y = features_df[target_col].to_numpy(dtype=np.int8)
        X_slice = X[start_idx:]
        y_slice = y[start_idx:]

        eval_size = min(args.eval_size, max(10_000, len(X_slice) // 5))
        split = len(X_slice) - eval_size
        if split <= 5_000:
            raise ValueError("Not enough rows in incremental window; increase lookback-rows.")

        X_train, y_train = X_slice[:split], y_slice[:split]
        X_val, y_val = X_slice[split:], y_slice[split:]

        params = dict(params_by_horizon[horizon])
        max_bin = int(params.get("max_bin", 256))
        dtrain = xgb.QuantileDMatrix(X_train, y_train, max_bin=max_bin)
        dval = xgb.QuantileDMatrix(X_val, y_val, ref=dtrain, max_bin=max_bin)

        model_path = models_dir / f"xgb_h{horizon}.json"
        kwargs = {}
        if model_path.exists():
            kwargs["xgb_model"] = str(model_path)

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=args.update_rounds,
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=False,
            **kwargs,
        )
        booster.save_model(model_path)

        val_pred = booster.predict(xgb.DMatrix(X_val)).astype(np.float32)
        m = compute_metrics(y_val, val_pred)
        m["horizon"] = horizon
        m["new_rows"] = new_rows
        m["start_idx"] = start_idx
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("horizon")
    metrics_df.to_csv(reports_dir / "online_update_metrics.csv", index=False)

    latest_open_time = int(open_times.max())
    new_state = {
        "last_open_time": latest_open_time,
        "last_update_utc": datetime.now(timezone.utc).isoformat(),
        "new_rows_seen": new_rows,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(new_state, indent=2), encoding="utf-8")

    print("Incremental update complete.")
    print(metrics_df[["horizon", "auc", "accuracy", "f1"]])


if __name__ == "__main__":
    main()
