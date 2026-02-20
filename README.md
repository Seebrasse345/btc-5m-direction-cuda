# BTC 5m Candle Direction (CUDA, Multi-Horizon, Leakage-Safe)

This project downloads full available `BTCUSDT` 5-minute candles, engineers leakage-safe features, and trains CUDA-accelerated multi-horizon classifiers to predict whether future candles close above or below their open.

## What it does

- Downloads and updates full Binance Spot `BTCUSDT` `5m` kline history.
- Builds lagged/rolling technical and market microstructure features.
- Trains one GPU XGBoost classifier per horizon (default: `1,3,6,12` candles = `5m,15m,30m,60m`).
- Runs per-horizon GPU hyperparameter search before final training.
- Optionally trains a CUDA LSTM sequence model and blends XGBoost+LSTM predictions.
- Learns confidence thresholds (abstention policy) from OOF predictions to improve win rate.
- Supports incremental online XGBoost updates as new 5m candles arrive.
- Evaluates with walk-forward time-series CV using an embargo gap to reduce leakage.
- Produces holdout reports, feature importances, prediction files, and model artifacts.
- Supports periodic retraining via GitHub Actions on a self-hosted GPU runner.

## Why this architecture

- XGBoost supports CUDA training/inference and is strong on structured/tabular data.
- Direct multi-horizon modeling avoids recursive error accumulation and is standard for multi-step forecasting tasks.
- Time-series CV with a `gap` reduces lookahead leakage around split boundaries.

## Quick start

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_full_pipeline.py --force-full-download
python scripts/analyze_reports.py

# Optional sequence model + blend
python scripts/train_lstm_multihorizon.py
python scripts/blend_xgb_lstm.py
```

## Individual commands

```powershell
# Download/update data only
python scripts/download_5m_data.py --force-full

# Train only
python scripts/train_multihorizon.py `
  --horizons 1,3,6,12 `
  --weights 0.4,0.25,0.2,0.15 `
  --n-splits 5 `
  --test-size 30000 `
  --holdout-fraction 0.2 `
  --tune-trials 12 `
  --min-signal-coverage 0.2

# LSTM sequence model (CUDA)
python scripts/train_lstm_multihorizon.py `
  --seq-len 96 `
  --epochs 10 `
  --batch-size 1024

# Blend XGB + LSTM
python scripts/blend_xgb_lstm.py

# Incremental update with newest candles
python scripts/online_update_xgb.py
```

## Leakage controls

- Targets are future candles: `target_h = 1{close[t+h] > open[t+h]}`.
- Features are built from current/past candles only.
- Validation uses contiguous time splits (`TimeSeriesSplit`) with `gap=max(horizons)`.
- Final evaluation is on the last chronological holdout block.
- Thresholds for trading signals are optimized only on development OOF predictions and then applied unchanged to holdout.

## Trading signal mode

For 5m BTC direction, raw class probabilities are close to random.  
The project therefore also produces a confidence-filtered signal policy (two-threshold abstention) to improve realized win rate on acted trades, with explicit coverage reporting.

## Outputs

- Raw data: `data/raw/btcusdt_5m.parquet`
- Feature matrix: `data/processed/btcusdt_5m_features.parquet`
- Models: `models/xgb_h*.json`
- Reports:
  - `reports/cv_metrics_by_fold.csv`
  - `reports/holdout_metrics.csv`
  - `reports/holdout_dummy_baseline_metrics.csv`
  - `reports/weighted_multi_horizon_metrics.json`
  - `reports/weighted_signal_policy_metrics.json`
  - `reports/best_signal_policy.json`
  - `reports/online_update_metrics.csv`
  - `reports/signal_policy_metrics.csv`
  - `reports/signal_policy_thresholds.csv`
  - `reports/feature_importance_h*.csv`
  - `reports/summary.json`
  - `reports/holdout_predictions.parquet`
  - `reports/oof_predictions.parquet`
  - `reports/lstm_validation_metrics.csv`
  - `reports/lstm_holdout_metrics.csv`
  - `reports/lstm_predictions.parquet`
  - `reports/blend_holdout_metrics.csv`
  - `reports/blend_summary.json`

## Online learning mode

`scripts/online_update_xgb.py` updates models with new rows only, continuing training from existing boosters.  
This uses XGBoost continuation training (`xgb_model`) and writes:
- `artifacts/online_state.json`
- `reports/online_update_metrics.csv`

## GitHub setup

```powershell
git init
git add .
git commit -m "Initial BTC CUDA multi-horizon pipeline"

# If GitHub CLI is authenticated:
gh repo create btc-direction-cuda --public --source . --remote origin --push
```

## Primary references

- Binance Spot API klines (interval, limits): https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md
- Binance public historical data (monthly/daily files): https://github.com/binance/binance-public-data
- XGBoost GPU support (`device=cuda`, `hist`, QuantileDMatrix): https://xgboost.readthedocs.io/en/stable/gpu/
- XGBoost continuation training (`xgb_model`): https://xgboost.readthedocs.io/en/stable/python/examples/continuation.html
- Threshold tuning for classification decisions: https://scikit-learn.org/stable/modules/classification_threshold.html
- Temporal Fusion Transformer (multi-horizon forecasting): https://arxiv.org/abs/1912.09363
- Multi-step forecasting strategies review (direct vs recursive): https://doi.org/10.1016/j.ijforecast.2011.03.006
- `TimeSeriesSplit` with `gap` for time-ordered validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
