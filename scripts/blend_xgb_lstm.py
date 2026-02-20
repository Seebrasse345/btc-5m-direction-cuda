from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from btc_direction.model import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend XGBoost and LSTM multi-horizon predictions.")
    parser.add_argument("--xgb-preds", default="reports/holdout_predictions.parquet", type=str)
    parser.add_argument("--lstm-preds", default="reports/lstm_predictions.parquet", type=str)
    parser.add_argument("--reports-dir", default="reports", type=str)
    parser.add_argument("--weights", default="0.4,0.25,0.2,0.15", type=str)
    parser.add_argument("--calib-fraction", default=0.5, type=float)
    return parser.parse_args()


def _parse_float_list(s: str) -> list[float]:
    out = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one float.")
    return out


def main() -> None:
    args = parse_args()
    xgb_df = pd.read_parquet(args.xgb_preds)
    lstm_df = pd.read_parquet(args.lstm_preds)

    if "split" in lstm_df.columns:
        lstm_df = lstm_df[lstm_df["split"] == "holdout"].copy()
        lstm_df = lstm_df.drop(columns=["split"])

    merged = xgb_df.merge(lstm_df, on="open_time", how="inner", suffixes=("_xgb", "_lstm"))
    if merged.empty:
        raise RuntimeError("No aligned rows between XGB and LSTM predictions.")

    horizon_cols = sorted(
        int(c.replace("proba_h", ""))
        for c in xgb_df.columns
        if c.startswith("proba_h")
    )

    calib_size = int(len(merged) * args.calib_fraction)
    calib_size = max(5_000, calib_size)
    calib_size = min(calib_size, len(merged) - 5_000)
    calib = merged.iloc[:calib_size].copy()
    test = merged.iloc[calib_size:].copy()

    alpha_grid = np.linspace(0.0, 1.0, 21)
    selected_alphas: dict[int, float] = {}
    rows = []

    for h in horizon_cols:
        y_calib = calib[f"target_h{h}_xgb"].to_numpy(dtype=np.int8)
        p_xgb_calib = calib[f"proba_h{h}_xgb"].to_numpy(dtype=np.float32)
        p_lstm_calib = calib[f"proba_h{h}_lstm"].to_numpy(dtype=np.float32)

        best_alpha = 0.5
        best_auc = -np.inf
        for alpha in alpha_grid:
            p = alpha * p_xgb_calib + (1.0 - alpha) * p_lstm_calib
            if len(np.unique(y_calib)) < 2:
                auc = 0.5
            else:
                auc = compute_metrics(y_calib, p)["auc"]
            if auc > best_auc:
                best_auc = auc
                best_alpha = float(alpha)

        selected_alphas[h] = best_alpha

        y_test = test[f"target_h{h}_xgb"].to_numpy(dtype=np.int8)
        p_xgb_test = test[f"proba_h{h}_xgb"].to_numpy(dtype=np.float32)
        p_lstm_test = test[f"proba_h{h}_lstm"].to_numpy(dtype=np.float32)
        p_blend = best_alpha * p_xgb_test + (1.0 - best_alpha) * p_lstm_test

        metrics = compute_metrics(y_test, p_blend)
        metrics["horizon"] = h
        metrics["alpha_xgb"] = best_alpha
        rows.append(metrics)

    out_df = pd.DataFrame(rows).sort_values("horizon")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(reports_dir / "blend_holdout_metrics.csv", index=False)

    weights = np.array(_parse_float_list(args.weights), dtype=np.float32)
    if len(weights) != len(horizon_cols):
        raise ValueError("weights length must match number of horizons.")
    weights /= weights.sum()

    weighted_proba = np.zeros(len(test), dtype=np.float32)
    weighted_truth = np.zeros(len(test), dtype=np.float32)
    for w, h in zip(weights, horizon_cols):
        alpha = selected_alphas[h]
        p = alpha * test[f"proba_h{h}_xgb"].to_numpy(dtype=np.float32) + (1.0 - alpha) * test[
            f"proba_h{h}_lstm"
        ].to_numpy(dtype=np.float32)
        y = test[f"target_h{h}_xgb"].to_numpy(dtype=np.float32)
        weighted_proba += w * p
        weighted_truth += w * y

    weighted_target = (weighted_truth >= 0.5).astype(np.int8)
    weighted_metrics = compute_metrics(weighted_target, weighted_proba)
    payload = {
        "selected_alphas": {str(k): float(v) for k, v in selected_alphas.items()},
        "weighted_holdout_metrics": weighted_metrics,
        "calibration_rows": int(len(calib)),
        "test_rows": int(len(test)),
    }
    (reports_dir / "blend_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Blend complete.")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
