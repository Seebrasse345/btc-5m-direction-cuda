from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def make_feature_frame(
    raw_df: pd.DataFrame,
    horizons: list[int],
    perp_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = raw_df.copy()
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    out = pd.DataFrame({"open_time": df["open_time"]})

    body = (df["close"] - df["open"]) / (df["open"] + 1e-12)
    spread = (df["high"] - df["low"]) / (df["open"] + 1e-12)
    upper_wick = (df["high"] - np.maximum(df["open"], df["close"])) / (df["open"] + 1e-12)
    lower_wick = (np.minimum(df["open"], df["close"]) - df["low"]) / (df["open"] + 1e-12)
    ret_1 = df["close"].pct_change()
    log_ret = np.log(df["close"] / df["close"].shift(1))

    out["f_body"] = body
    out["f_spread"] = spread
    out["f_upper_wick"] = upper_wick
    out["f_lower_wick"] = lower_wick
    out["f_ret_1"] = ret_1
    out["f_log_ret_1"] = log_ret

    for lag in [2, 3, 6, 12, 24, 48, 96]:
        out[f"f_ret_lag_{lag}"] = df["close"].pct_change(lag)
        out[f"f_vol_lag_{lag}"] = df["volume"].pct_change(lag)

    for window in [3, 6, 12, 24, 48, 96, 288]:
        out[f"f_ret_mean_{window}"] = ret_1.rolling(window).mean()
        out[f"f_ret_std_{window}"] = ret_1.rolling(window).std()
        out[f"f_ret_skew_{window}"] = ret_1.rolling(window).skew()
        out[f"f_volume_mean_{window}"] = df["volume"].rolling(window).mean()
        out[f"f_volume_std_{window}"] = df["volume"].rolling(window).std()
        out[f"f_range_mean_{window}"] = spread.rolling(window).mean()

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    out["f_rsi_14"] = _rsi(df["close"], period=14)
    out["f_atr_14"] = _atr(df["high"], df["low"], df["close"], period=14) / (df["close"] + 1e-12)
    out["f_macd"] = macd / (df["close"] + 1e-12)
    out["f_macd_signal"] = macd_signal / (df["close"] + 1e-12)
    out["f_ema12_ema26_ratio"] = ema_12 / (ema_26 + 1e-12)

    buy_pressure = df["taker_buy_base_asset_volume"] / (df["volume"] + 1e-12)
    out["f_buy_pressure"] = buy_pressure
    out["f_quote_volume_ratio"] = df["quote_asset_volume"] / (df["volume"] * df["close"] + 1e-12)
    out["f_trades_per_volume"] = df["num_trades"] / (df["volume"] + 1e-12)

    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    out["f_minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    out["f_minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    out["f_dayofweek_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0)
    out["f_dayofweek_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0)

    if perp_df is not None and not perp_df.empty:
        perp = perp_df.copy()
        for col in ["open_time", "open", "high", "low", "close", "volume"]:
            if col in perp.columns:
                perp[col] = pd.to_numeric(perp[col], errors="coerce")
        perp = perp[["open_time", "open", "high", "low", "close", "volume"]]
        perp = perp.sort_values("open_time").drop_duplicates("open_time")
        perp = perp.rename(
            columns={
                "open": "perp_open",
                "high": "perp_high",
                "low": "perp_low",
                "close": "perp_close",
                "volume": "perp_volume",
            }
        )
        merged = pd.DataFrame({"open_time": df["open_time"]}).merge(perp, on="open_time", how="left")

        perp_available = merged["perp_close"].notna().astype("float32")
        perp_open = merged["perp_open"].fillna(df["open"])
        perp_high = merged["perp_high"].fillna(df["high"])
        perp_low = merged["perp_low"].fillna(df["low"])
        perp_close = merged["perp_close"].fillna(df["close"])
        perp_volume = merged["perp_volume"].fillna(0.0)

        out["f_perp_available"] = perp_available
        out["f_perp_ret_1"] = perp_close.pct_change().fillna(0.0)
        out["f_basis_open"] = (perp_open - df["open"]) / (df["open"] + 1e-12)
        out["f_basis_close"] = (perp_close - df["close"]) / (df["close"] + 1e-12)
        out["f_basis_change"] = out["f_basis_close"] - out["f_basis_open"]
        out["f_perp_spot_vol_ratio"] = perp_volume / (df["volume"] + 1e-12)
        out["f_perp_body"] = (perp_close - perp_open) / (perp_open + 1e-12)

        for window in [3, 6, 12, 24, 48, 96, 288]:
            out[f"f_basis_mean_{window}"] = out["f_basis_close"].rolling(window).mean()
            out[f"f_basis_std_{window}"] = out["f_basis_close"].rolling(window).std()
            out[f"f_basis_z_{window}"] = (
                (out["f_basis_close"] - out["f_basis_close"].rolling(window).mean())
                / (out["f_basis_close"].rolling(window).std() + 1e-12)
            )

    target_cols: list[str] = []
    for h in horizons:
        col = f"target_h{h}"
        out[col] = (df["close"].shift(-h) > df["open"].shift(-h)).astype("float64")
        target_cols.append(col)

    feature_cols = [c for c in out.columns if c.startswith("f_")]
    out = out.replace([np.inf, -np.inf], np.nan)
    cols_for_na_check = feature_cols + target_cols
    out = out.dropna(subset=cols_for_na_check).reset_index(drop=True)

    for target_col in target_cols:
        out[target_col] = out[target_col].astype("int8")

    for feature_col in feature_cols:
        out[feature_col] = out[feature_col].astype("float32")

    return out, feature_cols, target_cols
