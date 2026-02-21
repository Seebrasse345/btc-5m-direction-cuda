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


def _symbol_prefix(symbol: str) -> str:
    return "".join(ch for ch in symbol.lower() if ch.isalnum())


def make_feature_frame(
    raw_df: pd.DataFrame,
    horizons: list[int],
    perp_df: pd.DataFrame | None = None,
    reference_dfs: dict[str, pd.DataFrame] | None = None,
    reference_perp_dfs: dict[str, pd.DataFrame] | None = None,
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

    # Candlestick-pattern features (contextual market structure)
    body_abs = (df["close"] - df["open"]).abs()
    candle_range = (df["high"] - df["low"]).abs() + 1e-12
    upper_wick_abs = (df["high"] - np.maximum(df["open"], df["close"])).clip(lower=0.0)
    lower_wick_abs = (np.minimum(df["open"], df["close"]) - df["low"]).clip(lower=0.0)
    body_to_range = body_abs / candle_range
    upper_to_range = upper_wick_abs / candle_range
    lower_to_range = lower_wick_abs / candle_range

    bullish = (df["close"] > df["open"]).astype("float32")
    bearish = (df["close"] < df["open"]).astype("float32")
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    prev_range = (prev_high - prev_low).abs() + 1e-12
    close_pos = (df["close"] - df["low"]) / candle_range
    sma_48 = df["close"].rolling(48).mean()

    out["f_body_to_range"] = body_to_range
    out["f_upper_wick_to_range"] = upper_to_range
    out["f_lower_wick_to_range"] = lower_to_range
    out["f_close_pos_in_range"] = close_pos

    out["f_pat_doji"] = (body_to_range <= 0.10).astype("float32")
    out["f_pat_hammer"] = (
        (lower_wick_abs >= 2.0 * body_abs)
        & (upper_wick_abs <= 0.35 * body_abs + 1e-12)
        & (body_to_range <= 0.45)
    ).astype("float32")
    out["f_pat_shooting_star"] = (
        (upper_wick_abs >= 2.0 * body_abs)
        & (lower_wick_abs <= 0.35 * body_abs + 1e-12)
        & (body_to_range <= 0.45)
    ).astype("float32")
    out["f_pat_bull_engulf"] = (
        (df["close"] > df["open"])
        & (prev_close < prev_open)
        & (df["open"] <= prev_close)
        & (df["close"] >= prev_open)
    ).astype("float32")
    out["f_pat_bear_engulf"] = (
        (df["close"] < df["open"])
        & (prev_close > prev_open)
        & (df["open"] >= prev_close)
        & (df["close"] <= prev_open)
    ).astype("float32")
    out["f_pat_inside_bar"] = ((df["high"] < prev_high) & (df["low"] > prev_low)).astype("float32")
    out["f_pat_outside_bar"] = ((df["high"] > prev_high) & (df["low"] < prev_low)).astype("float32")
    out["f_pat_three_up"] = bullish.rolling(3).sum().eq(3).astype("float32")
    out["f_pat_three_down"] = bearish.rolling(3).sum().eq(3).astype("float32")
    out["f_pat_breakout_up_20"] = (df["close"] > df["high"].shift(1).rolling(20).max()).astype("float32")
    out["f_pat_breakout_down_20"] = (df["close"] < df["low"].shift(1).rolling(20).min()).astype("float32")
    out["f_pat_marubozu_bull"] = (
        (df["close"] > df["open"])
        & (body_to_range >= 0.80)
        & (upper_to_range <= 0.10)
        & (lower_to_range <= 0.10)
    ).astype("float32")
    out["f_pat_marubozu_bear"] = (
        (df["close"] < df["open"])
        & (body_to_range >= 0.80)
        & (upper_to_range <= 0.10)
        & (lower_to_range <= 0.10)
    ).astype("float32")
    out["f_pat_spinning_top"] = (
        (body_to_range <= 0.25)
        & (upper_to_range >= 0.30)
        & (lower_to_range >= 0.30)
    ).astype("float32")
    out["f_pat_harami_bull"] = (
        (df["close"] > df["open"])
        & (prev_close < prev_open)
        & (df["open"] >= prev_close)
        & (df["close"] <= prev_open)
    ).astype("float32")
    out["f_pat_harami_bear"] = (
        (df["close"] < df["open"])
        & (prev_close > prev_open)
        & (df["open"] <= prev_close)
        & (df["close"] >= prev_open)
    ).astype("float32")
    out["f_pat_tweezer_top"] = (
        (prev_close > prev_open)
        & (df["close"] < df["open"])
        & (((df["high"] - prev_high).abs() / prev_range) <= 0.10)
    ).astype("float32")
    out["f_pat_tweezer_bottom"] = (
        (prev_close < prev_open)
        & (df["close"] > df["open"])
        & (((df["low"] - prev_low).abs() / prev_range) <= 0.10)
    ).astype("float32")
    out["f_pat_gap_up"] = (df["low"] > prev_high).astype("float32")
    out["f_pat_gap_down"] = (df["high"] < prev_low).astype("float32")
    range_abs = (df["high"] - df["low"]).abs()
    out["f_pat_nr7"] = (range_abs <= (range_abs.rolling(7).min() + 1e-12)).astype("float32")
    out["f_pat_hammer_downtrend"] = (out["f_pat_hammer"] * (df["close"] < sma_48).astype("float32")).astype("float32")
    out["f_pat_shooting_star_uptrend"] = (
        out["f_pat_shooting_star"] * (df["close"] > sma_48).astype("float32")
    ).astype("float32")
    bull_streak = bullish.groupby((bullish == 0).cumsum()).cumsum()
    bear_streak = bearish.groupby((bearish == 0).cumsum()).cumsum()
    out["f_bull_streak"] = bull_streak.astype("float32")
    out["f_bear_streak"] = bear_streak.astype("float32")

    pattern_cols = [
        "f_pat_doji",
        "f_pat_hammer",
        "f_pat_shooting_star",
        "f_pat_bull_engulf",
        "f_pat_bear_engulf",
        "f_pat_inside_bar",
        "f_pat_outside_bar",
        "f_pat_three_up",
        "f_pat_three_down",
        "f_pat_breakout_up_20",
        "f_pat_breakout_down_20",
        "f_pat_marubozu_bull",
        "f_pat_marubozu_bear",
        "f_pat_spinning_top",
        "f_pat_harami_bull",
        "f_pat_harami_bear",
        "f_pat_tweezer_top",
        "f_pat_tweezer_bottom",
        "f_pat_gap_up",
        "f_pat_gap_down",
        "f_pat_nr7",
        "f_pat_hammer_downtrend",
        "f_pat_shooting_star_uptrend",
    ]
    pattern_freq_cols: dict[str, pd.Series] = {}
    for window in [12, 48, 144]:
        for col in pattern_cols:
            pattern_freq_cols[f"{col}_freq_{window}"] = out[col].rolling(window).mean()
    out = pd.concat([out, pd.DataFrame(pattern_freq_cols)], axis=1)

    lag_cols: dict[str, pd.Series] = {}
    for lag in [2, 3, 6, 12, 24, 48, 96]:
        lag_cols[f"f_ret_lag_{lag}"] = df["close"].pct_change(lag)
        lag_cols[f"f_vol_lag_{lag}"] = df["volume"].pct_change(lag)
    out = pd.concat([out, pd.DataFrame(lag_cols)], axis=1)

    roll_cols: dict[str, pd.Series] = {}
    for window in [3, 6, 12, 24, 48, 96, 288]:
        roll_cols[f"f_ret_mean_{window}"] = ret_1.rolling(window).mean()
        roll_cols[f"f_ret_std_{window}"] = ret_1.rolling(window).std()
        roll_cols[f"f_ret_skew_{window}"] = ret_1.rolling(window).skew()
        roll_cols[f"f_volume_mean_{window}"] = df["volume"].rolling(window).mean()
        roll_cols[f"f_volume_std_{window}"] = df["volume"].rolling(window).std()
        roll_cols[f"f_range_mean_{window}"] = spread.rolling(window).mean()
    out = pd.concat([out, pd.DataFrame(roll_cols)], axis=1)

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

        basis_cols: dict[str, pd.Series] = {}
        for window in [3, 6, 12, 24, 48, 96, 288]:
            basis_mean = out["f_basis_close"].rolling(window).mean()
            basis_std = out["f_basis_close"].rolling(window).std()
            basis_cols[f"f_basis_mean_{window}"] = basis_mean
            basis_cols[f"f_basis_std_{window}"] = basis_std
            basis_cols[f"f_basis_z_{window}"] = (out["f_basis_close"] - basis_mean) / (basis_std + 1e-12)
        out = pd.concat([out, pd.DataFrame(basis_cols)], axis=1)

    if reference_dfs:
        reference_perp_dfs = reference_perp_dfs or {}
        btc_ret_1 = ret_1.fillna(0.0)
        ref_feature_cols: dict[str, pd.Series] = {}
        for symbol in sorted(reference_dfs):
            ref = reference_dfs[symbol].copy()
            for col in ["open_time", "open", "high", "low", "close", "volume"]:
                if col in ref.columns:
                    ref[col] = pd.to_numeric(ref[col], errors="coerce")
            ref = ref[["open_time", "open", "high", "low", "close", "volume"]]
            ref = ref.sort_values("open_time").drop_duplicates("open_time")
            merged_ref = pd.DataFrame({"open_time": df["open_time"]}).merge(ref, on="open_time", how="left")

            prefix = _symbol_prefix(symbol)
            ref_available = merged_ref["close"].notna().astype("float32")
            ref_close = merged_ref["close"].ffill().fillna(df["close"])
            ref_open = merged_ref["open"].ffill().fillna(df["open"])
            ref_volume = merged_ref["volume"].fillna(0.0)
            ref_ret_1 = ref_close.pct_change().fillna(0.0)

            ref_feature_cols[f"f_ref_{prefix}_available"] = ref_available
            ref_feature_cols[f"f_ref_{prefix}_ret_1"] = ref_ret_1
            ref_feature_cols[f"f_ref_{prefix}_ret_3"] = ref_close.pct_change(3).fillna(0.0)
            ref_feature_cols[f"f_ref_{prefix}_ret_12"] = ref_close.pct_change(12).fillna(0.0)
            ref_feature_cols[f"f_ref_{prefix}_ret_spread_1"] = ref_ret_1 - btc_ret_1
            ref_feature_cols[f"f_ref_{prefix}_body"] = (ref_close - ref_open) / (ref_open + 1e-12)
            ref_feature_cols[f"f_ref_{prefix}_vol_ratio"] = ref_volume / (df["volume"] + 1e-12)
            ref_feature_cols[f"f_ref_{prefix}_price_ratio"] = ref_close / (df["close"] + 1e-12)
            ref_feature_cols[f"f_ref_{prefix}_corr_96"] = ref_ret_1.rolling(96).corr(btc_ret_1)
            ref_feature_cols[f"f_ref_{prefix}_corr_288"] = ref_ret_1.rolling(288).corr(btc_ret_1)
            beta_den_96 = btc_ret_1.rolling(96).var() + 1e-12
            beta_den_288 = btc_ret_1.rolling(288).var() + 1e-12
            ref_feature_cols[f"f_ref_{prefix}_beta_96"] = ref_ret_1.rolling(96).cov(btc_ret_1) / beta_den_96
            ref_feature_cols[f"f_ref_{prefix}_beta_288"] = ref_ret_1.rolling(288).cov(btc_ret_1) / beta_den_288

            ref_perp = reference_perp_dfs.get(symbol)
            if ref_perp is not None and not ref_perp.empty:
                rperp = ref_perp.copy()
                for col in ["open_time", "open", "close", "volume"]:
                    if col in rperp.columns:
                        rperp[col] = pd.to_numeric(rperp[col], errors="coerce")
                rperp = rperp[["open_time", "open", "close", "volume"]]
                rperp = rperp.sort_values("open_time").drop_duplicates("open_time")
                merged_perp = pd.DataFrame({"open_time": df["open_time"]}).merge(rperp, on="open_time", how="left")

                ref_perp_available = merged_perp["close"].notna().astype("float32")
                ref_perp_open = merged_perp["open"].ffill().fillna(ref_open)
                ref_perp_close = merged_perp["close"].ffill().fillna(ref_close)
                ref_perp_volume = merged_perp["volume"].fillna(0.0)
                ref_basis_close = (ref_perp_close - ref_close) / (ref_close + 1e-12)

                ref_feature_cols[f"f_ref_{prefix}_perp_available"] = ref_perp_available
                ref_feature_cols[f"f_ref_{prefix}_perp_ret_1"] = ref_perp_close.pct_change().fillna(0.0)
                ref_feature_cols[f"f_ref_{prefix}_basis_close"] = ref_basis_close
                ref_feature_cols[f"f_ref_{prefix}_basis_change"] = (
                    (ref_perp_close - ref_close) / (ref_close + 1e-12)
                    - (ref_perp_open - ref_open) / (ref_open + 1e-12)
                )
                ref_feature_cols[f"f_ref_{prefix}_perp_spot_vol_ratio"] = ref_perp_volume / (ref_volume + 1e-12)
                for window in [12, 48, 144]:
                    basis_mean = ref_basis_close.rolling(window).mean()
                    basis_std = ref_basis_close.rolling(window).std()
                    ref_feature_cols[f"f_ref_{prefix}_basis_z_{window}"] = (
                        (ref_basis_close - basis_mean) / (basis_std + 1e-12)
                    )

        if ref_feature_cols:
            out = pd.concat([out, pd.DataFrame(ref_feature_cols)], axis=1)

    target_cols: list[str] = []
    target_map: dict[str, pd.Series] = {}
    for h in horizons:
        col = f"target_h{h}"
        target_map[col] = (df["close"].shift(-h) > df["open"].shift(-h)).astype("float64")
        target_cols.append(col)
    out = pd.concat([out, pd.DataFrame(target_map)], axis=1)

    feature_cols = [c for c in out.columns if c.startswith("f_")]
    out = out.replace([np.inf, -np.inf], np.nan)
    cols_for_na_check = feature_cols + target_cols
    out = out.dropna(subset=cols_for_na_check).reset_index(drop=True)

    for target_col in target_cols:
        out[target_col] = out[target_col].astype("int8")

    for feature_col in feature_cols:
        out[feature_col] = out[feature_col].astype("float32")

    return out, feature_cols, target_cols
