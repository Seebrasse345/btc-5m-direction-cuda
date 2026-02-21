from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]
INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _to_ms_timestamp(date_text: str) -> int:
    return int(pd.Timestamp(date_text, tz="UTC").timestamp() * 1000)


def _clean_klines(rows: Iterable[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if df.empty:
        return df

    numeric_cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop(columns=["ignore"])
    df = df.dropna(subset=["open_time", "open", "high", "low", "close"])
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    return df


def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "btc-direction-trainer/0.1"})
    return session


def iter_binance_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int | None = None,
    limit: int = 1000,
    sleep_seconds: float = 0.06,
    max_retries: int = 8,
    endpoint_url: str = BINANCE_KLINES_URL,
):
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Unsupported interval: {interval}")

    interval_ms = INTERVAL_TO_MS[interval]
    end_ms = _utc_now_ms() if end_ms is None else end_ms
    estimated_rows = max(0, (end_ms - start_ms) // interval_ms)
    estimated_calls = max(1, math.ceil(estimated_rows / limit))

    session = _new_session()
    next_start = start_ms

    with tqdm(total=estimated_calls, desc="Downloading klines", unit="req") as pbar:
        while next_start < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": next_start,
                "endTime": end_ms,
                "limit": limit,
            }
            payload = None
            for attempt in range(max_retries):
                try:
                    response = session.get(endpoint_url, params=params, timeout=45)
                    if response.status_code in (418, 429):
                        retry_after = int(response.headers.get("Retry-After", "2"))
                        time.sleep(max(2, retry_after))
                        continue

                    response.raise_for_status()
                    payload = response.json()
                    if isinstance(payload, dict):
                        raise RuntimeError(f"Unexpected Binance payload: {payload}")
                    break
                except (requests.RequestException, RuntimeError):
                    if attempt >= max_retries - 1:
                        raise
                    wait_seconds = min(90, 2 ** attempt)
                    time.sleep(wait_seconds)
                    session.close()
                    session = _new_session()

            if payload is None:
                break
            if not payload:
                break

            chunk = _clean_klines(payload)
            if not chunk.empty:
                yield chunk

            last_open = int(payload[-1][0])
            proposed_next = last_open + interval_ms
            if proposed_next <= next_start:
                break

            next_start = proposed_next
            pbar.update(1)
            time.sleep(sleep_seconds)

    session.close()


def download_binance_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int | None = None,
    limit: int = 1000,
    sleep_seconds: float = 0.06,
    max_retries: int = 8,
    endpoint_url: str = BINANCE_KLINES_URL,
) -> pd.DataFrame:
    chunks = list(
        iter_binance_klines(
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=limit,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
            endpoint_url=endpoint_url,
        )
    )
    if not chunks:
        return pd.DataFrame(columns=KLINE_COLUMNS[:-1])
    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return out


def _merge_frames(existing: pd.DataFrame, fresh_chunks: list[pd.DataFrame]) -> pd.DataFrame:
    if not fresh_chunks:
        return existing.reset_index(drop=True)
    frames = [existing] if not existing.empty else []
    frames.extend(fresh_chunks)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return merged


def load_or_update_btc_data(
    output_path: Path,
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start_date: str = "2017-08-17",
    force_full: bool = False,
    checkpoint_every_requests: int = 100,
) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force_full:
        existing = pd.read_parquet(output_path)
        existing = existing.sort_values("open_time").drop_duplicates("open_time")
        start_ms = int(existing["open_time"].max()) + INTERVAL_TO_MS[interval]
    else:
        existing = pd.DataFrame(columns=KLINE_COLUMNS[:-1])
        start_ms = _to_ms_timestamp(start_date)

    end_ms = _utc_now_ms()
    if start_ms >= end_ms:
        return existing.reset_index(drop=True)

    request_count = 0
    buffer: list[pd.DataFrame] = []

    for chunk in iter_binance_klines(
        symbol=symbol,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
        endpoint_url=BINANCE_KLINES_URL,
    ):
        buffer.append(chunk)
        request_count += 1
        if request_count % checkpoint_every_requests == 0:
            existing = _merge_frames(existing, buffer)
            existing.to_parquet(output_path, index=False)
            buffer.clear()

    existing = _merge_frames(existing, buffer)
    if existing.empty:
        raise RuntimeError("No kline data was downloaded.")

    existing.to_parquet(output_path, index=False)
    return existing


def load_or_update_futures_data(
    output_path: Path,
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start_date: str = "2019-09-08",
    force_full: bool = False,
    checkpoint_every_requests: int = 100,
) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force_full:
        existing = pd.read_parquet(output_path)
        existing = existing.sort_values("open_time").drop_duplicates("open_time")
        start_ms = int(existing["open_time"].max()) + INTERVAL_TO_MS[interval]
    else:
        existing = pd.DataFrame(columns=KLINE_COLUMNS[:-1])
        start_ms = _to_ms_timestamp(start_date)

    end_ms = _utc_now_ms()
    if start_ms >= end_ms:
        return existing.reset_index(drop=True)

    request_count = 0
    buffer: list[pd.DataFrame] = []

    for chunk in iter_binance_klines(
        symbol=symbol,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
        endpoint_url=BINANCE_FUTURES_KLINES_URL,
    ):
        buffer.append(chunk)
        request_count += 1
        if request_count % checkpoint_every_requests == 0:
            existing = _merge_frames(existing, buffer)
            existing.to_parquet(output_path, index=False)
            buffer.clear()

    existing = _merge_frames(existing, buffer)
    if existing.empty:
        raise RuntimeError("No futures kline data was downloaded.")

    existing.to_parquet(output_path, index=False)
    return existing
