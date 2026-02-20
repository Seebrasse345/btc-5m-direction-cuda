from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from btc_direction.config import PipelineConfig
from btc_direction.data import load_or_update_btc_data


def parse_args() -> argparse.Namespace:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description="Download/update BTCUSDT 5m candles from Binance.")
    parser.add_argument("--symbol", default=cfg.symbol, type=str)
    parser.add_argument("--interval", default=cfg.interval, type=str)
    parser.add_argument("--start-date", default=cfg.start_date, type=str)
    parser.add_argument("--output", default=str(cfg.raw_data_path), type=str)
    parser.add_argument("--force-full", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)

    df = load_or_update_btc_data(
        output_path=output,
        symbol=args.symbol.upper(),
        interval=args.interval,
        start_date=args.start_date,
        force_full=args.force_full,
    )

    start_utc = df["open_time"].min()
    end_utc = df["open_time"].max()
    print(f"Saved {len(df):,} rows to {output}")
    print(f"Open time range (ms): {start_utc} -> {end_utc}")


if __name__ == "__main__":
    main()
