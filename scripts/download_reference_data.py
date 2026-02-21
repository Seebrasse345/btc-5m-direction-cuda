from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from btc_direction.data import load_or_update_btc_data, load_or_update_futures_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/update reference crypto symbols (spot + perp) at 5m from Binance."
    )
    parser.add_argument("--symbols", default="ETHUSDT,BNBUSDT", type=str)
    parser.add_argument("--interval", default="5m", type=str)
    parser.add_argument("--spot-start-date", default="2017-08-17", type=str)
    parser.add_argument("--perp-start-date", default="2019-09-08", type=str)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "raw"), type=str)
    parser.add_argument("--skip-futures", action="store_true")
    parser.add_argument("--force-full", action="store_true")
    return parser.parse_args()


def _parse_symbols(s: str) -> list[str]:
    out = [x.strip().upper() for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one symbol.")
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = _parse_symbols(args.symbols)
    for symbol in symbols:
        spot_path = output_dir / f"{symbol.lower()}_5m.parquet"
        spot_df = load_or_update_btc_data(
            output_path=spot_path,
            symbol=symbol,
            interval=args.interval,
            start_date=args.spot_start_date,
            force_full=args.force_full,
        )
        print(f"{symbol} spot rows={len(spot_df):,} path={spot_path}")

        if not args.skip_futures:
            perp_path = output_dir / f"{symbol.lower()}_perp_5m.parquet"
            perp_df = load_or_update_futures_data(
                output_path=perp_path,
                symbol=symbol,
                interval=args.interval,
                start_date=args.perp_start_date,
                force_full=args.force_full,
            )
            print(f"{symbol} perp rows={len(perp_df):,} path={perp_path}")


if __name__ == "__main__":
    main()

