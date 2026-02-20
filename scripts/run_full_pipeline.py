from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end BTC direction pipeline.")
    parser.add_argument("--force-full-download", action="store_true")
    parser.add_argument("--with-lstm", action="store_true")
    parser.add_argument("--with-blend", action="store_true")
    parser.add_argument("--python", default=sys.executable, type=str)
    return parser.parse_args()


def run_cmd(cmd: list[str], root: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    env_python = args.python

    download_cmd = [env_python, "scripts/download_5m_data.py"]
    if args.force_full_download:
        download_cmd.append("--force-full")
    train_cmd = [env_python, "scripts/train_multihorizon.py"]
    lstm_cmd = [env_python, "scripts/train_lstm_multihorizon.py"]
    blend_cmd = [env_python, "scripts/blend_xgb_lstm.py"]

    run_cmd(download_cmd, root)
    run_cmd(train_cmd, root)
    if args.with_lstm:
        run_cmd(lstm_cmd, root)
    if args.with_blend:
        run_cmd(blend_cmd, root)


if __name__ == "__main__":
    main()
