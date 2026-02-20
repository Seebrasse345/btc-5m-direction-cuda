from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots from training reports.")
    parser.add_argument("--reports-dir", default="reports", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    holdout_path = reports_dir / "holdout_metrics.csv"
    cv_path = reports_dir / "cv_metrics_by_fold.csv"

    if not holdout_path.exists() or not cv_path.exists():
        raise FileNotFoundError("Run training before generating analysis plots.")

    holdout_df = pd.read_csv(holdout_path)
    cv_df = pd.read_csv(cv_path)

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=holdout_df, x="horizon", y="auc", ax=ax)
    ax.set_title("Holdout AUC by Horizon")
    ax.set_xlabel("Horizon (candles)")
    ax.set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig(reports_dir / "holdout_auc_by_horizon.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=cv_df, x="fold", y="auc", hue="horizon", marker="o", ax=ax)
    ax.set_title("CV AUC by Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig(reports_dir / "cv_auc_by_fold.png", dpi=180)
    plt.close(fig)

    print(f"Saved plots to {reports_dir}")


if __name__ == "__main__":
    main()
