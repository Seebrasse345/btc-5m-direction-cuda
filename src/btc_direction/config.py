from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    start_date: str = "2017-08-17"
    random_seed: int = 42
    horizons: tuple[int, ...] = (1, 3, 6, 12)
    horizon_weights: tuple[float, ...] = (0.40, 0.25, 0.20, 0.15)
    n_splits: int = 5
    holdout_fraction: float = 0.20
    test_size: int = 30000
    num_boost_round: int = 1600
    early_stopping_rounds: int = 120

    @property
    def raw_data_path(self) -> Path:
        return self.root / "data" / "raw" / "btcusdt_5m.parquet"

    @property
    def features_path(self) -> Path:
        return self.root / "data" / "processed" / "btcusdt_5m_features.parquet"

    @property
    def perp_raw_data_path(self) -> Path:
        return self.root / "data" / "raw" / "btcusdt_perp_5m.parquet"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"
