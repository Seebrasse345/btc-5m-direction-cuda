from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SignalThresholds:
    lower: float
    upper: float
    objective_score: float


def make_directional_signals(proba: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError("Expected thresholds 0 <= lower < upper <= 1.")
    signal = np.zeros(len(proba), dtype=np.int8)
    signal[proba >= upper] = 1
    signal[proba <= lower] = -1
    return signal


def signal_metrics(y_true: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    active = signal != 0
    coverage = float(active.mean()) if len(signal) > 0 else 0.0
    if not np.any(active):
        return {
            "coverage": coverage,
            "win_rate": float("nan"),
            "long_win_rate": float("nan"),
            "short_win_rate": float("nan"),
            "num_trades": 0.0,
        }

    y_active = y_true[active]
    s_active = signal[active]
    wins = ((s_active == 1) & (y_active == 1)) | ((s_active == -1) & (y_active == 0))

    longs = s_active == 1
    shorts = s_active == -1
    long_wins = ((s_active == 1) & (y_active == 1))[longs] if np.any(longs) else np.array([])
    short_wins = ((s_active == -1) & (y_active == 0))[shorts] if np.any(shorts) else np.array([])

    return {
        "coverage": coverage,
        "win_rate": float(wins.mean()),
        "long_win_rate": float(long_wins.mean()) if len(long_wins) > 0 else float("nan"),
        "short_win_rate": float(short_wins.mean()) if len(short_wins) > 0 else float("nan"),
        "num_trades": float(active.sum()),
    }


def optimize_signal_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    min_coverage: float = 0.15,
    lower_grid: np.ndarray | None = None,
    upper_grid: np.ndarray | None = None,
) -> SignalThresholds:
    mask = ~np.isnan(proba)
    y = y_true[mask]
    p = proba[mask]
    if len(y) == 0:
        return SignalThresholds(lower=0.45, upper=0.55, objective_score=float("-inf"))

    lower_grid = np.linspace(0.05, 0.495, 90) if lower_grid is None else lower_grid
    upper_grid = np.linspace(0.505, 0.95, 90) if upper_grid is None else upper_grid

    best = SignalThresholds(lower=0.45, upper=0.55, objective_score=float("-inf"))

    for lower in lower_grid:
        for upper in upper_grid:
            if upper <= lower:
                continue
            signal = make_directional_signals(p, lower=float(lower), upper=float(upper))
            m = signal_metrics(y, signal)
            coverage = m["coverage"]
            win_rate = m["win_rate"]
            if np.isnan(win_rate):
                continue

            coverage_shortfall = max(0.0, min_coverage - coverage)
            # Penalize low participation so the policy doesn't collapse to near-zero trade counts.
            score = float(win_rate - 2.0 * coverage_shortfall + 0.05 * coverage)
            if score > best.objective_score:
                best = SignalThresholds(lower=float(lower), upper=float(upper), objective_score=score)

    return best
