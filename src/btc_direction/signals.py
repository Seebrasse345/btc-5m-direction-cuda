from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SignalThresholds:
    lower: float
    upper: float
    objective_score: float


@dataclass(slots=True)
class HighPrecisionPolicy:
    lower: float
    upper: float
    target_win_rate: float
    realized_win_rate: float
    realized_coverage: float
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

    best_valid: SignalThresholds | None = None
    best_fallback: SignalThresholds | None = None
    best_valid_score = float("-inf")
    best_fallback_score = float("-inf")

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

            if coverage >= min_coverage:
                score = float(win_rate + 0.05 * coverage)
                if score > best_valid_score:
                    best_valid_score = score
                    best_valid = SignalThresholds(lower=float(lower), upper=float(upper), objective_score=score)

            fallback_score = float(-abs(coverage - min_coverage) + 0.02 * win_rate)
            if fallback_score > best_fallback_score:
                best_fallback_score = fallback_score
                best_fallback = SignalThresholds(
                    lower=float(lower),
                    upper=float(upper),
                    objective_score=fallback_score,
                )

    if best_valid is not None:
        return best_valid
    if best_fallback is not None:
        return best_fallback
    return SignalThresholds(lower=0.45, upper=0.55, objective_score=float("-inf"))


def optimize_high_precision_policy(
    y_true: np.ndarray,
    proba: np.ndarray,
    target_win_rate: float,
    min_coverage: float = 0.001,
    center: float = 0.5,
    quantiles: np.ndarray | None = None,
) -> HighPrecisionPolicy:
    if target_win_rate <= 0.0 or target_win_rate >= 1.0:
        raise ValueError("target_win_rate must be in (0, 1).")

    mask = ~np.isnan(proba)
    y = y_true[mask]
    p = proba[mask]
    if len(y) == 0:
        return HighPrecisionPolicy(
            lower=0.45,
            upper=0.55,
            target_win_rate=target_win_rate,
            realized_win_rate=float("nan"),
            realized_coverage=0.0,
            objective_score=float("-inf"),
        )

    if quantiles is None:
        lower_q_grid = np.linspace(0.005, 0.25, 24)
        upper_q_grid = np.linspace(0.75, 0.995, 24)
        edge_q_grid = np.linspace(0.80, 0.999, 40)
    else:
        q = np.asarray(quantiles, dtype=np.float64)
        lower_q_grid = q[(q > 0.0) & (q < center)]
        upper_q_grid = q[(q > center) & (q < 1.0)]
        edge_q_grid = q[(q > 0.5) & (q < 1.0)]
        if lower_q_grid.size == 0:
            lower_q_grid = np.linspace(0.01, 0.25, 12)
        if upper_q_grid.size == 0:
            upper_q_grid = np.linspace(0.75, 0.99, 12)
        if edge_q_grid.size == 0:
            edge_q_grid = np.linspace(0.80, 0.995, 20)

    best_meeting: HighPrecisionPolicy | None = None
    best_meeting_coverage = float("-inf")
    best_meeting_win_rate = float("-inf")
    best_fallback: HighPrecisionPolicy | None = None
    best_fallback_score = float("-inf")

    # Asymmetric threshold search (lets long/short confidence differ).
    for lq in lower_q_grid:
        lower = float(np.quantile(p, lq))
        for uq in upper_q_grid:
            upper = float(np.quantile(p, uq))
            if upper <= lower:
                continue
            signal = make_directional_signals(p, lower=lower, upper=upper)
            m = signal_metrics(y, signal)
            coverage = float(m["coverage"])
            win_rate = float(m["win_rate"]) if not np.isnan(m["win_rate"]) else float("nan")
            if np.isnan(win_rate) or coverage <= 0.0:
                continue

            if coverage >= min_coverage and win_rate >= target_win_rate:
                if (coverage > best_meeting_coverage) or (
                    np.isclose(coverage, best_meeting_coverage) and win_rate > best_meeting_win_rate
                ):
                    best_meeting_coverage = coverage
                    best_meeting_win_rate = win_rate
                    best_meeting = HighPrecisionPolicy(
                        lower=lower,
                        upper=upper,
                        target_win_rate=target_win_rate,
                        realized_win_rate=win_rate,
                        realized_coverage=coverage,
                        objective_score=coverage,
                    )

            fallback_score = float(win_rate - abs(target_win_rate - win_rate) + 0.03 * np.sqrt(coverage))
            if coverage >= min_coverage and fallback_score > best_fallback_score:
                best_fallback_score = fallback_score
                best_fallback = HighPrecisionPolicy(
                    lower=lower,
                    upper=upper,
                    target_win_rate=target_win_rate,
                    realized_win_rate=win_rate,
                    realized_coverage=coverage,
                    objective_score=fallback_score,
                )

    # Symmetric margin search as backstop.
    edge = np.abs(p - center)
    for q in edge_q_grid:
        margin = float(np.quantile(edge, q))
        lower = float(center - margin)
        upper = float(center + margin)
        signal = make_directional_signals(p, lower=lower, upper=upper)
        m = signal_metrics(y, signal)
        coverage = float(m["coverage"])
        win_rate = float(m["win_rate"]) if not np.isnan(m["win_rate"]) else float("nan")
        if np.isnan(win_rate) or coverage <= 0.0:
            continue

        if coverage >= min_coverage and win_rate >= target_win_rate:
            if (coverage > best_meeting_coverage) or (
                np.isclose(coverage, best_meeting_coverage) and win_rate > best_meeting_win_rate
            ):
                best_meeting_coverage = coverage
                best_meeting_win_rate = win_rate
                best_meeting = HighPrecisionPolicy(
                    lower=lower,
                    upper=upper,
                    target_win_rate=target_win_rate,
                    realized_win_rate=win_rate,
                    realized_coverage=coverage,
                    objective_score=coverage,
                )

        fallback_score = float(win_rate - abs(target_win_rate - win_rate) + 0.03 * np.sqrt(coverage))
        if coverage >= min_coverage and fallback_score > best_fallback_score:
            best_fallback_score = fallback_score
            best_fallback = HighPrecisionPolicy(
                lower=lower,
                upper=upper,
                target_win_rate=target_win_rate,
                realized_win_rate=win_rate,
                realized_coverage=coverage,
                objective_score=fallback_score,
            )

    if best_meeting is not None:
        return best_meeting
    if best_fallback is not None:
        return best_fallback
    return HighPrecisionPolicy(
        lower=0.45,
        upper=0.55,
        target_win_rate=target_win_rate,
        realized_win_rate=float("nan"),
        realized_coverage=0.0,
        objective_score=float("-inf"),
    )
