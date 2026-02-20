from __future__ import annotations

from sklearn.model_selection import TimeSeriesSplit


def build_time_series_cv(
    n_rows: int,
    n_splits: int,
    test_size: int,
    gap: int,
) -> TimeSeriesSplit:
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    if test_size <= 0:
        raise ValueError("test_size must be positive.")
    if gap < 0:
        raise ValueError("gap must be non-negative.")

    max_possible_splits = (n_rows - gap) // test_size - 1
    if max_possible_splits < 2:
        raise ValueError(
            f"Not enough rows ({n_rows}) for test_size={test_size} and gap={gap}. "
            "Reduce test_size or n_splits."
        )

    actual_splits = min(n_splits, max_possible_splits)
    return TimeSeriesSplit(
        n_splits=actual_splits,
        test_size=test_size,
        gap=gap,
    )
