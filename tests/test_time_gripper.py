"""Timing helpers: mean±sd and depth RMSE."""

from __future__ import annotations

import numpy as np

from time_gripper import depth_rmse, mean_sd, timed


def test_mean_sd_two_values() -> None:
    """Sample sd of 1 and 3 is sqrt(2)."""
    mean, sd = mean_sd([1.0, 3.0])
    assert abs(mean - 2.0) < 1e-9
    assert abs(sd - float(np.sqrt(2.0))) < 1e-9


def test_mean_sd_one_value() -> None:
    """A single sample has sd 0."""
    mean, sd = mean_sd([4.0])
    assert mean == 4.0
    assert sd == 0.0


def test_depth_rmse_identical() -> None:
    """Matching valid pixels give RMSE 0."""
    a = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    assert depth_rmse(a, a) == 0.0


def test_timed_returns_elapsed() -> None:
    """timed() returns the function result and a non-negative duration."""
    val, sec = timed(lambda: 7)
    assert val == 7
    assert sec >= 0.0
