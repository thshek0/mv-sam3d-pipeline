"""Timing helpers and ``time_run`` folder / flag defaults."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from helpers import REPO, depth_rmse, mean_sd, timed
from time_run import DEFAULT_GRIPPER_OUT, DEFAULT_SCENE, default_out_dir, discard_run_dirs, parse_args


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


def test_default_out_dir_uses_scene_name() -> None:
    """Gripper example keeps the old archive path; other scenes use their folder name."""
    assert default_out_dir(DEFAULT_SCENE) == DEFAULT_GRIPPER_OUT
    assert default_out_dir(Path("/tmp/captures_20261002_1606")) == REPO / "archive" / "captures_20261002_1606_timed"


def test_discard_run_dirs_keeps_summary(tmp_path: Path) -> None:
    """``run_*`` and ``work/`` go away; ``summary.json`` stays."""
    (tmp_path / "run_00").mkdir()
    (tmp_path / "work" / "images").mkdir(parents=True)
    (tmp_path / "summary.json").write_text("{}\n")
    discard_run_dirs(tmp_path)
    assert not (tmp_path / "run_00").exists()
    assert not (tmp_path / "work").exists()
    assert (tmp_path / "summary.json").is_file()


def test_parse_args_keep_and_preview_defaults() -> None:
    """Runs and depth preview stay on unless the no- flags are passed."""
    keep = parse_args([])
    assert keep.keep_runs is True
    assert keep.preview is True
    drop = parse_args(["--no-keep-runs", "--no-preview"])
    assert drop.keep_runs is False
    assert drop.preview is False
