"""Hardware-free checks for RealSense capture helpers and dump ingest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture_realsense import (  # noqa: E402
    DEFAULT_LOCK,
    parse_args,
    ranked_stream_pairs,
    raw_depth_preview,
    self_test_dump,
    write_capture_dump,
    _rs,
)
from helpers import convert_realsense_dump  # noqa: E402


def test_ranked_stream_pairs_prefers_same_size_then_fps() -> None:
    """Matched 1280x720 beats a mixed pair; higher color fps wins the tie."""
    depth = [(1280, 720, 6), (640, 480, 30)]
    color = [(1280, 720, 15), (1280, 720, 30), (640, 480, 30)]
    ranked = ranked_stream_pairs(depth, color)
    assert ranked[0] == ((1280, 720, 6), (1280, 720, 30))
    assert ranked[1] == ((1280, 720, 6), (1280, 720, 15))


def test_raw_depth_preview_keeps_zeros_black() -> None:
    """Invalid Z stays black; a 1 m pixel is in range on a 0–2 m jet."""
    depth = np.array([[0, 1000], [0, 0]], dtype=np.uint16)
    vis, frac = raw_depth_preview(depth, 0.001, 0.0, 2.0)
    assert vis.shape == (2, 2, 3)
    assert abs(frac - 0.25) < 1e-9
    assert int(vis[0, 0].sum()) == 0
    assert int(vis[0, 1].sum()) > 0


def test_write_capture_dump_roundtrips_convert(tmp_path: Path) -> None:
    """A written dump is ingestible: same K and meters depth as the JSON."""
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[1, 2] = (10, 20, 30)
    depth = np.zeros((3, 4), dtype=np.uint16)
    depth[1, 2] = 2000
    meta = {
        "fx": 10.0, "fy": 11.0, "cx": 1.5, "cy": 1.0, "depth_scale_m_per_unit": 0.001,
        "device": "test",
    }
    write_capture_dump(tmp_path, "a", rgb, depth, meta)
    assert (tmp_path / "a_rgb.png").is_file()
    assert (tmp_path / "a_depth.png").is_file()
    assert json.loads((tmp_path / "a.json").read_text())["fx"] == 10.0
    images_dir = tmp_path / "images"
    frames, depths, ks, stems = convert_realsense_dump(
        tmp_path, images_dir, rgb_suffix="_rgb.png", depth_suffix="_depth.png", meta_suffix=".json",
    )
    assert stems == ["a"]
    assert [p.name for p in frames] == ["0.png"]
    assert abs(float(ks[0, 0, 0]) - 10.0) < 1e-9
    assert abs(float(depths[0, 1, 2]) - 2.0) < 1e-6
    # cv2.imwrite is BGR; convert_realsense_dump reads RGB, so channels swap.
    assert tuple(Image.open(frames[0]).getpixel((2, 1))) == (30, 20, 10)


def test_parse_args_and_lock_defaults() -> None:
    """Default lock is env/camera_lock.json with High Accuracy + laser 150."""
    args = parse_args([])
    assert args.lock == DEFAULT_LOCK
    assert args.self_test is False
    lock = json.loads(DEFAULT_LOCK.read_text())
    assert lock["visual_preset"] == "High Accuracy"
    assert abs(float(lock["depth"]["laser_power"]) - 150.0) < 1e-9


def test_self_test_dump_ingests_without_realsense(tmp_path: Path) -> None:
    """Synthetic SPACE dump is readable by convert_realsense_dump."""
    rgb_path, depth_path, json_path = self_test_dump(tmp_path)
    assert rgb_path.is_file() and depth_path.is_file() and json_path.is_file()
    assert parse_args(["--self-test", "--out", str(tmp_path / "cli")]).self_test


def test_rs_import_fails_without_sdk() -> None:
    """No camera SDK in the repo venv: _rs raises, it does not crash on import."""
    try:
        _rs()
    except RuntimeError as exc:
        assert "pyrealsense2" in str(exc)
    except Exception as exc:
        raise AssertionError(f"expected RuntimeError, got {type(exc)}: {exc}") from exc
