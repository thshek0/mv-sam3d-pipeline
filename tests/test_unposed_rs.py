"""Unposed RealSense npz keeps depth + K without AprilTag extrinsics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from helpers import identity_w2c, load_realsense_depth_stack, realsense_dump_stems, write_unposed_rs_npz


def _write_dump(root: Path, stem: str, *, labelme: bool) -> None:
    """One tiny aligned RGB-D view."""
    Image.new("RGB", (4, 4), (10, 20, 30)).save(root / f"{stem}_rgb.png")
    Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16), mode="I;16").save(root / f"{stem}_depth.png")
    (root / f"{stem}.json").write_text(json.dumps({
        "fx": 2.0, "fy": 2.0, "cx": 1.5, "cy": 1.5, "depth_scale_m_per_unit": 0.001,
    }))
    if labelme:
        (root / f"{stem}_rgb.json").write_text("{}")


def test_identity_w2c_is_one_per_view() -> None:
    """Each view gets an identity 3x4."""
    ext = identity_w2c(2)
    assert ext.shape == (2, 3, 4)
    assert float(ext[0, 0, 0]) == 1.0
    assert float(ext[1, 2, 3]) == 0.0


def test_realsense_dump_stems_skips_unlabeled(tmp_path: Path) -> None:
    """``skip_incomplete`` keeps only views that have LabelMe sidecars."""
    _write_dump(tmp_path, "a", labelme=True)
    _write_dump(tmp_path, "b", labelme=False)
    stems = realsense_dump_stems(
        tmp_path, rgb_suffix="_rgb.png", depth_suffix="_depth.png", meta_suffix=".json",
        labelme_suffix="_rgb.json", skip_incomplete=True,
    )
    assert stems == ["a"]


def test_write_unposed_rs_npz_stores_depth_and_identity(tmp_path: Path) -> None:
    """Unposed npz has metric depth, K, and identity extrinsics."""
    _write_dump(tmp_path, "a", labelme=True)
    stems = realsense_dump_stems(
        tmp_path, rgb_suffix="_rgb.png", depth_suffix="_depth.png", meta_suffix=".json",
    )
    depths, ks = load_realsense_depth_stack(
        tmp_path, stems, depth_suffix="_depth.png", meta_suffix=".json",
    )
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(images / "0.png")
    npz = write_unposed_rs_npz(tmp_path / "rs.npz", depths, ks, [images / "0.png"], keep_stems=stems)
    raw = np.load(npz)
    assert raw["depth"].shape == (1, 4, 4)
    assert abs(float(raw["depth"][0, 0, 0]) - 1.0) < 1e-5
    assert int(raw["unposed_rs"]) == 1
    assert float(raw["extrinsics"][0, 0, 0]) == 1.0
