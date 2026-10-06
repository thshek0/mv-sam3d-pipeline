"""RealSense dump ingest, depth crop, and AprilTag npz without OpenCV."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from helpers import (
    convert_realsense_dump,
    crop_depth_intrinsics,
    crop_views_depth_to_masks,
)
from tags import convert_apriltag_to_da3_npz
from vis import _orbit_points, load_da3_npz, visualize_depth, write_da3_npz


def _identity_pose(_det: object, _world: object, _k: object) -> np.ndarray:
    """Return an identity 3x4 w2c (tests do not use OpenCV PnP)."""
    return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)


def test_convert_realsense_dump_aligned_z16() -> None:
    """Numbered RGB, meters depth, and K follow the dump suffixes."""
    with tempfile.TemporaryDirectory() as tmp_s:
        src = Path(tmp_s) / "dump"
        src.mkdir()
        rgb = Image.new("RGB", (4, 3), (10, 20, 30))
        depth_arr = np.array([[0, 1000, 0, 2000], [3000, 0, 4000, 0], [0, 5000, 0, 6000]], dtype=np.uint16)
        rgb.save(src / "a_rgb.png")
        Image.fromarray(depth_arr).save(src / "a_depth.png")
        (src / "a.json").write_text(json.dumps({
            "fx": 10.0, "fy": 11.0, "cx": 1.5, "cy": 1.0, "depth_scale_m_per_unit": 0.001,
        }))
        images_dir = Path(tmp_s) / "images"
        frames, depths, ks, stems = convert_realsense_dump(
            src, images_dir, rgb_suffix="_rgb.png", depth_suffix="_depth.png", meta_suffix=".json",
        )
        assert stems == ["a"]
        assert [p.name for p in frames] == ["0.png"]
        assert depths.shape == (1, 3, 4)
        assert abs(float(ks[0, 0, 0]) - 10.0) < 1e-9
        assert abs(float(depths[0, 0, 1]) - 1.0) < 1e-9
        assert float(depths[0, 0, 0]) == 0.0


def test_crop_depth_intrinsics_shifts_and_scales_k() -> None:
    """Crop origin moves cx/cy; process_res scales the first two rows of K."""
    depth = np.arange(20, dtype=np.float32).reshape(4, 5)
    k_mat = np.array([[10.0, 0.0, 2.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]])
    cropped, k_c = crop_depth_intrinsics(depth, k_mat, cx=2, cy=2, side=4, process_res=None)
    assert cropped.shape == (4, 4)
    x0, y0 = 2 - 2, 2 - 2
    assert abs(k_c[0, 2] - (2.0 - x0)) < 1e-9
    resized, k_r = crop_depth_intrinsics(depth, k_mat, cx=2, cy=2, side=4, process_res=8)
    assert resized.shape == (8, 8)
    assert abs(k_r[0, 0] - 20.0) < 1e-9


def test_crop_views_depth_to_masks_writes_npz() -> None:
    """Folder crop writes RGB/mask/depth at process_res and a valid npz."""
    with tempfile.TemporaryDirectory() as tmp_s:
        root = Path(tmp_s)
        images = root / "images"
        masks = root / "object"
        images.mkdir()
        masks.mkdir()
        Image.new("RGB", (8, 6), (30, 30, 30)).save(images / "0.png")
        rgba = Image.new("RGBA", (8, 6), (0, 0, 0, 0))
        for x in range(2, 5):
            for y in range(2, 5):
                rgba.putpixel((x, y), (255, 0, 0, 255))
        rgba.save(masks / "0.png")
        depth = np.ones((6, 8), dtype=np.float32)
        depth[0, 0] = 0.0
        k_mat = np.array([[8.0, 0.0, 3.5], [0.0, 8.0, 2.5], [0.0, 0.0, 1.0]])
        ext = np.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
        npz = crop_views_depth_to_masks(
            [images / "0.png"], masks, [depth], [k_mat], ext, root / "out.npz",
            margin=1.0, process_res=4,
        )
        data = load_da3_npz(npz)
        assert data["depth"].shape == (1, 4, 4)
        assert Image.open(images / "0.png").size == (4, 4)


def test_convert_apriltag_npz_with_injected_pose() -> None:
    """Two views with four corners each produce a 2-view npz without OpenCV."""
    with tempfile.TemporaryDirectory() as tmp_s:
        root = Path(tmp_s)
        images = root / "images"
        images.mkdir()
        Image.new("RGB", (8, 8), (1, 2, 3)).save(images / "0.png")
        Image.new("RGB", (8, 8), (4, 5, 6)).save(images / "1.png")
        depth = np.ones((8, 8), dtype=np.float32)
        depth[0, 0] = 0.0
        k_mat = np.array([[8.0, 0.0, 3.5], [0.0, 8.0, 3.5], [0.0, 0.0, 1.0]])
        corners = {(0, 0): np.array([2.0, 2.0]), (0, 1): np.array([5.0, 2.0]), (0, 2): np.array([5.0, 5.0]), (0, 3): np.array([2.0, 5.0])}
        npz, keep = convert_apriltag_to_da3_npz(
            [images / "0.png", images / "1.png"], [depth, depth], [k_mat, k_mat], root / "tags.npz",
            neigh=1, grow_steps=1, min_corners=4, min_posed=2, pnp_flags=0,
            detections=[corners, corners], pose_fn=_identity_pose, kept_images_dir=root / "kept",
        )
        assert keep == [0, 1]
        data = load_da3_npz(npz)
        assert data["depth"].shape[0] == 2
        assert not np.isfinite(data["pointmaps_sam3d"][0, 0, 0, 0])


def test_orbit_z_clip_drops_far_camera_z() -> None:
    """Orbit clip keeps near points and drops a far flyer (display only)."""
    with tempfile.TemporaryDirectory() as tmp_s:
        root = Path(tmp_s)
        depth = np.ones((1, 4, 4), dtype=np.float32) * 1.0
        depth[0, 0, 0] = 30.0
        k_mat = np.array([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32)
        ext = np.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]], dtype=np.float32)
        npz = write_da3_npz(root / "d.npz", depth, ext, k_mat[None, ...], ["0.png"], process_res=4)
        data = load_da3_npz(npz)
        far, _ = _orbit_points(data, None, None, None)
        near, _ = _orbit_points(data, None, 0.1, 2.0)
        assert far.shape[0] >= near.shape[0]
        written = visualize_depth(npz, root / "orb.png", mode="orbit", orbit_z_min=0.1, orbit_z_max=2.0)
        assert written[0].is_file()
