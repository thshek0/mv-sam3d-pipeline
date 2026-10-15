"""RS+DA3 combine: keep valid RS, fill remaining object pixels with scaled DA3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from helpers import append_color_legend
from methods import (
    SOURCE_MAP_LEGEND,
    depth_npz_is_unposed,
    fill_rs_holes,
    overlay_source,
    resize_depth,
    rs_valid,
)


def _lid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """8x8 object: lid Z=1, 2x2 hole, right-edge miss, DA3 plate + wall 1.3."""
    obj = np.zeros((8, 8), dtype=bool)
    obj[1:7, 1:7] = True
    rs = np.zeros((8, 8), dtype=np.float32)
    rs[obj] = 1.0
    rs[2:4, 2:4] = 0.0
    rs[1:7, 5:7] = 0.0
    da3 = np.zeros((8, 8), dtype=np.float32)
    da3[obj] = 1.0
    da3[1:7, 5:7] = 1.3
    return obj, rs, da3


def test_depth_npz_is_unposed_flags(tmp_path: Path) -> None:
    """``unposed_rs`` / ``unposed_hybrid`` mark identity-camera npzs."""
    from vis import write_da3_npz

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(images / "0.png")
    z = np.ones((1, 4, 4), dtype=np.float32)
    k_mat = np.eye(3, dtype=np.float32)[None, ...]
    ext = np.eye(4, dtype=np.float32)[:3][None, ...]
    posed = write_da3_npz(tmp_path / "posed.npz", z, ext, k_mat, [images / "0.png"], process_res=4)
    assert not depth_npz_is_unposed(posed)
    unposed = write_da3_npz(
        tmp_path / "rs.npz", z, ext, k_mat, [images / "0.png"], process_res=4,
        extras={"unposed_rs": np.asarray(1, dtype=np.int8)},
    )
    assert depth_npz_is_unposed(unposed)


def test_unposed_hybrid_uses_per_view_fill(tmp_path: Path) -> None:
    """Unposed RS + DA3 hybrid keeps RS and fills holes without free-space poses."""
    from methods import depth_hybrid
    from vis import write_da3_npz

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(images / "0.png")
    _obj, rs_z, da_z = _lid()
    k_mat = np.array([[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    ext = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    rs_npz = write_da3_npz(
        tmp_path / "rs.npz", rs_z[None, ...], ext[None, ...], k_mat[None, ...],
        [images / "0.png"], process_res=8, extras={"unposed_rs": np.asarray(1, dtype=np.int8)},
    )
    da_npz = write_da3_npz(
        tmp_path / "da.npz", da_z[None, ...], ext[None, ...], k_mat[None, ...],
        [images / "0.png"], process_res=8,
    )
    out = depth_hybrid(rs_npz, da_npz, images, tmp_path / "hybrid.npz")
    raw = np.load(out)
    assert int(raw["unposed_hybrid"]) == 1
    assert abs(float(raw["depth"][0, 1, 2]) - 1.0) < 1e-5
    assert abs(float(raw["depth"][0, 2, 2]) - 1.0) < 1e-5


def test_fill_rs_holes_keeps_rs_and_fills_object_holes() -> None:
    """Valid RS stays; object holes get scaled DA3; outside region keeps RS."""
    obj, rs, da3 = _lid()
    depth, scale = fill_rs_holes(rs, da3, obj)
    assert abs(scale - 1.0) < 1e-5
    assert abs(float(depth[1, 2]) - 1.0) < 1e-5
    assert abs(float(depth[2, 2]) - 1.0) < 1e-5
    assert abs(float(depth[3, 6]) - 1.3) < 1e-5
    assert float(depth[0, 0]) == 0.0


def test_fill_rs_holes_full_frame() -> None:
    """With no region, RS holes anywhere are filled with scaled DA3."""
    rs = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    da3 = np.array([[1.0, 1.0], [1.0, 1.5]], dtype=np.float32)
    depth, scale = fill_rs_holes(rs, da3)
    assert abs(scale - 2.0) < 1e-5
    assert abs(float(depth[0, 0]) - 2.0) < 1e-5
    assert abs(float(depth[0, 1]) - 2.0) < 1e-5
    assert abs(float(depth[1, 1]) - 3.0) < 1e-5


def test_fill_rs_holes_scales_da3_to_rs() -> None:
    """Overlap median RS/DA3 is applied to filled pixels."""
    rs = np.array([[2.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    da3 = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    depth, scale = fill_rs_holes(rs, da3)
    assert abs(scale - 2.0) < 1e-5
    assert abs(float(depth[1, 0]) - 2.0) < 1e-5


def test_resize_depth_nearest() -> None:
    """A 2x2 depth map scales to 4x4 without mixing zeros into valid pixels."""
    src = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
    out = resize_depth(src, 4, 4)
    assert out.shape == (4, 4)
    assert float(out[0, 0]) == 1.0
    assert float(out[0, 3]) == 2.0
    assert float(out[3, 0]) == 0.0
    assert float(out[3, 3]) == 3.0


def test_rs_valid_treats_zero_as_invalid() -> None:
    """RealSense writes 0 for invalid depth; only Z>0 counts."""
    z = np.array([[0.0, 1.2], [np.nan, -0.1]], dtype=np.float32)
    valid = rs_valid(z)
    assert bool(valid[0, 1])
    assert not bool(valid[0, 0])
    assert not bool(valid[1, 0])
    assert not bool(valid[1, 1])


def test_overlay_source_has_no_third_class() -> None:
    """Invalid RS pixels stay dim unless hybrid filled them from DA3."""
    rgb = Image.new("RGB", (2, 2), (80, 80, 80))
    rs = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    fused = np.array([[1.0, 1.5], [0.0, 0.0]], dtype=np.float32)
    out = np.asarray(overlay_source(rgb, rs, fused))
    assert tuple(out[0, 0]) != (80, 80, 80)
    assert tuple(int(c) for c in out[0, 1]) != (80, 80, 80)
    assert out[1, 0, 0] < 50 and out[1, 0, 1] < 50


def test_append_color_legend_paints_swatches() -> None:
    """Legend strip is taller than the board and contains the first swatch color."""
    image = Image.new("RGB", (120, 20), (10, 10, 10))
    out = append_color_legend(image, SOURCE_MAP_LEGEND)
    assert out.size[0] == 120
    assert out.size[1] > 20
    assert out.getpixel((27, 28)) == SOURCE_MAP_LEGEND[0][0]
