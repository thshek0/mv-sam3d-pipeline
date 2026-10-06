"""``visualize_depth`` checks npz shapes and writes preview PNGs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from vis import (
    DEPTH_DISPLAY_RANGE_M,
    _colorize_depth,
    _orbit_points,
    append_turbo_colorbar,
    depth_range_m,
    depth_to_pointmap,
    load_da3_npz,
    visualize_depth,
    write_depth_visibility_preview,
    write_depthmap_preview,
)


def _tiny_npz(path: Path, images_dir: Path) -> None:
    """Write a 2-view pinhole npz plus matching RGB stills."""
    height, width = 8, 8
    depth = np.ones((2, height, width), dtype=np.float32) * 1.5
    k_mat = np.array([[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    intrinsics = np.stack([k_mat, k_mat], axis=0)
    pointmaps = np.stack([depth_to_pointmap(depth[0], k_mat), depth_to_pointmap(depth[1], k_mat)], axis=0)
    pointmaps_sam3d = np.transpose(pointmaps, (0, 3, 1, 2))
    ext0 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    ext1 = np.array([[0.0, 0.0, 1.0, 0.2], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.1]], dtype=np.float32)
    images_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (180, 40, 40)).save(images_dir / "0.png")
    Image.new("RGB", (width, height), (40, 40, 180)).save(images_dir / "1.png")
    np.savez(
        path,
        depth=depth,
        pointmaps=pointmaps,
        pointmaps_sam3d=pointmaps_sam3d,
        extrinsics=np.stack([ext0, ext1], axis=0),
        intrinsics=intrinsics,
        image_files=np.array([str(images_dir / "0.png"), str(images_dir / "1.png")]),
        process_res=height,
    )


def test_visualize_depth_writes_both_previews() -> None:
    """A valid npz writes per-view and orbit PNGs."""
    with tempfile.TemporaryDirectory() as tmp_s:
        root = Path(tmp_s)
        npz = root / "da3_output.npz"
        images = root / "images"
        _tiny_npz(npz, images)
        out = root / "preview.png"
        written = visualize_depth(npz, out, images_dir=images, mode="both")
        names = {path.name for path in written}
        assert names == {"preview.png", "preview_orbit.png"}
        for path in written:
            assert path.is_file()
            assert path.stat().st_size > 0


def test_visualize_orbit_object_only_keeps_mask_pixels() -> None:
    """Orbit drops background pixels when object/ alpha is set."""
    with tempfile.TemporaryDirectory() as tmp_s:
        root = Path(tmp_s)
        npz = root / "da3_output.npz"
        images = root / "images"
        masks = root / "object"
        _tiny_npz(npz, images)
        masks.mkdir()
        fg = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        for y in range(2):
            for x in range(2):
                fg.putpixel((x, y), (255, 0, 0, 255))
        fg.save(masks / "0.png")
        fg.save(masks / "1.png")
        data = load_da3_npz(npz)
        full, _ = _orbit_points(data, images, None, None, mask_dir=None)
        obj, _ = _orbit_points(data, images, None, None, mask_dir=masks)
        assert obj.shape[0] < full.shape[0]
        assert obj.shape[0] == 8
        written = visualize_depth(npz, root / "preview.png", images_dir=images, mode="orbit")
        assert written[0].name == "preview.png"
        assert written[0].is_file()
        assert written[0].stat().st_size > 0


def test_visualize_depth_rejects_bad_pointmaps() -> None:
    """Stored pointmaps that do not match depth+K raise."""
    with tempfile.TemporaryDirectory() as tmp_s:
        root = Path(tmp_s)
        npz = root / "da3_output.npz"
        images = root / "images"
        _tiny_npz(npz, images)
        data = dict(np.load(npz))
        data["pointmaps_sam3d"] = data["pointmaps_sam3d"] + 1.0
        bad = root / "bad.npz"
        np.savez(bad, **data)
        try:
            visualize_depth(bad, root / "out.png", images_dir=images, mode="per_view")
        except ValueError as exc:
            assert "pointmaps_sam3d" in str(exc)
        else:
            raise AssertionError("expected ValueError for inconsistent pointmaps")


def test_depth_range_m_skips_zero() -> None:
    """Z=0 is invalid; range is the finite positive min/max in meters."""
    z = np.array([[[0.0, 1.0], [2.0, np.nan]]], dtype=np.float32)
    assert depth_range_m(z) == (1.0, 2.0)


def test_colorize_clips_to_display_range() -> None:
    """Values past 3 m map to the same turbo end as 3 m; 0.2 m is the start."""
    vmin, vmax = DEPTH_DISPLAY_RANGE_M
    lo = _colorize_depth(np.array([[vmin]], dtype=np.float32), vmin=vmin, vmax=vmax)
    below = _colorize_depth(np.array([[0.05]], dtype=np.float32), vmin=vmin, vmax=vmax)
    hi = _colorize_depth(np.array([[vmax]], dtype=np.float32), vmin=vmin, vmax=vmax)
    clip = _colorize_depth(np.array([[20.0]], dtype=np.float32), vmin=vmin, vmax=vmax)
    zero = _colorize_depth(np.array([[0.0]], dtype=np.float32), vmin=vmin, vmax=vmax)
    assert np.array_equal(lo, below)
    assert not np.array_equal(lo, hi)
    assert np.array_equal(hi, clip)
    assert np.array_equal(zero, np.array([[[20, 20, 20]]], dtype=np.uint8))


def test_append_turbo_colorbar_labels_meters(tmp_path: Path) -> None:
    """Colorbar strip is taller than the board."""
    image = Image.new("RGB", (200, 30), (10, 10, 10))
    out = append_turbo_colorbar(image, 0.40, 1.80)
    assert out.size[0] == 200
    assert out.size[1] > 30


def test_depthmap_and_valid_boards_write_legends(tmp_path: Path) -> None:
    """Depthmap gets a meter bar; valid board is taller than the raw grid."""
    images = tmp_path / "images"
    masks = tmp_path / "object"
    images.mkdir()
    masks.mkdir()
    Image.new("RGB", (8, 8), (20, 20, 20)).save(images / "0.png")
    Image.new("RGBA", (8, 8), (10, 10, 10, 255)).save(masks / "0.png")
    depth = np.zeros((1, 8, 8), dtype=np.float32)
    depth[0, 2:6, 2:6] = 1.25
    d_png = write_depthmap_preview(depth, [images / "0.png"], tmp_path / "d.png")
    v_png = write_depth_visibility_preview(depth, [images / "0.png"], masks, tmp_path / "v.png")
    assert Image.open(d_png).size[1] > 256
    assert Image.open(v_png).size[1] > 256
