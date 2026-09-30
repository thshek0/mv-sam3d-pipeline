"""``visualize_depth`` checks npz shapes and writes preview PNGs."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vis import _orbit_points, depth_to_pointmap, load_da3_npz, visualize_depth  # noqa: E402


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
