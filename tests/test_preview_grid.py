"""Dataset previews are a 3-column photo | overlay grid of every view."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helpers import PREVIEW_COLS, PREVIEW_THUMB, write_pair_preview  # noqa: E402
from tags import overlay_tags, save_tag_preview  # noqa: E402
from vis import write_depth_visibility_preview  # noqa: E402


def test_write_pair_preview_always_three_columns() -> None:
    """Two pairs still occupy three pair-columns; both views are drawn."""
    with tempfile.TemporaryDirectory() as tmp_s:
        out = Path(tmp_s) / "grid.png"
        a = Image.new("RGB", (20, 10), (255, 0, 0))
        b = Image.new("RGB", (20, 10), (0, 255, 0))
        write_pair_preview(
            [(a, b, "0 photo", "0 x"), (a, b, "1 photo", "1 x")],
            out,
        )
        im = Image.open(out)
        assert im.size == (PREVIEW_COLS * PREVIEW_THUMB * 2, PREVIEW_THUMB)


def test_save_tag_preview_is_pairwise() -> None:
    """Tags preview is photo | overlay and includes every still."""
    with tempfile.TemporaryDirectory() as tmp_s:
        root = Path(tmp_s)
        images = []
        dets = []
        k_mat = np.array([[20.0, 0.0, 8.0], [0.0, 20.0, 6.0], [0.0, 0.0, 1.0]])
        for i in range(4):
            path = root / f"{i}.png"
            Image.new("RGB", (16, 12), (30, 30, 30)).save(path)
            images.append(path)
            dets.append({
                (i, 0): np.array([2.0, 2.0]),
                (i, 1): np.array([10.0, 2.0]),
                (i, 2): np.array([10.0, 8.0]),
                (i, 3): np.array([2.0, 8.0]),
            })
        out = root / "tags_preview.png"
        save_tag_preview(images, dets, out, ks=[k_mat] * 4)
        im = Image.open(out)
        assert im.size == (PREVIEW_COLS * PREVIEW_THUMB * 2, 2 * PREVIEW_THUMB)
        marked = overlay_tags(Image.open(images[0]), dets[0], k_mat)
        arr = np.asarray(marked)
        base = np.asarray(Image.open(images[0]))
        assert arr.sum() > base.sum()
        assert int((arr[..., 0] > arr[..., 1] + 20).sum()) > 0
        assert int((arr[..., 1] > arr[..., 0] + 20).sum()) > 0


def test_depth_visibility_preview_is_pairwise() -> None:
    """Depth visibility preview is photo | overlay for every depth view."""
    with tempfile.TemporaryDirectory() as tmp_s:
        root = Path(tmp_s)
        images_dir = root / "images"
        mask_dir = root / "object"
        images_dir.mkdir()
        mask_dir.mkdir()
        depth = np.ones((2, 8, 8), dtype=np.float32)
        depth[0, 0, 0] = 0.0
        paths = []
        for i in range(2):
            rgb = images_dir / f"{i}.png"
            Image.new("RGB", (8, 8), (80, 80, 80)).save(rgb)
            mask = Image.new("L", (8, 8), 0)
            mask.putpixel((0, 0), 255)
            mask.putpixel((1, 1), 255)
            mask.save(mask_dir / f"{i}.png")
            paths.append(rgb)
        out = root / "depth_visibility_preview.png"
        write_depth_visibility_preview(depth, paths, mask_dir, out)
        im = Image.open(out)
        assert im.size == (PREVIEW_COLS * PREVIEW_THUMB * 2, PREVIEW_THUMB)
