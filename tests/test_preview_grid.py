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
