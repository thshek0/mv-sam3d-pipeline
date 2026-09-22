"""Mask-centered crop should keep the foreground and emit a square."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helpers import crop_pair_to_mask, image_center_square, mask_square  # noqa: E402


def test_mask_square_uses_bbox_center() -> None:
    """A blob in the lower-left should put the square center on that blob."""
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[14:19, 2:9] = 255
    cx, cy, side = mask_square(mask > 0, margin=1.0)
    assert cx == 5
    assert cy == 16
    assert side == 7


def test_crop_pair_is_square_and_keeps_foreground() -> None:
    """RGB and alpha stay aligned; output is square; the painted pixels remain."""
    rgb = Image.new("RGB", (40, 60), (10, 10, 10))
    rgba = Image.new("RGBA", (40, 60), (0, 0, 0, 0))
    for x in range(4, 16):
        for y in range(40, 52):
            rgb.putpixel((x, y), (200, 30, 30))
            rgba.putpixel((x, y), (200, 30, 30, 255))
    out_rgb, out_mask = crop_pair_to_mask(rgb, rgba, margin=1.0)
    assert out_rgb.size[0] == out_rgb.size[1]
    assert out_rgb.size == out_mask.size
    alpha = np.asarray(out_mask.getchannel("A"))
    assert int(np.count_nonzero(alpha)) == 12 * 12
    ys, xs = np.nonzero(alpha)
    assert abs(float(xs.mean()) - out_rgb.size[0] / 2.0) < 2.0
    assert abs(float(ys.mean()) - out_rgb.size[1] / 2.0) < 2.0


def test_image_center_square_uses_short_side() -> None:
    """No-mask crop is a square of min(width, height) at the photo center."""
    cx, cy, side = image_center_square(40, 60)
    assert (cx, cy, side) == (20, 30, 40)


def test_empty_or_missing_mask_falls_back_to_image_center() -> None:
    """Empty alpha or no mask should keep the DA3-style image-center square."""
    rgb = Image.new("RGB", (40, 60), (10, 10, 10))
    rgb.putpixel((20, 30), (200, 30, 30))
    empty = Image.new("RGBA", (40, 60), (0, 0, 0, 0))
    out_empty, mask_empty = crop_pair_to_mask(rgb, empty, margin=1.0)
    out_none, mask_none = crop_pair_to_mask(rgb, None, margin=1.0)
    assert out_empty.size == (40, 40)
    assert out_none.size == (40, 40)
    assert mask_empty is not None and mask_empty.size == (40, 40)
    assert mask_none is None
    assert out_empty.getpixel((20, 20)) == (200, 30, 30)
    assert out_none.getpixel((20, 20)) == (200, 30, 30)


if __name__ == "__main__":
    test_mask_square_uses_bbox_center()
    test_crop_pair_is_square_and_keeps_foreground()
    test_image_center_square_uses_short_side()
    test_empty_or_missing_mask_falls_back_to_image_center()
    print("ok")
