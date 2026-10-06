"""LabelMe mask crops must become full-frame RGBA for MV-SAM3D."""

from __future__ import annotations

import base64
import io
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from helpers import (
    labelme_json_for_stills,
    mask_from_labelme,
    resize_mask_to_rgb,
    rgba_from_labelme,
    write_labelme_masks,
)
from helpers import ingest_stills


def _crop_png_b64(arr: np.ndarray) -> str:
    """Encode a uint8 crop as a LabelMe-style base64 PNG."""
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_labelme_json_for_stills_all_or_nothing() -> None:
    """No JSON means rembg; a partial set is an error, not a mix."""
    with tempfile.TemporaryDirectory() as raw_s:
        raw = Path(raw_s)
        Image.new("RGB", (4, 4), (1, 2, 3)).save(raw / "a.png")
        Image.new("RGB", (4, 4), (4, 5, 6)).save(raw / "b.png")
        assert labelme_json_for_stills(raw) is None
        (raw / "a.json").write_text("{}")
        try:
            labelme_json_for_stills(raw)
            raise AssertionError("partial LabelMe should fail")
        except FileNotFoundError as exc:
            assert "b.json" in str(exc)
        (raw / "b.json").write_text("{}")
        jsons = labelme_json_for_stills(raw)
        assert jsons is not None
        assert [p.name for p in jsons] == ["a.json", "b.json"]


def test_mask_from_labelme_pastes_inclusive_bbox() -> None:
    """A 0/1 crop lands in the inclusive bbox and is scaled to 0/255."""
    crop = np.zeros((2, 3), dtype=np.uint8)
    crop[0, 1] = 1
    crop[1, 2] = 1
    data = {
        "imageWidth": 10,
        "imageHeight": 8,
        "shapes": [
            {
                "label": "gripper",
                "shape_type": "mask",
                "points": [[2, 3], [4, 4]],
                "mask": _crop_png_b64(crop),
            }
        ],
    }
    mask = np.asarray(mask_from_labelme(data, (10, 8)))
    assert mask.shape == (8, 10)
    assert mask[3, 3] == 255
    assert mask[4, 4] == 255
    assert mask[3, 2] == 0
    assert int(mask.sum()) == 255 * 2


def test_write_labelme_masks_matches_ingested_size() -> None:
    """RGBA masks share the ingested still size and keep the gripper pixels."""
    with tempfile.TemporaryDirectory() as raw_s, tempfile.TemporaryDirectory() as img_s, tempfile.TemporaryDirectory() as mask_s:
        raw = Path(raw_s)
        rgb = Image.new("RGB", (20, 10), (10, 20, 30))
        rgb.save(raw / "IMG_0001.png")
        crop = np.ones((4, 6), dtype=np.uint8)
        payload = {
            "version": "5.0.0",
            "imagePath": "IMG_0001.png",
            "imageWidth": 20,
            "imageHeight": 10,
            "shapes": [
                {
                    "label": "gripper",
                    "shape_type": "mask",
                    "points": [[2, 1], [7, 4]],
                    "mask": _crop_png_b64(crop),
                }
            ],
        }
        (raw / "IMG_0001.json").write_text(json.dumps(payload))
        frames = ingest_stills(raw, Path(img_s), max_side=10)
        masks = write_labelme_masks(raw, frames, Path(mask_s))
        assert [p.name for p in masks] == ["0.png"]
        rgba = Image.open(masks[0])
        assert rgba.mode == "RGBA"
        assert rgba.size == Image.open(frames[0]).size
        alpha = np.asarray(rgba.getchannel("A"))
        assert alpha.max() == 255
        assert int(np.count_nonzero(alpha)) > 0
        rebuilt = rgba_from_labelme(raw / "IMG_0001.json", Image.open(frames[0]))
        assert np.array_equal(np.asarray(rebuilt.getchannel("A")), alpha)


def test_resize_mask_rejects_transposed_rgb() -> None:
    """A portrait mask must not be squashed onto a landscape RGB."""
    mask = Image.new("L", (10, 20), 255)
    try:
        resize_mask_to_rgb(mask, (20, 10))
        raise AssertionError("transposed mask should fail")
    except ValueError as exc:
        assert "transposed" in str(exc)


def test_write_labelme_masks_follows_exif_display_size() -> None:
    """LabelMe canvas is display-oriented; ingest with EXIF 6 must match it."""
    with tempfile.TemporaryDirectory() as raw_s, tempfile.TemporaryDirectory() as img_s, tempfile.TemporaryDirectory() as mask_s:
        raw = Path(raw_s)
        im = Image.new("RGB", (20, 10), (10, 20, 30))
        exif = Image.Exif()
        exif[0x0112] = 6
        im.save(raw / "IMG_0001.jpg", format="JPEG", quality=95, exif=exif)
        crop = np.ones((4, 3), dtype=np.uint8)
        payload = {
            "version": "5.0.0",
            "imagePath": "IMG_0001.jpg",
            "imageWidth": 10,
            "imageHeight": 20,
            "shapes": [
                {
                    "label": "gripper",
                    "shape_type": "mask",
                    "points": [[2, 4], [4, 7]],
                    "mask": _crop_png_b64(crop),
                }
            ],
        }
        (raw / "IMG_0001.json").write_text(json.dumps(payload))
        frames = ingest_stills(raw, Path(img_s), max_side=20)
        assert Image.open(frames[0]).size == (10, 20)
        masks = write_labelme_masks(raw, frames, Path(mask_s))
        alpha = np.asarray(Image.open(masks[0]).getchannel("A"))
        assert alpha.shape == (20, 10)
        assert alpha[4, 2] == 255
        assert alpha[7, 4] == 255


if __name__ == "__main__":
    test_labelme_json_for_stills_all_or_nothing()
    test_mask_from_labelme_pastes_inclusive_bbox()
    test_write_labelme_masks_matches_ingested_size()
    test_resize_mask_rejects_transposed_rgb()
    test_write_labelme_masks_follows_exif_display_size()
    print("ok")
