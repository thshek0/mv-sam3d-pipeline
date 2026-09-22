"""Public convert_* helpers write the MV-SAM3D scene pieces."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helpers import (  # noqa: E402
    convert_labelme_json_to_rgba,
    convert_labelme_to_masks,
    convert_rembg_to_masks,
    convert_stills_to_images,
    convert_video_to_frames,
    env_python,
    listed_images,
)


def _crop_png_b64(arr: np.ndarray) -> str:
    """Encode a uint8 crop as a LabelMe-style base64 PNG."""
    import base64
    import io

    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_convert_stills_to_images_numbers_and_exif() -> None:
    """Public stills converter writes 0.png and applies Orientation 6."""
    with tempfile.TemporaryDirectory() as raw_s, tempfile.TemporaryDirectory() as out_s:
        raw = Path(raw_s)
        Image.new("RGB", (100, 80), (1, 2, 3)).save(raw / "a.png")
        im = Image.new("RGB", (20, 10), (10, 20, 30))
        exif = Image.Exif()
        exif[0x0112] = 6
        im.save(raw / "phone.jpg", format="JPEG", quality=95, exif=exif)
        frames = convert_stills_to_images(raw, Path(out_s), max_side=200)
        assert [path.name for path in frames] == ["0.png", "1.png"]
        assert Image.open(frames[0]).size == (100, 80)
        assert Image.open(frames[1]).size == (10, 20)


def test_convert_labelme_to_masks_matches_stems() -> None:
    """Folder LabelMe API writes RGBA next to ingested stems."""
    with tempfile.TemporaryDirectory() as raw_s:
        raw = Path(raw_s)
        rgb = Image.new("RGB", (12, 8), (10, 20, 30))
        rgb.save(raw / "shot.png")
        crop = np.ones((3, 3), dtype=np.uint8)
        payload = {
            "imagePath": "shot.png",
            "imageWidth": 12,
            "imageHeight": 8,
            "shapes": [
                {
                    "label": "object",
                    "shape_type": "mask",
                    "points": [[1, 1], [3, 3]],
                    "mask": _crop_png_b64(crop),
                }
            ],
        }
        (raw / "shot.json").write_text(json.dumps(payload))
        images_dir = raw / "images"
        mask_dir = raw / "object"
        frames = convert_stills_to_images(raw, images_dir, max_side=12)
        masks = convert_labelme_to_masks(raw, frames, mask_dir)
        assert [path.name for path in masks] == ["0.png"]
        rgba = convert_labelme_json_to_rgba(raw / "shot.json", Image.open(frames[0]))
        assert rgba.mode == "RGBA"
        assert int(np.count_nonzero(np.asarray(rgba.getchannel("A")))) > 0
        assert Image.open(masks[0]).mode == "RGBA"


def test_convert_rembg_to_masks_requires_python() -> None:
    """Empty rembg interpreter is an error, not a silent skip."""
    try:
        convert_rembg_to_masks([Path("0.png")], Path("."), "")
    except RuntimeError as exc:
        assert "REMBG_PYTHON" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_listed_images_numeric_order() -> None:
    """10.png sorts after 2.png."""
    with tempfile.TemporaryDirectory() as raw_s:
        folder = Path(raw_s)
        for name in ("10.png", "2.png", "0.png"):
            Image.new("RGB", (2, 2), (1, 1, 1)).save(folder / name)
        assert [path.name for path in listed_images(folder)] == ["0.png", "2.png", "10.png"]


def test_env_python_reads_first_set_var() -> None:
    """First non-empty name wins."""
    os.environ.pop("TEST_DA3_PYTHON", None)
    os.environ.pop("TEST_SAM3D_PYTHON", None)
    assert env_python("TEST_DA3_PYTHON", "TEST_SAM3D_PYTHON") is None
    os.environ["TEST_SAM3D_PYTHON"] = "/tmp/fake-python"
    try:
        assert env_python("TEST_DA3_PYTHON", "TEST_SAM3D_PYTHON") == "/tmp/fake-python"
    finally:
        os.environ.pop("TEST_SAM3D_PYTHON", None)


def test_convert_video_to_frames_writes_pngs() -> None:
    """ffmpeg lavfi clip becomes numbered PNGs."""
    if shutil.which("ffmpeg") is None:
        raise AssertionError("ffmpeg is required for convert_video_to_frames")
    with tempfile.TemporaryDirectory() as raw_s:
        root = Path(raw_s)
        video = root / "clip.mp4"
        images = root / "images"
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=16x16:d=1",
                "-r", "2", str(video),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        frames = convert_video_to_frames(video, images, fps=2.0, max_frames=3)
        assert frames
        assert all(path.suffix == ".png" for path in frames)
        assert frames[0].name == "0.png"
        assert Image.open(frames[0]).size == (16, 16)


if __name__ == "__main__":
    test_convert_stills_to_images_numbers_and_exif()
    test_convert_labelme_to_masks_matches_stems()
    test_convert_rembg_to_masks_requires_python()
    test_listed_images_numeric_order()
    test_env_python_reads_first_set_var()
    test_convert_video_to_frames_writes_pngs()
    print("ok converters")
