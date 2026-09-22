"""``validate_scene`` requires matching RGBA masks with some alpha."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helpers import validate_scene  # noqa: E402


def test_validate_scene_accepts_matching_rgba() -> None:
    """Equal stems and non-empty alpha pass."""
    with tempfile.TemporaryDirectory() as raw_s:
        scene = Path(raw_s)
        images = scene / "images"
        masks = scene / "object"
        images.mkdir()
        masks.mkdir()
        Image.new("RGB", (8, 8), (10, 20, 30)).save(images / "0.png")
        rgba = Image.new("RGBA", (8, 8), (1, 2, 3, 0))
        rgba.putpixel((2, 2), (255, 0, 0, 255))
        rgba.save(masks / "0.png")
        assert validate_scene(scene, "object") == [images / "0.png"]


def test_validate_scene_rejects_empty_alpha() -> None:
    """A fully transparent mask is invalid."""
    with tempfile.TemporaryDirectory() as raw_s:
        scene = Path(raw_s)
        images = scene / "images"
        masks = scene / "object"
        images.mkdir()
        masks.mkdir()
        Image.new("RGB", (4, 4), (1, 1, 1)).save(images / "0.png")
        Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(masks / "0.png")
        try:
            validate_scene(scene, "object")
        except ValueError as exc:
            assert "empty alpha" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_validate_scene_missing_mask() -> None:
    """A missing sidecar mask is FileNotFoundError."""
    with tempfile.TemporaryDirectory() as raw_s:
        scene = Path(raw_s)
        (scene / "images").mkdir()
        (scene / "object").mkdir()
        Image.new("RGB", (4, 4), (1, 1, 1)).save(scene / "images" / "0.png")
        try:
            validate_scene(scene, "object")
        except FileNotFoundError as exc:
            assert "0.png" in str(exc)
        else:
            raise AssertionError("expected FileNotFoundError")


def test_validate_scene_rejects_rgb_mask() -> None:
    """Mask must be RGBA."""
    with tempfile.TemporaryDirectory() as raw_s:
        scene = Path(raw_s)
        (scene / "images").mkdir()
        (scene / "object").mkdir()
        Image.new("RGB", (4, 4), (1, 1, 1)).save(scene / "images" / "0.png")
        Image.new("RGB", (4, 4), (2, 2, 2)).save(scene / "object" / "0.png")
        try:
            validate_scene(scene, "object")
        except ValueError as exc:
            assert "RGBA" in str(exc)
        else:
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_validate_scene_accepts_matching_rgba()
    test_validate_scene_rejects_empty_alpha()
    test_validate_scene_missing_mask()
    test_validate_scene_rejects_rgb_mask()
    print("ok")
