"""CLI composition accepts a processed scene and raises when DA3 is missing."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import main  # noqa: E402


def _write_processed_scene(src: Path) -> None:
    """Write a one-view ``images/`` + RGBA ``object/`` scene."""
    images = src / "images"
    masks = src / "object"
    images.mkdir(parents=True)
    masks.mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(images / "0.png")
    Image.new("RGBA", (8, 8), (10, 20, 30, 255)).save(masks / "0.png")


def test_main_raises_without_da3_on_processed_scene() -> None:
    """A processed scene fails without ``--da3-npz`` / ``--da3-python``."""
    with tempfile.TemporaryDirectory() as raw_s:
        root = Path(raw_s)
        src = root / "processed"
        _write_processed_scene(src)
        try:
            main(
                [
                    "-i", str(src),
                    "--scene", "toy",
                    "--work-dir", str(root / "work"),
                    "--output-dir", str(root / "out"),
                ]
            )
        except RuntimeError as exc:
            assert "da3" in str(exc).lower()
        else:
            raise AssertionError("expected RuntimeError for missing DA3")
        staged = root / "work" / "toy"
        assert (staged / "images" / "0.png").is_file()
        assert (staged / "object" / "0.png").is_file()
        assert Image.open(staged / "object" / "0.png").mode == "RGBA"


def test_main_rejects_raw_stills() -> None:
    """A stills folder without ``images/`` is not a processed scene."""
    with tempfile.TemporaryDirectory() as raw_s:
        root = Path(raw_s)
        src = root / "raw"
        src.mkdir()
        Image.new("RGB", (8, 8), (1, 2, 3)).save(src / "a.png")
        try:
            main(
                [
                    "-i", str(src),
                    "--scene", "bare",
                    "--work-dir", str(root / "work"),
                    "--output-dir", str(root / "out"),
                ]
            )
        except RuntimeError as exc:
            assert "processed scene" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for a raw folder")


if __name__ == "__main__":
    test_main_raises_without_da3_on_processed_scene()
    test_main_rejects_raw_stills()
    print("ok pipeline")
