"""Still-folder ingest should write numbered PNGs."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from helpers import convert_stills_to_images


def test_convert_stills_to_images_writes_numbered_pngs() -> None:
    """Two stills become 0.png and 1.png, scaled to the long-edge cap."""
    with tempfile.TemporaryDirectory() as raw_s, tempfile.TemporaryDirectory() as out_s:
        raw = Path(raw_s)
        out = Path(out_s)
        Image.new("RGB", (400, 200), (10, 20, 30)).save(raw / "b.png")
        Image.new("RGB", (100, 80), (40, 50, 60)).save(raw / "a.png")
        frames = convert_stills_to_images(raw, out, max_side=200)
        assert [p.name for p in frames] == ["0.png", "1.png"]
        assert Image.open(out / "0.png").size == (100, 80)
        assert Image.open(out / "1.png").size == (200, 100)


def test_convert_stills_to_images_applies_exif_orientation() -> None:
    """A landscape JPEG with Orientation 6 is ingested as the upright portrait."""
    with tempfile.TemporaryDirectory() as raw_s, tempfile.TemporaryDirectory() as out_s:
        raw = Path(raw_s)
        src = raw / "phone.jpg"
        im = Image.new("RGB", (20, 10), (10, 20, 30))
        im.putpixel((19, 0), (255, 0, 0))
        exif = Image.Exif()
        exif[0x0112] = 6
        im.save(src, format="JPEG", quality=95, exif=exif)
        frames = convert_stills_to_images(raw, Path(out_s), max_side=20)
        out = Image.open(frames[0])
        assert out.size == (10, 20)


if __name__ == "__main__":
    test_convert_stills_to_images_writes_numbered_pngs()
    test_convert_stills_to_images_applies_exif_orientation()
    print("ok")
