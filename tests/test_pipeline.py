"""CLI composition accepts a processed scene and raises when DA3 is missing."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import main  # noqa: E402
from vis import depth_to_pointmap  # noqa: E402


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


def _one_view_npz(path: Path, images_dir: Path) -> None:
    """Write a one-view npz matching ``images/0.png``."""
    height, width = 8, 8
    depth = np.ones((1, height, width), dtype=np.float32) * 1.5
    k_mat = np.array([[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    pointmaps = depth_to_pointmap(depth[0], k_mat)[None, ...]
    ext = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    np.savez(
        path,
        depth=depth,
        pointmaps=pointmaps,
        pointmaps_sam3d=np.transpose(pointmaps, (0, 3, 1, 2)),
        extrinsics=ext[None, ...],
        intrinsics=k_mat[None, ...],
        image_files=np.array([str(images_dir / "0.png")]),
        process_res=height,
    )


def test_main_writes_depth_preview_before_mvsam() -> None:
    """``--da3-npz`` writes ``depth_preview.png`` then fails without MV-SAM3D."""
    with tempfile.TemporaryDirectory() as raw_s:
        root = Path(raw_s)
        src = root / "processed"
        _write_processed_scene(src)
        npz = root / "da3_output.npz"
        _one_view_npz(npz, src / "images")
        out = root / "out"
        try:
            main(
                [
                    "-i", str(src),
                    "--scene", "toy",
                    "--work-dir", str(root / "work"),
                    "--output-dir", str(out),
                    "--da3-npz", str(npz),
                ]
            )
        except RuntimeError as exc:
            assert "MVSAM" in str(exc) or "SAM3D" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for missing MV-SAM3D")
        preview = out / "toy" / "depth_preview.png"
        assert preview.is_file()
        assert preview.stat().st_size > 0


if __name__ == "__main__":
    test_main_raises_without_da3_on_processed_scene()
    test_main_rejects_raw_stills()
    test_main_writes_depth_preview_before_mvsam()
    print("ok pipeline")
