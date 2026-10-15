"""``resolve_depth`` picks an npz from --method without running GPU jobs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from helpers import dataset_depth_npz
from pipeline import resolve_depth
from vis import write_da3_npz


def _tiny_npz(path: Path, images: list[Path], depth: np.ndarray) -> Path:
    """Write a one-view DA3-style npz."""
    k_mat = np.array([[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    ext = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    return write_da3_npz(path, depth, ext[None, ...], k_mat[None, ...], images, process_res=int(depth.shape[2]))


def test_resolve_da3_uses_existing_npz() -> None:
    """``da3`` with ``--da3-npz`` copies into ``depth/da3.npz`` and does not call DA3."""
    with tempfile.TemporaryDirectory() as tmp_s:
        root = Path(tmp_s)
        images = root / "images"
        images.mkdir()
        Image.new("RGB", (8, 8), (1, 2, 3)).save(images / "0.png")
        npz = _tiny_npz(root / "a.npz", [images / "0.png"], np.ones((1, 8, 8), dtype=np.float32))
        got = resolve_depth(
            "da3",
            images_dir=images, out_dir=root, dataset="toy",
            mvsam_root=root, da3_python=None, da3_npz=npz, rs_npz=None, pose_npz=None, process_res=8,
        )
        assert got == dataset_depth_npz(root, "toy", "da3")
        assert got.is_file()


def test_resolve_hybrid_needs_both_npzs() -> None:
    """``hybrid`` without both npzs raises."""
    with tempfile.TemporaryDirectory() as tmp_s:
        root = Path(tmp_s)
        try:
            resolve_depth(
                "hybrid",
                images_dir=root, out_dir=root, dataset="toy",
                mvsam_root=root, da3_python=None, da3_npz=None, rs_npz=None, pose_npz=None, process_res=8,
            )
        except RuntimeError as exc:
            assert "rs-npz" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")
