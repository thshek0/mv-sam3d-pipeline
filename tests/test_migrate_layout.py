"""``migrate_dataset_layout`` moves flat output trees into scene/depth/meshes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from helpers import migrate_dataset_layout


def test_migrate_dataset_layout_moves_scene_depth_meshes(tmp_path: Path) -> None:
    """Flat images/masks/da3/hybrid_sam3d become scene/, depth/, meshes/."""
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    da3 = tmp_path / "da3"
    hybrid = tmp_path / "hybrid_sam3d"
    images.mkdir()
    masks.mkdir()
    da3.mkdir()
    hybrid.mkdir()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(images / "0.png")
    Image.new("RGBA", (4, 4), (1, 2, 3, 255)).save(masks / "0.png")
    np.savez(da3 / "da3_output.npz", depth=np.ones((1, 4, 4), dtype=np.float32))
    (hybrid / "mesh.glb").write_bytes(b"glb")
    Image.new("RGB", (4, 4), (10, 10, 10)).save(hybrid / "mesh.png")
    (tmp_path / "depth_summary.png").write_bytes(b"board")
    (tmp_path / "da3_depthmap.png").write_bytes(b"map")
    (tmp_path / "rs_valid.png").write_bytes(b"valid")

    log = migrate_dataset_layout(tmp_path)
    assert any("scene/images" in line for line in log)
    assert (tmp_path / "scene" / "images" / "0.png").is_file()
    assert (tmp_path / "scene" / "masks" / "0.png").is_file()
    assert (tmp_path / "depth" / "da3.npz").is_file()
    assert (tmp_path / "depth" / "da3_depthmap.png").is_file()
    assert (tmp_path / "depth" / "rs_valid.png").is_file()
    assert (tmp_path / "meshes" / "hybrid_sam3d" / "mesh.glb").is_file()
    assert (tmp_path / "depth_summary.png").is_file()
    assert not (tmp_path / "images").exists()
    assert not (tmp_path / "da3").exists()
    assert not (tmp_path / "hybrid_sam3d").exists()
    assert not (tmp_path / "da3_depthmap.png").exists()

    assert migrate_dataset_layout(tmp_path) == []  # idempotent
    assert (tmp_path / "depth" / "da3.npz").is_file()
