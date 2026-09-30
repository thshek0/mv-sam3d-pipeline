"""TSDF fusion skips Z<=0 and zeros depth outside the object mask."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsdf import apply_mask_to_depth, keep_largest_triangle_component, tsdf_from_da3_npz  # noqa: E402
from vis import write_da3_npz  # noqa: E402


def test_apply_mask_to_depth_zeros_background() -> None:
    """Alpha 0 (or L=0) sets depth to 0; foreground Z is unchanged."""
    depth = np.array([[0.4, 0.5], [0.6, 0.7]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp_s:
        mask_path = Path(tmp_s) / "0.png"
        Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8), mode="L").save(mask_path)
        out = apply_mask_to_depth(depth, mask_path)
    assert float(out[0, 0]) == 0.0
    assert abs(float(out[0, 1]) - 0.5) < 1e-6
    assert abs(float(out[1, 0]) - 0.6) < 1e-6
    assert float(out[1, 1]) == 0.0


def test_tsdf_from_da3_npz_extracts_mesh_and_leaves_zero_z() -> None:
    """A planar depth with a center hole integrates to a non-empty triangle mesh."""
    height, width = 48, 48
    depth = np.full((1, height, width), 0.40, dtype=np.float32)
    depth[0, 18:30, 18:30] = 0.0
    k_mat = np.array([[80.0, 0.0, 24.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    w2c = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp_s:
        root = Path(tmp_s)
        images = root / "images"
        images.mkdir()
        Image.new("RGB", (width, height), (200, 200, 200)).save(images / "0.png")
        npz_path = write_da3_npz(
            root / "da3.npz", depth, w2c[None, ...], k_mat[None, ...], [images / "0.png"], process_res=width,
        )
        mesh_path = tsdf_from_da3_npz(
            npz_path, images, root / "mesh.ply",
            voxel_length=0.01, sdf_trunc=0.04, depth_trunc=1.0,
        )
        assert mesh_path.is_file()
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        assert np.asarray(mesh.vertices).shape[0] > 0
        assert np.asarray(mesh.triangles).shape[0] > 0


def test_keep_largest_triangle_component_drops_the_small_shell() -> None:
    """Two disconnected triangles: the cluster with more faces is kept."""
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0],
            [10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0],
        ], dtype=np.float64),
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int32),
    )
    out = keep_largest_triangle_component(mesh)
    faces = np.asarray(out.triangles)
    verts = np.asarray(out.vertices)
    assert faces.shape[0] == 2
    assert verts.shape[0] == 4
    assert float(verts[:, 0].max()) < 2.0
