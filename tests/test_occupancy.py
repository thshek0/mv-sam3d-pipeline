"""Silhouette IoU, K scale, unproject, and a one-view synthetic score."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from occupancy import (  # noqa: E402
    apply_sim,
    load_mask_bool,
    masked_world_points,
    mesh_faces_for_occupancy,
    object_size_m,
    occupancy_pr,
    project_mesh_silhouette,
    project_occupancy,
    scale_intrinsics,
    silhouette_iou,
    sim_from_moments,
    yaw_matrix,
)
from vis import write_da3_npz  # noqa: E402


def test_silhouette_iou_and_pr() -> None:
    """Identical masks are 1; disjoint are 0; one-of-two overlap is 1/2."""
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    assert silhouette_iou(a, b) == 1.0
    assert occupancy_pr(a, b) == (1.0, 1.0)
    a[0, 0] = True
    b[1, 1] = True
    assert silhouette_iou(a, b) == 0.0
    a[1, 1] = True
    assert abs(silhouette_iou(a, b) - 0.5) < 1e-9
    p, r = occupancy_pr(a, b)
    assert abs(p - 0.5) < 1e-9
    assert abs(r - 1.0) < 1e-9


def test_scale_intrinsics_doubles_width() -> None:
    """fx and cx double when width doubles; fy/cy follow height."""
    k_mat = np.array([[10.0, 0.0, 4.0], [0.0, 8.0, 3.0], [0.0, 0.0, 1.0]])
    out = scale_intrinsics(k_mat, (6, 8), (12, 16))
    assert abs(out[0, 0] - 20.0) < 1e-9
    assert abs(out[0, 2] - 8.0) < 1e-9
    assert abs(out[1, 1] - 16.0) < 1e-9
    assert abs(out[1, 2] - 6.0) < 1e-9


def test_project_occupancy_hits_principal_point() -> None:
    """A point on the optical axis lands on (cx, cy) and IoU with that pixel is 1."""
    k_mat = np.array([[4.0, 0.0, 3.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]])
    w2c = np.eye(4)
    pred = project_occupancy(
        np.array([[0.0, 0.0, 1.0]]), k_mat, w2c, 5, 7, splat_px=0, fill_hull=False,
    )
    assert pred[2, 3]
    assert int(pred.sum()) == 1
    hull_pts = np.array([[-0.4, -0.25, 1.0], [0.4, -0.25, 1.0], [0.0, 0.4, 1.0]])
    hull = project_occupancy(hull_pts, k_mat, w2c, 5, 7, fill_hull=True)
    assert int(hull.sum()) > 3


def test_mesh_faces_for_occupancy_keeps_a_small_mesh() -> None:
    """A 12-face box is not randomly thinned."""
    import trimesh

    mesh = trimesh.creation.box(extents=(1.0, 0.5, 0.25))
    vertices, faces = mesh_faces_for_occupancy(mesh, max_faces=100)
    assert faces.shape[0] == len(mesh.faces)
    assert vertices.shape[0] == len(mesh.vertices)


def test_project_mesh_silhouette_fills_a_box() -> None:
    """A front-facing quad fills a solid disk, not scattered pixels."""
    k_mat = np.array([[8.0, 0.0, 16.0], [0.0, 8.0, 16.0], [0.0, 0.0, 1.0]])
    verts = np.array([
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    occ = project_mesh_silhouette(verts, faces, k_mat, np.eye(4), 32, 32)
    assert int(occ.sum()) > 100
    assert float(occ[8:24, 8:24].mean()) > 0.9


def test_project_mesh_silhouette_keeps_l_notch() -> None:
    """An L of two rectangles does not fill the missing corner; a 2-D hull does."""
    k_mat = np.array([[8.0, 0.0, 16.0], [0.0, 8.0, 16.0], [0.0, 0.0, 1.0]])
    verts = np.array([
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, -0.2, 1.0], [-1.0, -0.2, 1.0],
        [-1.0, -0.2, 1.0], [-0.2, -0.2, 1.0], [-0.2, 1.0, 1.0], [-1.0, 1.0, 1.0],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int32)
    occ = project_mesh_silhouette(verts, faces, k_mat, np.eye(4), 32, 32)
    hull = project_occupancy(verts, k_mat, np.eye(4), 32, 32, fill_hull=True)
    assert int(occ.sum()) > 20
    assert int(occ.sum()) < int(hull.sum())


def test_yaw_matrix_identity_and_90z() -> None:
    """0° is I; 90° about z sends +x to +y."""
    assert np.allclose(yaw_matrix(0.0, "z"), np.eye(3))
    rotated = yaw_matrix(90.0, "z") @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


def test_object_size_m_pinhole() -> None:
    """A 4 px box at Z=2 with f=4 is 2 m wide."""
    depth = np.full((6, 6), 2.0, dtype=np.float32)
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    k_mat = np.array([[4.0, 0.0, 3.0], [0.0, 4.0, 3.0], [0.0, 0.0, 1.0]])
    assert abs(object_size_m(depth, k_mat, mask) - 2.0) < 1e-6


def test_sim_from_moments_recovers_scale_and_shift() -> None:
    """Moments recover a 2x scale and +1 translation on a 3-point set."""
    src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    dst = apply_sim(src, 2.0, np.eye(3), np.array([1.0, 0.0, 0.0]))
    scale, _rot, trans = sim_from_moments(src, dst)
    assert abs(scale - 2.0) < 1e-6
    assert np.allclose(trans, [1.0, 0.0, 0.0], atol=1e-6)


def test_masked_world_points_drops_background(tmp_path: Path) -> None:
    """Only the masked Z>0 pixel is unprojected."""
    depth = np.zeros((3, 3), dtype=np.float32)
    depth[1, 1] = 2.0
    depth[0, 0] = 9.0
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    k_mat = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    pts = masked_world_points(depth, k_mat, np.eye(4), mask)
    assert pts.shape == (1, 3)
    assert np.allclose(pts[0], [0.0, 0.0, 2.0])


def test_load_mask_bool_uses_alpha(tmp_path: Path) -> None:
    """RGBA alpha, not RGB, is the occupancy."""
    rgba = Image.new("RGBA", (2, 2), (10, 20, 30, 0))
    rgba.putpixel((0, 0), (10, 20, 30, 255))
    path = tmp_path / "0.png"
    rgba.save(path)
    mask = load_mask_bool(path)
    assert bool(mask[0, 0]) and not bool(mask[0, 1])


def test_score_helpers_with_write_npz(tmp_path: Path) -> None:
    """A written 1-view npz unprojects to one world point at (0,0,1)."""
    depth = np.ones((1, 4, 4), dtype=np.float32)
    k_mat = np.array([[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    ext = np.zeros((1, 3, 4), dtype=np.float32)
    ext[0, 0, 0] = 1.0
    ext[0, 1, 1] = 1.0
    ext[0, 2, 2] = 1.0
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (4, 4), (8, 8, 8)).save(images / "0.png")
    npz = tmp_path / "da3.npz"
    write_da3_npz(npz, depth, ext, k_mat[None], [images / "0.png"], process_res=4)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    pts = masked_world_points(depth[0], k_mat, ext[0], mask)
    assert pts.shape[0] == 4
    assert np.allclose(pts[:, 2], 1.0)
