"""TSDF fusion from a DA3-style npz (known K, w2c, metric depth)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from helpers import listed_images
from vis import _as_w2c44, load_da3_npz, write_depth_visibility_preview


def _require_open3d() -> Any:
    """Import Open3D (already in the driver venv)."""
    import open3d as o3d

    return o3d


def apply_mask_to_depth(depth: np.ndarray, mask_path: Path | None) -> np.ndarray:
    """Zero depth outside the mask alpha (or L > 0). ``Z = 0`` stays invalid."""
    if mask_path is None or not mask_path.is_file():
        return depth
    mask_img = Image.open(mask_path)
    if mask_img.size != (depth.shape[1], depth.shape[0]):
        raise ValueError(f"{mask_path} size {mask_img.size} != depth {depth.shape[1]}x{depth.shape[0]}")
    if mask_img.mode in {"RGBA", "LA"}:
        alpha = np.asarray(mask_img.getchannel("A"))
    else:
        alpha = np.asarray(mask_img.convert("L"))
    out = depth.copy()
    out[alpha == 0] = 0.0
    return out


def keep_largest_triangle_component(mesh: Any) -> Any:
    """Keep the connected triangle cluster with the most faces. Does not fill holes."""
    triangle_clusters, cluster_n_triangles, _cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if cluster_n_triangles.size == 0:
        return mesh
    largest = int(np.argmax(cluster_n_triangles))
    mesh.remove_triangles_by_mask(triangle_clusters != largest)
    mesh.remove_unreferenced_vertices()
    return mesh


def tsdf_from_da3_npz(
    npz_path: Path,
    images_dir: Path,
    mesh_path: Path,
    *,
    voxel_length: float,
    sdf_trunc: float,
    depth_trunc: float,
    mask_dir: Path | None = None,
    keep_largest: bool = False,
) -> Path:
    """Fuse metric depth into a TSDF and write a triangle mesh.

    Extrinsics are world-to-camera (OpenCV). Depth is meters; ``Z <= 0`` is skipped.
    Optional RGBA/L masks zero background depth (table). Well pixels with no Z
    do not carve or fill — they simply never update the volume.
    ``keep_largest`` drops flyer components; it does not fill holes.
    """
    if voxel_length <= 0 or sdf_trunc <= 0 or depth_trunc <= 0:
        raise ValueError("voxel_length, sdf_trunc, and depth_trunc must be positive")
    o3d = _require_open3d()
    data = load_da3_npz(npz_path)
    depth = np.asarray(data["depth"], dtype=np.float32)
    ks = np.asarray(data["intrinsics"])
    exts = np.asarray(data["extrinsics"])
    n_view, height, width = depth.shape
    images = listed_images(images_dir)
    if len(images) != n_view:
        raise ValueError(f"{images_dir} has {len(images)} RGBs, npz has {n_view} views")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for i, rgb_path in enumerate(images):
        z = depth[i]
        if mask_dir is not None:
            z = apply_mask_to_depth(z, mask_dir / rgb_path.name)
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        if rgb.shape[0] != height or rgb.shape[1] != width:
            raise ValueError(f"{rgb_path} {rgb.shape[:2]} != depth {height}x{width}")
        color = o3d.geometry.Image(rgb)
        depth_im = o3d.geometry.Image(z)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth_im, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
        )
        fx, fy = float(ks[i, 0, 0]), float(ks[i, 1, 1])
        cx, cy = float(ks[i, 0, 2]), float(ks[i, 1, 2])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        volume.integrate(rgbd, intrinsic, _as_w2c44(exts[i]))
        print(f"tsdf view {i} valid={100.0 * (z > 0).mean():.1f}%", flush=True)
    if mask_dir is not None:
        write_depth_visibility_preview(depth, images, mask_dir, mesh_path.parent / "depth_visibility_preview.png")
    mesh = volume.extract_triangle_mesh()
    if keep_largest:
        mesh = keep_largest_triangle_component(mesh)
    mesh.compute_vertex_normals()
    n_vert = np.asarray(mesh.vertices).shape[0]
    n_face = np.asarray(mesh.triangles).shape[0]
    if n_face == 0:
        raise RuntimeError("TSDF extracted an empty mesh")
    mesh_path = mesh_path.expanduser()
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh):
        raise RuntimeError(f"failed to write {mesh_path}")
    print(f"tsdf mesh → {mesh_path} verts={n_vert} faces={n_face}", flush=True)
    return mesh_path


def parse_args() -> argparse.Namespace:
    """CLI for ``tsdf_from_da3_npz``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output .ply or .glb")
    parser.add_argument("--masks", type=Path, default=None)
    parser.add_argument("--voxel-length", required=True, type=float)
    parser.add_argument("--sdf-trunc", required=True, type=float)
    parser.add_argument("--depth-trunc", required=True, type=float)
    parser.add_argument("--keep-largest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tsdf_from_da3_npz(
        args.npz, args.images, args.output,
        voxel_length=args.voxel_length, sdf_trunc=args.sdf_trunc, depth_trunc=args.depth_trunc,
        mask_dir=args.masks, keep_largest=args.keep_largest,
    )
