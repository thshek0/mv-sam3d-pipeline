"""Depth and mesh recipes. Compose these; do not subclass.

Depth: ``depth_da3``, ``depth_da3_posed``, ``depth_rs_da3`` (RS, then scaled
DA3 in every object hole). Mesh: ``mesh_sam3d``, ``mesh_tsdf``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from helpers import append_color_legend, listed_images, run_da3, write_pair_preview
from vis import _mask_keep, write_da3_npz, write_depthmap_preview

SOURCE_MAP_LEGEND: tuple[tuple[tuple[int, int, int], str], ...] = (
    ((0, 200, 220), "RealSense"),
    ((240, 200, 40), "DA3"),
)


def rs_valid(depth: np.ndarray) -> np.ndarray:
    """RealSense valid depth: finite and ``Z > 0``. ``Z = 0`` is invalid."""
    z = np.asarray(depth)
    return np.isfinite(z) & (z > 0)


def resize_depth(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize one depth view to ``height x width``."""
    im = Image.fromarray(np.asarray(depth, dtype=np.float32), mode="F")
    return np.array(im.resize((width, height), Image.Resampling.NEAREST), dtype=np.float32)


def combine_rs_da3(rs: np.ndarray, da3: np.ndarray, obj: np.ndarray) -> tuple[np.ndarray, float]:
    """Keep valid RS on the object; fill remaining object pixels with scaled DA3.

    ``scale`` is the median ``RS / DA3`` on pixels where both are valid.
    """
    keep = obj & rs_valid(rs)
    overlap = keep & rs_valid(da3)
    scale = 1.0
    if np.any(overlap):
        scale = float(np.median(rs[overlap] / np.maximum(da3[overlap], 1e-6)))
    out = np.zeros_like(rs)
    out[keep] = rs[keep]
    fill = obj & (~keep)
    out[fill] = da3[fill] * scale
    return out, scale


def overlay_source(
    rgb: Image.Image, obj: np.ndarray, rs: np.ndarray, fused: np.ndarray,
) -> Image.Image:
    """Cyan = valid RealSense (Z>0). Yellow = fused pixel whose RS Z was 0.

    ``Z = 0`` is invalid (RealSense). Those pixels stay dim if the fuse left them 0.
    """
    photo = rgb.convert("RGB")
    if photo.size != (rs.shape[1], rs.shape[0]):
        photo = photo.resize((rs.shape[1], rs.shape[0]), Image.Resampling.BILINEAR)
    arr = np.asarray(photo, dtype=np.float64)
    out = arr * 0.35
    keep = obj & rs_valid(rs)
    from_da3 = obj & (~keep) & rs_valid(fused)
    out[keep] = 0.25 * arr[keep] + np.array([0.0, 200.0, 220.0])
    out[from_da3] = 0.25 * arr[from_da3] + np.array([240.0, 200.0, 40.0])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def write_hybrid_source_map(
    rs_npz: Path, fused_npz: Path, images_dir: Path, mask_dir: Path, out_png: Path,
) -> Path:
    """Write ``hybrid_source_map.png``: photo | source label, plus a color legend.

    Cyan = valid RealSense (Z>0). Yellow = fused pixel whose RS Z was 0.
    """
    rs = np.load(rs_npz)
    fused = np.load(fused_npz)
    rs_z = np.asarray(rs["depth"], dtype=np.float32)
    fu_z = np.asarray(fused["depth"], dtype=np.float32)
    n_view, height, width = rs_z.shape
    if fu_z.shape[0] != n_view:
        raise ValueError(f"RS N={n_view} fused N={fu_z.shape[0]}")
    images = listed_images(images_dir)
    if len(images) != n_view:
        raise ValueError(f"{len(images)} RGBs, depth has {n_view} views")
    fu_full = np.stack([resize_depth(fu_z[i], height, width) for i in range(n_view)], axis=0)
    pairs: list[tuple[Image.Image, Image.Image, str, str]] = []
    for i, rgb_path in enumerate(images):
        obj = _mask_keep(rgb_path.stem, mask_dir, height, width)
        if obj is None:
            raise FileNotFoundError(f"missing mask for {rgb_path.name}")
        rgb = Image.open(rgb_path).convert("RGB")
        overlay = overlay_source(rgb, obj, rs_z[i], fu_full[i])
        pairs.append((rgb, overlay, f"{i} photo", f"{i} source map"))
    write_pair_preview(pairs, out_png)
    board = append_color_legend(Image.open(out_png), SOURCE_MAP_LEGEND)
    board.save(out_png)
    print(f"hybrid source map → {out_png}", flush=True)
    return out_png


def depth_da3(da3_python: str, mvsam_root: Path, images_dir: Path, da3_out: Path) -> Path:
    """Pose-free DA3 via MV-SAM3D's runner. Writes ``da3_out/da3_output.npz``."""
    return run_da3(da3_python, mvsam_root, images_dir, da3_out)


def depth_da3_posed(images_dir: Path, pose_npz: Path, output: Path, process_res: int) -> Path:
    """DA3 with known w2c + K. Implemented in ``da3_posed``."""
    from da3_posed import run_da3_posed

    return run_da3_posed(images_dir, pose_npz, output, process_res)


def depth_rs_da3(
    rs_npz: Path,
    da3_npz: Path,
    images_dir: Path,
    mask_dir: Path,
    out_npz: Path,
    *,
    preview_dir: Path | None = None,
) -> Path:
    """Keep valid RS; fill remaining object pixels with posed DA3 scaled to RS."""
    rs = np.load(rs_npz)
    da = np.load(da3_npz)
    rs_z = np.asarray(rs["depth"], dtype=np.float32)
    da_z = np.asarray(da["depth"], dtype=np.float32)
    n_view, height, width = rs_z.shape
    if da_z.shape[0] != n_view:
        raise ValueError(f"RS N={n_view} DA3 N={da_z.shape[0]}")
    images = listed_images(images_dir)
    if len(images) != n_view:
        raise ValueError(f"{len(images)} RGBs, depth has {n_view} views")
    da_full = np.stack([resize_depth(da_z[i], height, width) for i in range(n_view)], axis=0)
    combined = np.zeros_like(rs_z)
    objs: list[np.ndarray] = []
    for i, rgb_path in enumerate(images):
        obj = _mask_keep(rgb_path.stem, mask_dir, height, width)
        if obj is None:
            raise FileNotFoundError(f"missing mask for {rgb_path.name}")
        depth_i, scale = combine_rs_da3(rs_z[i], da_full[i], obj)
        combined[i] = depth_i
        objs.append(obj)
        print(f"view {i:2d} fill={int((obj & ~rs_valid(rs_z[i])).sum())} scale={scale:.3f}", flush=True)
    keep_stems = list(rs["keep_stems"]) if "keep_stems" in rs.files else None
    write_da3_npz(
        out_npz, combined, rs["extrinsics"], rs["intrinsics"], images,
        process_res=width, keep_stems=keep_stems,
    )
    if preview_dir is not None:
        write_rs_da3_previews(preview_dir, images, objs, rs_z, da_full, combined)
    return out_npz


def write_rs_da3_previews(
    out_dir: Path,
    images: list[Path],
    objs: list[np.ndarray],
    rs_z: np.ndarray,
    da_z: np.ndarray,
    depth: np.ndarray,
) -> None:
    """Write shared ``rs_`` / ``da3_posed_`` / ``hybrid_`` depth boards on ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_src: list[tuple[Image.Image, Image.Image, str, str]] = []
    for i, rgb_path in enumerate(images):
        rgb = Image.open(rgb_path).convert("RGB")
        pairs_src.append(
            (rgb, overlay_source(rgb, objs[i], rs_z[i], depth[i]), f"{i} photo", f"{i} source map"),
        )
    write_depthmap_preview(rs_z, images, out_dir / "rs_depthmap.png", label="RS")
    write_depthmap_preview(da_z, images, out_dir / "da3_posed_depthmap.png", label="da3_posed")
    write_depthmap_preview(depth, images, out_dir / "hybrid_depthmap.png", label="hybrid")
    src_path = out_dir / "hybrid_source_map.png"
    write_pair_preview(pairs_src, src_path)
    append_color_legend(Image.open(src_path), SOURCE_MAP_LEGEND).save(src_path)


def mesh_sam3d(
    scene_dir: Path,
    object_name: str,
    npz: Path,
    run_dir: Path,
    mvsam_python: str,
    mvsam_root: Path,
    *,
    merge_da3_glb: bool = False,
    seal: bool = False,
    coacd: bool = False,
) -> Path:
    """MV-SAM3D on a processed scene + npz. Writes ``run_dir/mesh.glb`` and ``mesh.png``."""
    import shutil

    from helpers import run_mvsam, visualize_depth
    from post import preview_mesh, seal_mesh, write_coacd

    images_dir = scene_dir / "images"
    visualize_depth(
        npz, run_dir / "depth_preview.png", images_dir=images_dir, mode="both",
        mask_dir=scene_dir / object_name,
    )
    if not mvsam_python:
        raise RuntimeError("Pass --mvsam-python or set MVSAM_PYTHON / SAM3D_PYTHON")
    glb = run_mvsam(mvsam_python, mvsam_root, scene_dir, object_name, npz, merge_da3_glb)
    raw = run_dir / "mesh.glb"
    shutil.copy2(glb, raw)
    preview_mesh(raw, run_dir / "mesh.png")
    if seal or coacd:
        closed = seal_mesh(raw)
        closed.export(run_dir / "sealed.stl")
        preview_mesh(run_dir / "sealed.stl", run_dir / "sealed.png")
        if coacd:
            write_coacd(closed, run_dir)
    print(f"DONE → {raw}", flush=True)
    return raw


def mesh_tsdf(
    npz: Path,
    images_dir: Path,
    mesh_path: Path,
    *,
    voxel_length: float,
    sdf_trunc: float,
    depth_trunc: float,
    mask_dir: Path | None = None,
    keep_largest: bool = False,
) -> Path:
    """TSDF from a DA3-style npz. Implemented in ``tsdf``."""
    from tsdf import tsdf_from_da3_npz

    return tsdf_from_da3_npz(
        npz, images_dir, mesh_path,
        voxel_length=voxel_length, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc,
        mask_dir=mask_dir, keep_largest=keep_largest,
    )
