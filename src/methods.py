"""Depth and mesh recipes. Compose these; do not subclass.

Depth: ``depth_da3``, ``depth_da3_posed``, ``depth_rs_da3`` (hybrid).
Mesh: ``mesh_sam3d``, ``mesh_tsdf``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from helpers import append_color_legend, listed_images, run_da3, valid_depth, write_pair_preview
from vis import write_da3_npz

SOURCE_MAP_LEGEND: tuple[tuple[tuple[int, int, int], str], ...] = (
    ((0, 200, 220), "RealSense"),
    ((240, 200, 40), "DA3"),
)


def rs_valid(depth: np.ndarray) -> np.ndarray:
    """Finite and ``Z > 0``. ``Z = 0`` is invalid."""
    return valid_depth(depth)


def resize_depth(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize one depth view to ``height x width``."""
    im = Image.fromarray(np.asarray(depth, dtype=np.float32), mode="F")
    return np.array(im.resize((width, height), Image.Resampling.NEAREST), dtype=np.float32)


def combine_rs_da3(
    rs: np.ndarray, da3: np.ndarray, region: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Keep valid RS; fill RS holes with scaled DA3 (full frame if ``region`` is None)."""
    out = np.asarray(rs, dtype=np.float32).copy()
    da = np.asarray(da3, dtype=np.float32)
    if region is None:
        region = np.ones(out.shape, dtype=bool)
    else:
        region = np.asarray(region, dtype=bool)
    keep = region & rs_valid(out)
    overlap = keep & rs_valid(da)
    scale = 1.0
    if np.any(overlap):
        scale = float(np.median(out[overlap] / np.maximum(da[overlap], 1e-6)))
    fill = region & (~rs_valid(out)) & rs_valid(da)
    out[fill] = da[fill] * scale
    return out, scale


def overlay_source(rgb: Image.Image, rs: np.ndarray, fused: np.ndarray) -> Image.Image:
    """Cyan = valid RS. Yellow = fused fill where RS was invalid."""
    photo = rgb.convert("RGB")
    if photo.size != (rs.shape[1], rs.shape[0]):
        photo = photo.resize((rs.shape[1], rs.shape[0]), Image.Resampling.BILINEAR)
    arr = np.asarray(photo, dtype=np.float64)
    out = arr * 0.35
    keep = rs_valid(rs)
    from_da3 = (~keep) & rs_valid(fused)
    out[keep] = 0.25 * arr[keep] + np.array([0.0, 200.0, 220.0])
    out[from_da3] = 0.25 * arr[from_da3] + np.array([240.0, 200.0, 40.0])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def write_hybrid_source_map(
    rs_npz: Path, fused_npz: Path, images_dir: Path, out_png: Path,
) -> Path:
    """Write ``hybrid_source_map.png``: photo | RS vs DA3 fill, plus a color legend."""
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
        rgb = Image.open(rgb_path).convert("RGB")
        overlay = overlay_source(rgb, rs_z[i], fu_full[i])
        pairs.append((rgb, overlay, f"{i} photo", f"{i} source map"))
    write_pair_preview(pairs, out_png)
    append_color_legend(Image.open(out_png), SOURCE_MAP_LEGEND).save(out_png)
    print(f"hybrid source map → {out_png}", flush=True)
    return out_png


def depth_da3(da3_python: str, mvsam_root: Path, images_dir: Path, da3_out: Path) -> Path:
    """Pose-free DA3 via MV-SAM3D's runner. Writes ``da3_out/da3_output.npz``."""
    return run_da3(da3_python, mvsam_root, images_dir, da3_out)


def depth_da3_posed(images_dir: Path, pose_npz: Path, output: Path, process_res: int) -> Path:
    """DA3 with known w2c + K. Implemented in ``da3_posed``."""
    from da3_posed import run_da3_posed

    return run_da3_posed(images_dir, pose_npz, output, process_res)


def _rs_is_unposed(rs: np.lib.npyio.NpzFile) -> bool:
    """True when the RS npz was written without AprilTag poses."""
    return "unposed_rs" in rs.files and bool(np.asarray(rs["unposed_rs"]))


def depth_rs_da3(rs_npz: Path, da3_npz: Path, images_dir: Path, out_npz: Path) -> Path:
    """Full-frame hybrid: keep valid RS; fill RS holes with scaled DA3.

    Posed RS uses free-space-aware fill (``fuse_hybrid2``); unposed RS is per-view.
    """
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
    keep_stems = list(rs["keep_stems"]) if "keep_stems" in rs.files else None

    if _rs_is_unposed(rs):
        fused = np.zeros_like(rs_z)
        scales = np.ones(n_view, dtype=np.float32)
        for i in range(n_view):
            depth_i, scale = combine_rs_da3(rs_z[i], da_full[i])
            fused[i] = depth_i
            scales[i] = np.float32(scale)
            fill_n = int((~rs_valid(rs_z[i]) & rs_valid(da_full[i])).sum())
            print(f"view {i:2d} fill={fill_n} scale={scale:.3f}", flush=True)
        extras: dict[str, np.ndarray] = {
            "hybrid_scale": scales, "unposed_hybrid": np.asarray(1, dtype=np.int8),
        }
    else:
        from hybrid2 import fuse_hybrid2

        full = np.ones((n_view, height, width), dtype=bool)
        fused, scale_f = fuse_hybrid2(rs_z, da_full, full, rs["intrinsics"], rs["extrinsics"])
        extras = {"hybrid_scale": scale_f.astype(np.float32)}

    write_da3_npz(
        out_npz, fused, rs["extrinsics"], rs["intrinsics"], images,
        process_res=width, keep_stems=keep_stems, extras=extras,
    )
    return out_npz


def mesh_sam3d(
    scene_dir: Path,
    masks_name: str,
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

    from helpers import run_mvsam
    from post import preview_mesh, seal_mesh, write_coacd

    if not mvsam_python:
        raise RuntimeError("Pass --mvsam-python or set MVSAM_PYTHON / SAM3D_PYTHON")
    glb = run_mvsam(mvsam_python, mvsam_root, scene_dir, masks_name, npz, merge_da3_glb)
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
    """TSDF from a DA3-style npz. Writes the mesh and ``mesh.png`` beside it."""
    from post import preview_mesh
    from tsdf import tsdf_from_da3_npz

    path = tsdf_from_da3_npz(
        npz, images_dir, mesh_path,
        voxel_length=voxel_length, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc,
        mask_dir=mask_dir, keep_largest=keep_largest,
    )
    preview_mesh(path, path.with_suffix(".png"))
    print(f"DONE → {path}", flush=True)
    return path
