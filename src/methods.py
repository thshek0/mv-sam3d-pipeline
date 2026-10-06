"""Depth and mesh recipes. Compose these; do not subclass.

Depth: ``depth_da3``, ``depth_da3_posed``, ``depth_rs_da3`` (``fill_wells=`` for
the old fill-all-holes map). Mesh: ``mesh_sam3d``, ``mesh_tsdf``.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from helpers import listed_images, run_da3, write_pair_preview
from vis import _colorize_depth, _mask_keep, overlay_depth_visibility, write_da3_npz


def label_cc(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected labels for a boolean mask. Returns ``(labels, n_components)``."""
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    n_comp = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            n_comp += 1
            queue: deque[tuple[int, int]] = deque([(y, x)])
            labels[y, x] = n_comp
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = n_comp
                        queue.append((ny, nx))
    return labels, n_comp


def classify_miss(obj: np.ndarray, rs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split object-mask ``Z=0`` into well (enclosed) vs wall (touches mask border)."""
    miss = obj & (~np.isfinite(rs) | (rs <= 0))
    labels, n_comp = label_cc(miss)
    well = np.zeros_like(obj)
    wall = np.zeros_like(obj)
    height, width = obj.shape
    for lab in range(1, n_comp + 1):
        blob = labels == lab
        ys, xs = np.nonzero(blob)
        touches_border = False
        for y, x in zip(ys, xs):
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width or not obj[ny, nx]:
                    touches_border = True
                    break
            if touches_border:
                break
        if touches_border:
            wall |= blob
        else:
            well |= blob
    return well, wall


def resize_depth(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize one depth view to ``height x width``."""
    im = Image.fromarray(np.asarray(depth, dtype=np.float32), mode="F")
    return np.array(im.resize((width, height), Image.Resampling.NEAREST), dtype=np.float32)


def combine_rs_da3(
    rs: np.ndarray, da3: np.ndarray, obj: np.ndarray, *, fill_wells: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """One view: keep RS, optionally fill wells, copy DA3 onto walls.

    Returns ``(depth, well, wall, scale)``. ``scale`` is median RS/DA3 on overlap.
    """
    well, wall = classify_miss(obj, rs)
    if fill_wells:
        wall = wall | well
        well = np.zeros_like(obj)
    keep = obj & np.isfinite(rs) & (rs > 0)
    overlap = keep & np.isfinite(da3) & (da3 > 0)
    scale = 1.0
    if np.any(overlap):
        scale = float(np.median(rs[overlap] / np.maximum(da3[overlap], 1e-6)))
    out = np.zeros_like(rs)
    out[keep] = rs[keep]
    out[wall] = da3[wall] * scale
    return out, well, wall, scale


def overlay_source(
    rgb: Image.Image, obj: np.ndarray, rs: np.ndarray, well: np.ndarray, wall: np.ndarray,
) -> Image.Image:
    """Cyan = keep RS, red = leave well empty, yellow = copy DA3."""
    photo = rgb.convert("RGB")
    if photo.size != (rs.shape[1], rs.shape[0]):
        photo = photo.resize((rs.shape[1], rs.shape[0]), Image.Resampling.BILINEAR)
    arr = np.asarray(photo, dtype=np.float64)
    out = arr * 0.35
    keep = obj & np.isfinite(rs) & (rs > 0)
    out[keep] = 0.25 * arr[keep] + np.array([0.0, 200.0, 220.0])
    out[well] = 0.25 * arr[well] + np.array([220.0, 40.0, 40.0])
    out[wall] = 0.25 * arr[wall] + np.array([240.0, 200.0, 40.0])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


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
    fill_wells: bool = False,
    preview_dir: Path | None = None,
) -> Path:
    """Align posed DA3 onto the RS grid and combine. Writes a DA3-style npz.

    ``fill_wells=False`` leaves enclosed holes at 0. ``True`` is fill-all-holes.
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
    combined = np.zeros_like(rs_z)
    wells: list[np.ndarray] = []
    walls: list[np.ndarray] = []
    objs: list[np.ndarray] = []
    for i, rgb_path in enumerate(images):
        obj = _mask_keep(rgb_path.stem, mask_dir, height, width)
        if obj is None:
            raise FileNotFoundError(f"missing mask for {rgb_path.name}")
        depth_i, well, wall, scale = combine_rs_da3(rs_z[i], da_full[i], obj, fill_wells=fill_wells)
        combined[i] = depth_i
        wells.append(well)
        walls.append(wall)
        objs.append(obj)
        print(
            f"view {i:2d} well={int(well.sum())} wall={int(wall.sum())} scale={scale:.3f} "
            f"fill_wells={fill_wells}",
            flush=True,
        )
    keep_stems = list(rs["keep_stems"]) if "keep_stems" in rs.files else None
    write_da3_npz(
        out_npz, combined, rs["extrinsics"], rs["intrinsics"], images,
        process_res=width, keep_stems=keep_stems,
    )
    if preview_dir is not None:
        write_rs_da3_previews(preview_dir, images, objs, rs_z, da_full, combined, wells, walls)
    return out_npz


def write_rs_da3_previews(
    out_dir: Path,
    images: list[Path],
    objs: list[np.ndarray],
    rs_z: np.ndarray,
    da_z: np.ndarray,
    depth: np.ndarray,
    wells: list[np.ndarray],
    walls: list[np.ndarray],
) -> None:
    """Write ``rs.png``, ``da3.png``, ``source.png``, ``depth.png``, ``visibility.png``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_rs: list[tuple[Image.Image, Image.Image, str, str]] = []
    pairs_da: list[tuple[Image.Image, Image.Image, str, str]] = []
    pairs_src: list[tuple[Image.Image, Image.Image, str, str]] = []
    pairs_z: list[tuple[Image.Image, Image.Image, str, str]] = []
    pairs_vis: list[tuple[Image.Image, Image.Image, str, str]] = []
    for i, rgb_path in enumerate(images):
        rgb = Image.open(rgb_path).convert("RGB")
        pairs_rs.append((rgb, Image.fromarray(_colorize_depth(rs_z[i])), f"{i} photo", f"{i} RS"))
        pairs_da.append((rgb, Image.fromarray(_colorize_depth(da_z[i])), f"{i} photo", f"{i} DA3"))
        pairs_src.append(
            (rgb, overlay_source(rgb, objs[i], rs_z[i], wells[i], walls[i]), f"{i} photo", f"{i} source"),
        )
        pairs_z.append((rgb, Image.fromarray(_colorize_depth(depth[i])), f"{i} photo", f"{i} depth"))
        pairs_vis.append((rgb, overlay_depth_visibility(rgb, depth[i], objs[i]), f"{i} photo", f"{i} vis"))
    write_pair_preview(pairs_rs, out_dir / "rs.png")
    write_pair_preview(pairs_da, out_dir / "da3.png")
    write_pair_preview(pairs_src, out_dir / "source.png")
    write_pair_preview(pairs_z, out_dir / "depth.png")
    write_pair_preview(pairs_vis, out_dir / "visibility.png")


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
