"""Depth and mesh recipes. Compose these; do not subclass.

Depth: ``depth_da3``, ``depth_da3_posed``, ``depth_hybrid``.
Mesh: ``mesh_sam3d``, ``mesh_tsdf``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from helpers import append_color_legend, listed_images, run_da3, valid_depth, write_pair_preview
from vis import _as_w2c44, write_da3_npz

SOURCE_MAP_LEGEND: tuple[tuple[tuple[int, int, int], str], ...] = (
    ((0, 200, 220), "RealSense"),
    ((240, 200, 40), "DA3"),
)

FLYING_JUMP_M = 0.01
FREE_MARGIN_M = 0.01
SCALE_THRESH_M = 0.01
RANSAC_ITERS = 64
_NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1),
)


def rs_valid(depth: np.ndarray) -> np.ndarray:
    """Finite and ``Z > 0``. ``Z = 0`` is invalid."""
    return valid_depth(depth)


def resize_depth(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize one depth view to ``height x width``."""
    im = Image.fromarray(np.asarray(depth, dtype=np.float32), mode="F")
    return np.array(im.resize((width, height), Image.Resampling.NEAREST), dtype=np.float32)


def fill_rs_holes(
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


def drop_flying_pixels(depth: np.ndarray, jump_m: float = FLYING_JUMP_M, min_neighbors: int = 3) -> np.ndarray:
    """Zero pixels that jump from the median of their valid 3x3 neighbors.

    Isolated valid pixels (fewer than ``min_neighbors``) are dropped too.
    ``Z = 0`` stays invalid. Nothing is inpainted.
    """
    z = np.asarray(depth, dtype=np.float32)
    valid = valid_depth(z)
    padded_z = np.pad(z.astype(np.float64), 1, constant_values=np.nan)
    padded_v = np.pad(valid, 1, constant_values=False)
    height, width = z.shape
    samples: list[np.ndarray] = []
    neighbor_valid: list[np.ndarray] = []
    for dy, dx in _NEIGHBOR_OFFSETS:
        y0, x0 = dy + 1, dx + 1
        samples.append(padded_z[y0 : y0 + height, x0 : x0 + width])
        neighbor_valid.append(padded_v[y0 : y0 + height, x0 : x0 + width])
    stack = np.where(np.stack(neighbor_valid, axis=0), np.stack(samples, axis=0), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(stack, axis=0)
    count = np.stack(neighbor_valid, axis=0).sum(axis=0)
    flyer = valid & ((count < min_neighbors) | (np.abs(z - median) > jump_m))
    out = z.copy()
    out[flyer] = 0.0
    return out


def ransac_scale(
    rs: np.ndarray, da: np.ndarray, mask: np.ndarray, *,
    thresh_m: float = SCALE_THRESH_M, n_iter: int = RANSAC_ITERS, seed: int = 0,
) -> tuple[float, float]:
    """Fit ``rs ≈ a * da`` on ``mask``. Return ``(a, inlier_fraction)``.

    One-point RANSAC, then the median ratio of the winning inliers. No shift.
    Empty overlap returns ``(1, 0)``.
    """
    ys = np.asarray(rs, dtype=np.float64)[mask]
    xs = np.asarray(da, dtype=np.float64)[mask]
    ok = np.isfinite(ys) & np.isfinite(xs) & (xs > 1e-6) & (ys > 0)
    xs, ys = xs[ok], ys[ok]
    if xs.size == 0:
        return 1.0, 0.0
    if xs.size == 1:
        return float(ys[0] / xs[0]), 1.0
    rng = np.random.default_rng(seed)
    best_a = float(np.median(ys / xs))
    best_inliers = np.ones(xs.size, dtype=bool)
    best_count = -1
    for _ in range(n_iter):
        index = int(rng.integers(0, xs.size))
        scale = float(ys[index] / xs[index])
        if not np.isfinite(scale) or scale <= 0:
            continue
        inliers = np.abs(ys - scale * xs) < thresh_m
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_a = scale
            best_inliers = inliers
    if best_count > 0:
        best_a = float(np.median(ys[best_inliers] / xs[best_inliers]))
    return best_a, float(best_count / xs.size)


def _unproject(u: np.ndarray, v: np.ndarray, z: np.ndarray, k_mat: np.ndarray) -> np.ndarray:
    """Pinhole unproject pixel columns to camera-frame points ``(M, 3)``."""
    fx, fy = float(k_mat[0, 0]), float(k_mat[1, 1])
    cx, cy = float(k_mat[0, 2]), float(k_mat[1, 2])
    depth = z.astype(np.float64)
    x = (u.astype(np.float64) - cx) * depth / fx
    y = (v.astype(np.float64) - cy) * depth / fy
    return np.stack([x, y, depth], axis=1)


def _to_world(points_cam: np.ndarray, ext: np.ndarray) -> np.ndarray:
    """Map camera-frame points through the inverse of a w2c extrinsic."""
    c2w = np.linalg.inv(_as_w2c44(ext))
    homo = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float64)], axis=1)
    return (homo @ c2w.T)[:, :3]


def in_free_space(
    points_world: np.ndarray, rs: np.ndarray, k_mat: np.ndarray, ext: np.ndarray, margin_m: float,
) -> np.ndarray:
    """True where a world point is in front of a valid RealSense hit along its ray."""
    n_pts = points_world.shape[0]
    free = np.zeros(n_pts, dtype=bool)
    if n_pts == 0:
        return free
    w2c = _as_w2c44(ext)
    homo = np.concatenate([points_world, np.ones((n_pts, 1), dtype=np.float64)], axis=1)
    cam = (homo @ w2c.T)[:, :3]
    z = cam[:, 2]
    fx, fy = float(k_mat[0, 0]), float(k_mat[1, 1])
    cx, cy = float(k_mat[0, 2]), float(k_mat[1, 2])
    safe_z = np.maximum(z, 1e-6)
    u = np.rint(fx * cam[:, 0] / safe_z + cx).astype(np.int32)
    v = np.rint(fy * cam[:, 1] / safe_z + cy).astype(np.int32)
    height, width = rs.shape
    inside = (z > 0) & (u >= 0) & (v >= 0) & (u < width) & (v < height)
    if not np.any(inside):
        return free
    z_rs = rs[v[inside], u[inside]]
    z_in = z[inside]
    closer = valid_depth(z_rs) & (z_in < (z_rs - margin_m))
    free[np.flatnonzero(inside)[closer]] = True
    return free


def fuse_hybrid(
    rs_z: np.ndarray, da_z: np.ndarray, obj: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray, *,
    jump_m: float = FLYING_JUMP_M, free_margin_m: float = FREE_MARGIN_M, scale_thresh_m: float = SCALE_THRESH_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse cleaned RS with scaled DA. Return ``(depth, scales)``.

    DA fills an object pixel only when that 3D point is not in front of a
    RealSense measurement in any other view. Rejected pixels stay 0.
    """
    rs = np.asarray(rs_z, dtype=np.float32)
    da = np.asarray(da_z, dtype=np.float32)
    keep_obj = np.asarray(obj, dtype=bool)
    if rs.shape != da.shape or rs.shape != keep_obj.shape:
        raise ValueError(f"shape rs {rs.shape} da {da.shape} obj {keep_obj.shape}")
    n_view = rs.shape[0]
    if np.asarray(intrinsics).shape[0] != n_view or np.asarray(extrinsics).shape[0] != n_view:
        raise ValueError("intrinsics and extrinsics must have one matrix per view")
    cleaned = np.stack([drop_flying_pixels(rs[i], jump_m=jump_m) for i in range(n_view)], axis=0)
    fused = np.zeros_like(cleaned)
    scales = np.ones(n_view, dtype=np.float64)
    for i in range(n_view):
        overlap = keep_obj[i] & valid_depth(cleaned[i]) & valid_depth(da[i])
        scale, inlier_frac = ransac_scale(cleaned[i], da[i], overlap, thresh_m=scale_thresh_m)
        scales[i] = scale
        aligned = da[i] * np.float32(scale)
        keep = keep_obj[i] & valid_depth(cleaned[i])
        fused[i][keep] = cleaned[i][keep]
        candidate = keep_obj[i] & (~keep) & valid_depth(aligned)
        count = int(candidate.sum())
        rejected = 0
        if count:
            vv, uu = np.nonzero(candidate)
            pts = _to_world(_unproject(uu, vv, aligned[vv, uu], intrinsics[i]), extrinsics[i])
            free = np.zeros(count, dtype=bool)
            for j in range(n_view):
                if j == i:
                    continue
                free |= in_free_space(pts, cleaned[j], intrinsics[j], extrinsics[j], free_margin_m)
            rejected = int(free.sum())
            accept = ~free
            fused[i][vv[accept], uu[accept]] = aligned[vv[accept], uu[accept]]
        flyers = int((keep_obj[i] & valid_depth(rs[i]) & ~valid_depth(cleaned[i])).sum())
        print(
            f"view {i:2d} scale={scale:.3f} inliers={inlier_frac:.2f} "
            f"flyers={flyers} fill={count - rejected} rejected={rejected}",
            flush=True,
        )
    return fused, scales


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


def depth_npz_is_unposed(npz: Path) -> bool:
    """True when the depth npz has identity cameras (no AprilTag / shared poses).

    Multi-view TSDF on such an npz is meaningless: every view shares ``w2c = I``.
    """
    raw = np.load(npz)
    for key in ("unposed_rs", "unposed_hybrid"):
        if key in raw.files and bool(np.asarray(raw[key])):
            return True
    return False


def depth_hybrid(rs_npz: Path, da3_npz: Path, images_dir: Path, out_npz: Path) -> Path:
    """Full-frame hybrid: keep valid RS; fill RS holes with scaled DA3.

    Posed RS uses free-space-aware fill (``fuse_hybrid``); unposed RS is per-view.
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
            depth_i, scale = fill_rs_holes(rs_z[i], da_full[i])
            fused[i] = depth_i
            scales[i] = np.float32(scale)
            fill_n = int((~rs_valid(rs_z[i]) & rs_valid(da_full[i])).sum())
            print(f"view {i:2d} fill={fill_n} scale={scale:.3f}", flush=True)
        extras: dict[str, np.ndarray] = {
            "hybrid_scale": scales, "unposed_hybrid": np.asarray(1, dtype=np.int8),
        }
    else:
        full = np.ones((n_view, height, width), dtype=bool)
        fused, scale_f = fuse_hybrid(rs_z, da_full, full, rs["intrinsics"], rs["extrinsics"])
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
    mesh_glb: Path,
    mvsam_python: str,
    mvsam_root: Path,
    *,
    merge_da3_glb: bool = False,
    seal: bool = False,
    coacd: bool = False,
) -> Path:
    """MV-SAM3D on a processed scene + npz. Writes ``{slug}_mesh.glb`` / ``.png``."""
    import shutil

    from helpers import mesh_slug_from_glb, run_mvsam
    from post import preview_mesh, seal_mesh, write_coacd

    if not mvsam_python:
        raise RuntimeError("Pass --mvsam-python or set MVSAM_PYTHON / SAM3D_PYTHON")
    mesh_glb = mesh_glb.expanduser()
    mesh_glb.parent.mkdir(parents=True, exist_ok=True)
    glb = run_mvsam(mvsam_python, mvsam_root, scene_dir, masks_name, npz, merge_da3_glb)
    shutil.copy2(glb, mesh_glb)
    preview_mesh(mesh_glb, mesh_glb.with_suffix(".png"))
    if seal or coacd:
        slug = mesh_slug_from_glb(mesh_glb)
        prefix = f"{slug}_" if slug else ""
        sealed = mesh_glb.parent / f"{prefix}sealed.stl"
        closed = seal_mesh(mesh_glb)
        closed.export(sealed)
        preview_mesh(sealed, sealed.with_suffix(".png"))
        if coacd:
            write_coacd(closed, mesh_glb.parent)
    print(f"DONE → {mesh_glb}", flush=True)
    return mesh_glb


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
