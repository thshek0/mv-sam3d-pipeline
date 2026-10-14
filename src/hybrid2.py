"""Hybrid depth fuse: RealSense first, DA only in uncarved gaps.

Used by ``--method hybrid``. RealSense ``Z = 0`` stays unknown until a DA
sample is accepted. Flying pixels are dropped. One positive scale maps DA onto
the cleaned overlap. A DA sample is written only when no other RealSense ray
already passed through that 3D point.
"""

from __future__ import annotations

import warnings

import numpy as np

from helpers import valid_depth
from vis import _as_w2c44

FLYING_JUMP_M = 0.01
FREE_MARGIN_M = 0.01
SCALE_THRESH_M = 0.01
RANSAC_ITERS = 64
_NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1),
)


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


def fuse_hybrid2(
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
