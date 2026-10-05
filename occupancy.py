"""Per-view silhouette IoU: project a GLB onto each mask. CPU, no mesh render.

Aligns a quadric-decimated mesh to view-0 unprojected depth (scale + centroid +
90° yaw). Rasterizes triangles so concavities stay (not a 2-D hull). Color is
ignored. ``mesh.png`` is a canonical 3-view still, not this camera projection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import trimesh
from PIL import Image

from helpers import listed_images, write_pair_preview
from post import _decimate_for_preview, load_triangle_mesh
from vis import _as_w2c44, camera_to_world, depth_to_pointmap, load_da3_npz

N_SAMPLES = 20000
SPLAT_PX = 2
MAX_OCC_FACES = 20000
LOOSE_MASK = 0.90
CAPTURES = (
    "captures_20261002_1547",
    "captures_20261002_1556",
    "captures_20261002_1558",
    "captures_20261002_1601",
    "captures_20261002_1606",
)


def load_mask_bool(path: Path) -> np.ndarray:
    """Return a boolean HW mask from RGBA alpha or an L still."""
    image = Image.open(path)
    if image.mode in {"RGBA", "LA"}:
        alpha = np.asarray(image.getchannel("A"))
    else:
        alpha = np.asarray(image.convert("L"))
    return alpha > 0


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize a boolean mask to ``height x width``."""
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * np.uint8(255), mode="L")
    out = np.asarray(image.resize((width, height), Image.Resampling.NEAREST))
    return out > 0


def scale_intrinsics(k_mat: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    """Scale a pinhole ``K`` from ``src_hw`` (H, W) to ``dst_hw``."""
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    out = np.asarray(k_mat, dtype=np.float64).copy()
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def silhouette_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU of two boolean occupancy maps. Empty∩empty is 1."""
    pred_b = np.asarray(pred, dtype=bool)
    gt_b = np.asarray(gt, dtype=bool)
    inter = int(np.count_nonzero(pred_b & gt_b))
    union = int(np.count_nonzero(pred_b | gt_b))
    if union == 0:
        return 1.0
    return inter / float(union)


def occupancy_pr(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Precision and recall of ``pred`` vs ``gt``. Empty pred/gt → 1 if both empty else 0."""
    pred_b = np.asarray(pred, dtype=bool)
    gt_b = np.asarray(gt, dtype=bool)
    inter = int(np.count_nonzero(pred_b & gt_b))
    n_pred = int(np.count_nonzero(pred_b))
    n_gt = int(np.count_nonzero(gt_b))
    if n_pred == 0 and n_gt == 0:
        return 1.0, 1.0
    precision = inter / float(n_pred) if n_pred else 0.0
    recall = inter / float(n_gt) if n_gt else 0.0
    return precision, recall


def sample_mesh_points(mesh: trimesh.Trimesh, n_samples: int) -> np.ndarray:
    """Uniform surface samples, or vertices if the mesh is tiny."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise ValueError("mesh has no faces/vertices")
    if len(mesh.faces) * 3 < n_samples:
        return np.asarray(mesh.vertices, dtype=np.float64)
    return np.asarray(mesh.sample(n_samples), dtype=np.float64)


def clip_cloud(points: np.ndarray, lo_q: float = 5.0, hi_q: float = 95.0) -> np.ndarray:
    """Drop depth flyers by keeping the ``lo_q``–``hi_q`` percentile of |p|."""
    if points.shape[0] < 8:
        return points
    radii = np.linalg.norm(points - points.mean(axis=0), axis=1)
    lo, hi = np.percentile(radii, (lo_q, hi_q))
    keep = (radii >= lo) & (radii <= hi)
    return points[keep] if np.any(keep) else points


def masked_world_points(depth: np.ndarray, k_mat: np.ndarray, w2c: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Unproject ``depth`` pixels that are in ``mask`` and ``Z>0`` into world XYZ."""
    if depth.shape != mask.shape:
        raise ValueError(f"depth {depth.shape} != mask {mask.shape}")
    pointmap = depth_to_pointmap(np.asarray(depth, dtype=np.float64), np.asarray(k_mat, dtype=np.float64))
    valid = mask & np.isfinite(pointmap).all(axis=2) & (np.asarray(depth) > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    return camera_to_world(pointmap[valid], w2c)


def sim_from_moments(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Uniform scale + translation so ``src`` matches ``dst`` centroid and RMS radius."""
    if src.shape[0] < 3 or dst.shape[0] < 3:
        raise ValueError("need at least 3 points on each side to align")
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    rms_src = float(np.sqrt(np.mean(np.sum((src - c_src) ** 2, axis=1))))
    rms_dst = float(np.sqrt(np.mean(np.sum((dst - c_dst) ** 2, axis=1))))
    scale = rms_dst / max(rms_src, 1e-9)
    rot = np.eye(3, dtype=np.float64)
    trans = c_dst - scale * c_src
    return scale, rot, trans


def apply_sim(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Return ``scale * points @ R.T + t``."""
    return scale * (np.asarray(points, dtype=np.float64) @ np.asarray(rot, dtype=np.float64).T) + np.asarray(trans)


def object_size_m(depth: np.ndarray, k_mat: np.ndarray, mask: np.ndarray) -> float:
    """Approx. object size (m): mask bbox in pixels times median Z over f."""
    valid = mask & np.isfinite(depth) & (np.asarray(depth) > 0)
    if not np.any(valid):
        raise RuntimeError("no valid depth in mask")
    ys, xs = np.nonzero(mask)
    side = float(max(int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1))
    z_med = float(np.median(np.asarray(depth, dtype=np.float64)[valid]))
    f_len = 0.5 * (float(k_mat[0, 0]) + float(k_mat[1, 1]))
    return side * z_med / max(f_len, 1e-9)


def mesh_extent(points: np.ndarray) -> float:
    """Longest axis-aligned side of a point set."""
    return float(np.max(points.max(axis=0) - points.min(axis=0)))


def yaw_matrix(degrees: float, axis: str) -> np.ndarray:
    """Right-handed rotation by ``degrees`` about ``axis`` ``x``/``y``/``z``."""
    rad = np.deg2rad(float(degrees))
    cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]])
    if axis == "y":
        return np.array([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]])
    if axis == "z":
        return np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"axis must be x, y, or z, got {axis!r}")


def mesh_faces_for_occupancy(mesh: trimesh.Trimesh, max_faces: int = MAX_OCC_FACES) -> tuple[np.ndarray, np.ndarray]:
    """Return a filled surface: Open3D quadric, same path as ``mesh.png``.

    Random face subsample is not used. That left 2% of a 1M-face GLB and a
    holey silhouette (high precision, recall ~0.2) that did not match the preview.
    """
    simple = _decimate_for_preview(mesh, max_faces)
    return np.asarray(simple.vertices, dtype=np.float64), np.asarray(simple.faces, dtype=np.int32)


def project_mesh_silhouette(
    vertices: np.ndarray, faces: np.ndarray, k_mat: np.ndarray, w2c: np.ndarray, height: int, width: int,
) -> np.ndarray:
    """Fill projected triangles. Keeps concavities (the L-notch); not a 2-D convex hull."""
    occ = np.zeros((height, width), dtype=np.uint8)
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return occ.astype(bool)
    w2c44 = _as_w2c44(w2c)
    ones = np.ones((vertices.shape[0], 1), dtype=np.float64)
    cam = np.concatenate([np.asarray(vertices, dtype=np.float64), ones], axis=1) @ w2c44.T
    zz = cam[:, 2]
    uu = k_mat[0, 0] * cam[:, 0] / np.maximum(zz, 1e-6) + k_mat[0, 2]
    vv = k_mat[1, 1] * cam[:, 1] / np.maximum(zz, 1e-6) + k_mat[1, 2]
    z_ok = zz[faces].min(axis=1) > 1e-6
    tris = np.stack([uu[faces], vv[faces]], axis=-1)
    for tri, ok in zip(tris, z_ok):
        if not ok:
            continue
        pts = np.ascontiguousarray(np.rint(tri).astype(np.int32).reshape(-1, 1, 2))
        cv2.fillConvexPoly(occ, pts, 1)
    if int(occ.sum()) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, kernel)
    return occ.astype(bool)


def align_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    cloud: np.ndarray,
    scale: float,
    k_mat: np.ndarray,
    w2c: np.ndarray,
    gt: np.ndarray,
    height: int,
    width: int,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Centroid + ``scale``, then pick the yaw (x/y/z × 90°) with best view-0 IoU.

    Scores the triangle silhouette, not a 2-D hull. ICP is not used.
    """
    if vertices.shape[0] < 3 or cloud.shape[0] < 3:
        raise ValueError("need at least 3 points on each side to align")
    c_src = vertices.mean(axis=0)
    c_dst = cloud.mean(axis=0)
    best: tuple[float, np.ndarray, np.ndarray, float] | None = None
    for axis in ("x", "y", "z"):
        for degrees in (0.0, 90.0, 180.0, 270.0):
            rot = yaw_matrix(degrees, axis)
            trans = c_dst - scale * (c_src @ rot.T)
            pred = project_mesh_silhouette(apply_sim(vertices, scale, rot, trans), faces, k_mat, w2c, height, width)
            iou = silhouette_iou(pred, gt)
            if best is None or iou > best[3]:
                best = (scale, rot, trans, iou)
    if best is None:
        raise RuntimeError("yaw search produced no candidate")
    return best


def project_uv(points: np.ndarray, k_mat: np.ndarray, w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points. Returns ``(u, v, keep)`` with ``keep`` = Z>0."""
    w2c44 = _as_w2c44(w2c)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    cam = np.concatenate([np.asarray(points, dtype=np.float64), ones], axis=1) @ w2c44.T
    zz = cam[:, 2]
    keep = zz > 1e-6
    uu = np.full(points.shape[0], np.nan, dtype=np.float64)
    vv = np.full(points.shape[0], np.nan, dtype=np.float64)
    uu[keep] = k_mat[0, 0] * cam[keep, 0] / zz[keep] + k_mat[0, 2]
    vv[keep] = k_mat[1, 1] * cam[keep, 1] / zz[keep] + k_mat[1, 2]
    return uu, vv, keep


def project_occupancy(
    points: np.ndarray,
    k_mat: np.ndarray,
    w2c: np.ndarray,
    height: int,
    width: int,
    splat_px: int = SPLAT_PX,
    fill_hull: bool = True,
) -> np.ndarray:
    """Boolean HW occupancy from projected points. No triangle raster.

    ``fill_hull`` fills the 2-D convex hull (silhouette of the samples).
    Otherwise mark hits and dilate by ``splat_px``.
    """
    occ = np.zeros((height, width), dtype=np.uint8)
    if points.shape[0] == 0:
        return occ.astype(bool)
    uu, vv, keep = project_uv(points, k_mat, w2c)
    ui = np.rint(uu[keep]).astype(np.int32)
    vi = np.rint(vv[keep]).astype(np.int32)
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if not np.any(inside):
        return occ.astype(bool)
    pts = np.stack([ui[inside], vi[inside]], axis=1)
    if fill_hull and pts.shape[0] >= 3:
        cv2.fillConvexPoly(occ, cv2.convexHull(pts), 1)
        return occ.astype(bool)
    occ[pts[:, 1], pts[:, 0]] = 1
    if splat_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * splat_px + 1, 2 * splat_px + 1))
        occ = cv2.dilate(occ, kernel)
    return occ.astype(bool)


def overlay_occupancy(rgb: Image.Image, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    """Cyan = mask only, red = pred only, white = both."""
    base = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    out = base.copy()
    only_gt = gt & (~pred)
    only_pred = pred & (~gt)
    both = gt & pred
    out[only_gt] = 0.35 * base[only_gt] + 0.65 * np.array([0.0, 220.0, 220.0])
    out[only_pred] = 0.35 * base[only_pred] + 0.65 * np.array([220.0, 40.0, 40.0])
    out[both] = 0.25 * base[both] + 0.75 * np.array([240.0, 240.0, 240.0])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _subsample(points: np.ndarray, n_keep: int, rng: np.random.Generator) -> np.ndarray:
    """Random subset, or all points if already short."""
    if points.shape[0] <= n_keep:
        return points
    idx = rng.choice(points.shape[0], size=n_keep, replace=False)
    return points[idx]


def score_mesh_views(
    mesh_path: Path,
    npz_path: Path,
    mask_dir: Path,
    out_dir: Path | None = None,
    images_dir: Path | None = None,
    n_samples: int = N_SAMPLES,
    splat_px: int = SPLAT_PX,
    fill_hull: bool = True,
) -> dict[str, Any]:
    """Align ``mesh_path`` to view 0 and score silhouette IoU on every mask.

    Writes ``occupancy.json`` and ``occupancy.png`` when ``out_dir`` is set.
    """
    data = load_da3_npz(npz_path)
    depth = np.asarray(data["depth"])
    k_all = np.asarray(data["intrinsics"], dtype=np.float64)
    ext_all = np.asarray(data["extrinsics"], dtype=np.float64)
    n_view, d_h, d_w = depth.shape
    mask_paths = listed_images(mask_dir)
    if len(mask_paths) != n_view:
        raise ValueError(f"{mask_dir} has {len(mask_paths)} masks, npz has {n_view} views")
    masks = [load_mask_bool(path) for path in mask_paths]
    height, width = masks[0].shape
    mesh = load_triangle_mesh(mesh_path)
    vertices, faces = mesh_faces_for_occupancy(mesh)
    mask0 = resize_mask(masks[0], d_h, d_w)
    cloud = clip_cloud(masked_world_points(depth[0], k_all[0], ext_all[0], mask0))
    if cloud.shape[0] < 32:
        raise RuntimeError(f"view 0 has {cloud.shape[0]} unprojected points; cannot align")
    size_m = object_size_m(depth[0], k_all[0], mask0)
    scale0 = size_m / max(mesh_extent(vertices), 1e-9)
    k0 = scale_intrinsics(k_all[0], (d_h, d_w), (height, width))
    scale, rot, trans, iou0 = align_silhouette(
        vertices, faces, cloud, scale0, k0, ext_all[0], masks[0], height, width,
    )
    print(f"align scale={scale:.3f} view0_iou={iou0:.3f} size_m={size_m:.3f}", flush=True)
    world_verts = apply_sim(vertices, scale, rot, trans)
    rows: list[dict[str, Any]] = []
    pairs: list[tuple[Image.Image, Image.Image, str, str]] = []
    rgb_paths = listed_images(images_dir) if images_dir is not None else []
    for i in range(n_view):
        k_full = scale_intrinsics(k_all[i], (d_h, d_w), (height, width))
        pred = project_mesh_silhouette(world_verts, faces, k_full, ext_all[i], height, width)
        gt = masks[i]
        iou = silhouette_iou(pred, gt)
        precision, recall = occupancy_pr(pred, gt)
        fg = float(np.mean(gt))
        loose = fg >= LOOSE_MASK
        rows.append({
            "view": i, "iou": iou, "precision": precision, "recall": recall,
            "mask_fg": fg, "loose_mask": loose,
        })
        print(
            f"view {i:2d} iou={iou:.3f} p={precision:.3f} r={recall:.3f} fg={fg:.2f}"
            f"{' LOOSE' if loose else ''}",
            flush=True,
        )
        if rgb_paths:
            rgb = Image.open(rgb_paths[i]).convert("RGB")
            pairs.append((rgb, overlay_occupancy(rgb, gt, pred), f"{i} photo", f"{i} occ"))
    ious = [row["iou"] for row in rows]
    usable = [row["iou"] for row in rows if not row["loose_mask"]]
    report: dict[str, Any] = {
        "mesh": str(mesh_path),
        "npz": str(npz_path),
        "n_views": n_view,
        "align_scale": float(scale),
        "mean_iou": float(np.mean(ious)),
        "mean_iou_usable": float(np.mean(usable)) if usable else float("nan"),
        "min_iou": float(np.min(ious)),
        "views": rows,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "occupancy.json").write_text(json.dumps(report, indent=2) + "\n")
        if pairs:
            write_pair_preview(pairs, out_dir / "occupancy.png")
            print(f"occupancy preview → {out_dir / 'occupancy.png'}", flush=True)
        print(
            f"occupancy json → {out_dir / 'occupancy.json'} "
            f"mean_iou={report['mean_iou']:.3f} usable={report['mean_iou_usable']:.3f}",
            flush=True,
        )
    return report


def default_jobs(repo: Path) -> list[tuple[str, Path, Path, Path, Path, Path]]:
    """(label, mesh, npz, masks, images, out) for posed and hybrid on the five captures."""
    jobs: list[tuple[str, Path, Path, Path, Path, Path]] = []
    for name in CAPTURES:
        masks = repo / "work" / name / "scene" / "object"
        images = repo / "work" / name / "scene" / "images"
        for method in ("posed", "hybrid"):
            out = repo / "output" / f"{name}_{method}"
            jobs.append((
                f"{name}_{method}",
                out / "mesh.glb",
                repo / "work" / f"{name}_{method}" / "da3_output.npz",
                masks,
                images,
                out,
            ))
    return jobs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI: one GLB, or all default capture jobs when ``--mesh`` is omitted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--masks", type=Path, default=None)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--splat-px", type=int, default=SPLAT_PX)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Score one run, or every default posed/hybrid capture."""
    args = parse_args(argv)
    if args.mesh is not None:
        if args.npz is None or args.masks is None:
            raise RuntimeError("--mesh needs --npz and --masks")
        score_mesh_views(
            args.mesh, args.npz, args.masks, out_dir=args.out, images_dir=args.images,
            n_samples=args.n_samples, splat_px=args.splat_px,
        )
        return 0
    repo = Path(__file__).resolve().parent
    for label, mesh, npz, masks, images, out in default_jobs(repo):
        if not mesh.is_file() or not npz.is_file():
            print(f"skip {label} (missing mesh or npz)", flush=True)
            continue
        print(f"======== {label} ========", flush=True)
        score_mesh_views(
            mesh, npz, masks, out_dir=out, images_dir=images,
            n_samples=args.n_samples, splat_px=args.splat_px,
        )
    print("DONE occupancy", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
