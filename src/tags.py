"""AprilTag detections to a DA3-style npz (RealSense depth + PnP w2c)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from helpers import write_pair_preview
from vis import write_da3_npz

CornerKey = tuple[int, int]
Detections = Mapping[CornerKey, np.ndarray]
PoseFn = Callable[[Detections, dict[CornerKey, np.ndarray], np.ndarray], np.ndarray | None]


def _require_cv2() -> object:
    """Import OpenCV or raise a clear error."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("convert_apriltag_to_da3_npz needs opencv-python when detect/pose are not passed") from exc
    return cv2


def detect_apriltags(rgb_path: Path, dictionary: int, refine: bool) -> dict[CornerKey, np.ndarray]:
    """Detect tag corners in one RGB still: ``(tag_id, corner_i) → (u, v)``."""
    cv2 = _require_cv2()
    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise FileNotFoundError(rgb_path)
    params = cv2.aruco.DetectorParameters()
    if refine:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary), params)
    corners, ids, _ = detector.detectMarkers(bgr)
    found: dict[CornerKey, np.ndarray] = {}
    if ids is None:
        return found
    for pts, tag_id in zip(corners, ids.flatten()):
        for corner_i, xy in enumerate(pts.reshape(4, 2)):
            found[(int(tag_id), corner_i)] = xy.astype(np.float64)
    return found


def overlay_tags(
    rgb: Image.Image, detections: Detections, k_mat: np.ndarray | None = None,
) -> Image.Image:
    """Draw tag quads, ids, and RGB axes (X red, Y green, Z blue) on ``rgb``."""
    out = rgb.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    quads: dict[int, list[tuple[float, float] | None]] = {}
    for (tag_id, corner_i), uv in detections.items():
        pts = quads.setdefault(int(tag_id), [None, None, None, None])
        if 0 <= int(corner_i) < 4:
            pts[int(corner_i)] = (float(uv[0]), float(uv[1]))
    line_w = max(3, min(out.size) // 280)
    k_arr = np.asarray(k_mat, dtype=np.float64) if k_mat is not None else None
    for tag_id, pts in sorted(quads.items()):
        if any(p is None for p in pts):
            continue
        xy = np.array([(float(p[0]), float(p[1])) for p in pts if p is not None], dtype=np.float64)
        draw.line([tuple(p) for p in xy] + [tuple(xy[0])], fill=(0, 220, 255), width=line_w)
        draw.text(tuple(xy[0]), str(tag_id), fill=(255, 220, 0))
        _draw_tag_axes(draw, xy, k_arr, line_w)
    return out


def _planar_homography(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """3x3 H mapping planar ``src_xy`` (N,2) to ``dst_xy`` (N,2)."""
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(src_xy, dst_xy):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    h_mat = vt[-1].reshape(3, 3)
    return h_mat / h_mat[2, 2]


def _tag_pose(corners: np.ndarray, k_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Rotation and translation of a unit-side tag (Y up) from 4 image corners."""
    obj = np.array(
        [[-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0]],
        dtype=np.float64,
    )
    try:
        h_mat = _planar_homography(obj[:, :2], corners)
        kinv = np.linalg.inv(k_mat)
    except np.linalg.LinAlgError:
        return None
    h1, h2, h3 = h_mat[:, 0], h_mat[:, 1], h_mat[:, 2]
    nrm = float(np.linalg.norm(kinv @ h1))
    if nrm < 1e-12:
        return None
    lam = 1.0 / nrm
    r1 = lam * (kinv @ h1)
    r2 = lam * (kinv @ h2)
    t_vec = lam * (kinv @ h3)
    if t_vec[2] < 0:
        r1, r2, t_vec = -r1, -r2, -t_vec
    r3 = np.cross(r1, r2)
    rot = np.column_stack((r1, r2, r3))
    u_mat, _, vt = np.linalg.svd(rot)
    rot = u_mat @ vt
    if np.linalg.det(rot) < 0:
        rot[:, 2] *= -1
    return rot, t_vec


def _project_cam(xyz: np.ndarray, rot: np.ndarray, t_vec: np.ndarray, k_mat: np.ndarray) -> tuple[float, float] | None:
    """Project a tag-frame point to pixels, or None if behind the camera."""
    cam = rot @ xyz + t_vec
    if cam[2] <= 1e-9:
        return None
    uv = k_mat @ cam
    return float(uv[0] / uv[2]), float(uv[1] / uv[2])


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    origin: tuple[float, float],
    tip: tuple[float, float],
    color: tuple[int, int, int],
    width: int,
) -> None:
    """Draw a 2-D axis with a small arrow head."""
    ox, oy = origin
    tx, ty = tip
    draw.line([(ox, oy), (tx, ty)], fill=color, width=width)
    vx, vy = tx - ox, ty - oy
    nrm = max((vx * vx + vy * vy) ** 0.5, 1e-6)
    ux, uy = vx / nrm, vy / nrm
    px, py = -uy, ux
    ah = max(4.0, float(width * 3))
    p1 = (tx - ux * ah + px * ah * 0.4, ty - uy * ah + py * ah * 0.4)
    p2 = (tx - ux * ah - px * ah * 0.4, ty - uy * ah - py * ah * 0.4)
    draw.line([p1, (tx, ty), p2], fill=color, width=width)


def _draw_tag_axes(
    draw: ImageDraw.ImageDraw, corners: np.ndarray, k_mat: np.ndarray | None, width: int,
) -> None:
    """RGB axes at the tag: X red, Y green, Z blue (OpenCV ``drawFrameAxes``)."""
    axis_rgb = ((220, 40, 40), (40, 200, 60), (50, 90, 255))
    origin_uv: tuple[float, float] | None = None
    tips: list[tuple[float, float] | None] = [None, None, None]
    if k_mat is not None:
        pose = _tag_pose(corners, k_mat)
        if pose is not None:
            rot, t_vec = pose
            origin_uv = _project_cam(np.zeros(3), rot, t_vec, k_mat)
            axis_len = 1.5
            tips = [
                _project_cam(np.array([axis_len, 0.0, 0.0]), rot, t_vec, k_mat),
                _project_cam(np.array([0.0, axis_len, 0.0]), rot, t_vec, k_mat),
                _project_cam(np.array([0.0, 0.0, axis_len]), rot, t_vec, k_mat),
            ]
    if origin_uv is None or any(t is None for t in tips):
        origin = corners.mean(axis=0)
        right = 0.5 * ((corners[1] + corners[2]) / 2.0 - origin)
        up = 0.5 * ((corners[0] + corners[1]) / 2.0 - origin)
        z_xy = np.array([-right[1], right[0]], dtype=np.float64)
        z_n = float(np.linalg.norm(z_xy))
        if z_n > 1e-6:
            z_xy = z_xy * (0.45 * float(np.linalg.norm(right)) / z_n)
        origin_uv = (float(origin[0]), float(origin[1]))
        tips = [
            (float(origin[0] + right[0]), float(origin[1] + right[1])),
            (float(origin[0] + up[0]), float(origin[1] + up[1])),
            (float(origin[0] + z_xy[0]), float(origin[1] + z_xy[1])),
        ]
    for tip, color in zip(tips, axis_rgb):
        if tip is None:
            continue
        _draw_arrow(draw, origin_uv, tip, color, width)


def save_tag_preview(
    images: Sequence[Path],
    detections: Sequence[Detections],
    out_png: Path,
    ks: Sequence[np.ndarray] | None = None,
) -> Path:
    """Write a 3-column photo | tags grid with every dataset view."""
    if len(images) != len(detections):
        raise ValueError(f"images/detections length mismatch: {len(images)}, {len(detections)}")
    if ks is not None and len(ks) != len(images):
        raise ValueError(f"ks length {len(ks)} != {len(images)}")
    pairs: list[tuple[Image.Image, Image.Image, str, str]] = []
    for i, (path, det) in enumerate(zip(images, detections)):
        rgb = Image.open(path).convert("RGB")
        k_mat = None if ks is None else np.asarray(ks[i])
        pairs.append((rgb, overlay_tags(rgb, det, k_mat), f"{i} photo", f"{i} tags"))
    write_pair_preview(pairs, out_png)
    print(f"tags preview → {out_png}", flush=True)
    return out_png


def sample_depth(depth: np.ndarray, u: float, v: float, neigh: int) -> float:
    """Median of valid (Z > 0) depth in a ``neigh x neigh`` window, or 0."""
    if neigh < 1:
        raise ValueError(f"neigh must be >= 1, got {neigh}")
    height, width = depth.shape
    x, y = int(round(u)), int(round(v))
    rad = neigh // 2
    patch = depth[max(0, y - rad) : min(height, y + rad + 1), max(0, x - rad) : min(width, x + rad + 1)]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else 0.0


def unproject_pixel(u: float, v: float, z_m: float, k_mat: np.ndarray) -> np.ndarray:
    """Pixel + camera Z → camera XYZ (meters)."""
    fx, fy, cx, cy = k_mat[0, 0], k_mat[1, 1], k_mat[0, 2], k_mat[1, 2]
    return np.array([(u - cx) * z_m / fx, (v - cy) * z_m / fy, z_m], dtype=np.float64)


def cam_points(detections: Detections, depth: np.ndarray, k_mat: np.ndarray, neigh: int) -> dict[CornerKey, np.ndarray]:
    """Unproject detected corners that have valid depth."""
    out: dict[CornerKey, np.ndarray] = {}
    for key, uv in detections.items():
        z_m = sample_depth(depth, float(uv[0]), float(uv[1]), neigh)
        if z_m <= 0:
            continue
        out[key] = unproject_pixel(float(uv[0]), float(uv[1]), z_m, k_mat)
    return out


def solve_w2c(
    detections: Detections, world: dict[CornerKey, np.ndarray], k_mat: np.ndarray,
    pnp_flags: int, min_corners: int,
) -> np.ndarray | None:
    """3x4 world-to-camera from shared corners, or None if PnP fails."""
    keys = sorted(set(detections) & set(world))
    if len(keys) < min_corners:
        return None
    cv2 = _require_cv2()
    obj = np.stack([world[k] for k in keys], axis=0)
    img = np.stack([detections[k] for k in keys], axis=0)
    ok, rvec, tvec = cv2.solvePnP(obj, img, k_mat, np.zeros(5), flags=pnp_flags)
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    ext = np.zeros((3, 4), dtype=np.float64)
    ext[:3, :3] = rot
    ext[:3, 3] = tvec.reshape(3)
    return ext


def _c2w(ext: np.ndarray) -> np.ndarray:
    """4x4 camera-to-world from a 3x4 w2c."""
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :] = ext
    return np.linalg.inv(w2c)


def default_apriltag_dict() -> int:
    """``DICT_APRILTAG_16h5`` (the lab tags)."""
    cv2 = _require_cv2()
    return int(cv2.aruco.DICT_APRILTAG_16h5)


def _identity_w2c() -> np.ndarray:
    """3x4 identity world-to-camera (reference view)."""
    return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)


def _absorb_corners(world: dict[CornerKey, np.ndarray], cam: dict[CornerKey, np.ndarray], c2w: np.ndarray) -> int:
    """Add camera-frame corners that are not yet in ``world``. Return how many."""
    added = 0
    for key, xyz in cam.items():
        if key in world:
            continue
        homo = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
        world[key] = (c2w @ homo)[:3]
        added += 1
    return added


def grow_world(
    detections: Sequence[Detections],
    depths: Sequence[np.ndarray],
    ks: Sequence[np.ndarray],
    *,
    neigh: int,
    grow_steps: int,
    min_corners: int,
    pose_fn: PoseFn,
) -> tuple[dict[CornerKey, np.ndarray], list[np.ndarray | None]]:
    """Seed world from the view with the most unprojected corners, then grow."""
    if grow_steps < 1:
        raise ValueError(f"grow_steps must be >= 1, got {grow_steps}")
    scores = [len(cam_points(det, depth, k_mat, neigh)) for det, depth, k_mat in zip(detections, depths, ks)]
    ref_i = int(np.argmax(scores))
    world = cam_points(detections[ref_i], depths[ref_i], ks[ref_i], neigh)
    if len(world) < min_corners:
        raise RuntimeError(f"no view has {min_corners} unprojected tag corners")
    extras: list[np.ndarray | None] = [None] * len(detections)
    extras[ref_i] = _identity_w2c()
    print(f"reference view={ref_i} world_corners={len(world)}", flush=True)
    for step in range(grow_steps):
        added = 0
        for i, (det, depth, k_mat) in enumerate(zip(detections, depths, ks)):
            if extras[i] is None:
                extras[i] = pose_fn(det, world, k_mat)
            if extras[i] is None:
                continue
            added += _absorb_corners(world, cam_points(det, depth, k_mat, neigh), _c2w(extras[i]))
        print(f"grow step={step} world_corners={len(world)} posed={sum(e is not None for e in extras)} added={added}", flush=True)
        if added == 0:
            break
        updated: list[np.ndarray | None] = []
        for i, (det, k_mat) in enumerate(zip(detections, ks)):
            if i == ref_i:
                updated.append(extras[i])
                continue
            solved = pose_fn(det, world, k_mat)
            updated.append(solved if solved is not None else extras[i])
        extras = updated
    return world, extras


def convert_apriltag_to_da3_npz(
    images: Sequence[Path],
    depths: Sequence[np.ndarray],
    ks: Sequence[np.ndarray],
    npz_path: Path,
    *,
    neigh: int,
    grow_steps: int,
    min_corners: int,
    min_posed: int,
    pnp_flags: int,
    dictionary: int | None = None,
    refine: bool | None = None,
    detections: Sequence[Detections] | None = None,
    pose_fn: PoseFn | None = None,
    kept_images_dir: Path | None = None,
    keep_stems: Sequence[str] | None = None,
    preview_png: Path | None = None,
) -> tuple[Path, list[int]]:
    """Write a native-resolution DA3 npz from depth, factory K, and tag PnP.

    World points are unprojected tag corners (needs valid Z), not a printed size.
    Views with no pose are dropped. ``Z <= 0`` becomes NaN in the pointmaps.

    Pass ``detections`` to skip OpenCV detect. Pass ``pose_fn`` to skip OpenCV PnP.
    Detect defaults to ``DICT_APRILTAG_16h5`` and ``refine=False``. ``pnp_flags``
    is passed to ``cv2.solvePnP``. Writes ``tags_preview.png`` (photo | tags, all views).
    """
    n_view = len(images)
    if not (n_view == len(depths) == len(ks)):
        raise ValueError(f"images/depths/K length mismatch: {n_view}, {len(depths)}, {len(ks)}")
    if detections is None:
        if refine is None:
            refine = False
        detections = [
            detect_apriltags(path, default_apriltag_dict() if dictionary is None else dictionary, refine)
            for path in images
        ]
    if len(detections) != n_view:
        raise ValueError(f"detections length {len(detections)} != {n_view}")
    out_preview = preview_png if preview_png is not None else npz_path.parent / "tags_preview.png"
    save_tag_preview(images, detections, out_preview, ks=ks)
    solver: PoseFn = pose_fn if pose_fn is not None else (
        lambda det, world, k_mat: solve_w2c(det, world, k_mat, pnp_flags, min_corners)
    )
    _, extras = grow_world(
        detections, depths, ks, neigh=neigh, grow_steps=grow_steps, min_corners=min_corners, pose_fn=solver,
    )
    keep = [i for i, ext in enumerate(extras) if ext is not None]
    print(f"posed={len(keep)} dropped={[i for i in range(n_view) if i not in keep]}", flush=True)
    if len(keep) < min_posed:
        raise RuntimeError(f"posed {len(keep)} views; need at least {min_posed}")
    depth_n = np.stack([np.asarray(depths[i]) for i in keep], axis=0)
    k_n = np.stack([np.asarray(ks[i]) for i in keep], axis=0)
    ext_n = np.stack([extras[i] for i in keep], axis=0)
    stems = list(keep_stems) if keep_stems is not None else [path.stem for path in images]
    if len(stems) != n_view:
        raise ValueError(f"keep_stems length {len(stems)} != {n_view}")
    image_files: list[Path] = []
    if kept_images_dir is not None:
        if kept_images_dir.exists():
            for old in kept_images_dir.glob("*"):
                old.unlink()
        kept_images_dir.mkdir(parents=True, exist_ok=True)
        for j, i in enumerate(keep):
            dest = kept_images_dir / f"{j}.png"
            Image.open(images[i]).convert("RGB").save(dest)
            image_files.append(dest)
    else:
        image_files = [images[i] for i in keep]
    npz_path = write_da3_npz(
        npz_path, depth_n, ext_n, k_n, image_files, process_res=int(depth_n.shape[2]),
        keep_stems=[stems[i] for i in keep],
    )
    return npz_path, keep
