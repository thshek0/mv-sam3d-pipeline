"""Verify a DA3-style ``da3_output.npz`` with RGB|depth grids and a 3-view orbit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from helpers import listed_images, write_pair_preview

REQUIRED_KEYS: tuple[str, ...] = (
    "depth",
    "pointmaps_sam3d",
    "extrinsics",
    "intrinsics",
)
ORBIT_VIEWS: tuple[tuple[str, float, float], ...] = (
    ("front", 20.0, -60.0),
    ("side", 15.0, 30.0),
    ("top", 75.0, -90.0),
)
MAX_ORBIT_POINTS = 24000
DEPTH_RECON_ATOL = 1e-3


def write_da3_npz(
    npz_path: Path,
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_files: Sequence[Path | str],
    process_res: int,
    keep_stems: Sequence[str] | None = None,
) -> Path:
    """Write a DA3-style npz. ``Z <= 0`` is stored as NaN in the pointmaps."""
    depth_n = np.asarray(depth, dtype=np.float32)
    if depth_n.ndim != 3:
        raise ValueError(f"depth must be (N, H, W), got {depth_n.shape}")
    k_n = np.asarray(intrinsics, dtype=np.float32)
    ext_n = np.asarray(extrinsics, dtype=np.float32)
    pointmaps = np.stack([depth_to_pointmap(depth_n[i], k_n[i]) for i in range(depth_n.shape[0])], axis=0)
    pointmaps[depth_n <= 0] = np.nan
    payload: dict[str, np.ndarray] = {
        "depth": depth_n,
        "pointmaps": pointmaps.astype(np.float32),
        "pointmaps_sam3d": np.transpose(pointmaps, (0, 3, 1, 2)).astype(np.float32),
        "extrinsics": ext_n,
        "intrinsics": k_n,
        "image_files": np.array([str(p) for p in image_files]),
        "process_res": np.asarray(process_res),
    }
    if keep_stems is not None:
        payload["keep_stems"] = np.array(list(keep_stems))
    npz_path = npz_path.expanduser()
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **payload)
    print(f"npz → {npz_path} N={depth_n.shape[0]} {depth_n.shape[1]}x{depth_n.shape[2]}", flush=True)
    return npz_path


def depth_to_pointmap(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Unproject ``depth`` (H, W) with pinhole ``K`` to camera-frame XYZ (H, W, 3)."""
    height, width = depth.shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    vv, uu = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    zz = depth
    xx = (uu - cx) * zz / fx
    yy = (vv - cy) * zz / fy
    return np.stack([xx, yy, zz], axis=-1)


def _as_w2c44(ext: np.ndarray) -> np.ndarray:
    """Return a 4x4 world-to-camera matrix from (3, 4) or (4, 4)."""
    if ext.shape == (4, 4):
        return np.asarray(ext, dtype=np.float64)
    if ext.shape == (3, 4):
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :] = ext
        return mat
    raise ValueError(f"extrinsic shape {ext.shape} is not (3, 4) or (4, 4)")


def camera_to_world(points: np.ndarray, ext: np.ndarray) -> np.ndarray:
    """Map camera-frame points (N, 3) to world using a w2c extrinsic."""
    c2w = np.linalg.inv(_as_w2c44(ext))
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([points.astype(np.float64), ones], axis=1)
    return (homo @ c2w.T)[:, :3]


def load_da3_npz(npz_path: Path) -> dict[str, np.ndarray]:
    """Load and check the keys MV-SAM3D reads from ``da3_output.npz``."""
    path = npz_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = np.load(path)
    missing = [key for key in REQUIRED_KEYS if key not in raw.files]
    if missing:
        raise ValueError(f"{path} missing {missing}; have {raw.files}")
    data = {key: raw[key] for key in raw.files}
    depth = np.asarray(data["depth"])
    if depth.ndim != 3:
        raise ValueError(f"depth must be (N, H, W), got {depth.shape}")
    n_view, height, width = depth.shape
    pms = np.asarray(data["pointmaps_sam3d"])
    if pms.shape != (n_view, 3, height, width):
        raise ValueError(f"pointmaps_sam3d {pms.shape} != {(n_view, 3, height, width)}")
    if np.asarray(data["intrinsics"]).shape != (n_view, 3, 3):
        raise ValueError(f"intrinsics {data['intrinsics'].shape} != {(n_view, 3, 3)}")
    ext = np.asarray(data["extrinsics"])
    if ext.shape not in {(n_view, 3, 4), (n_view, 4, 4)}:
        raise ValueError(f"extrinsics {ext.shape} not (N, 3, 4) or (N, 4, 4)")
    return data


def verify_da3_npz(data: dict[str, np.ndarray]) -> dict[str, float]:
    """Check depth range and that stored pointmaps match depth + K."""
    depth = np.asarray(data["depth"])
    pms = np.asarray(data["pointmaps_sam3d"])
    k_all = np.asarray(data["intrinsics"])
    n_view = depth.shape[0]
    finite = float(np.isfinite(depth).mean())
    positive = float((depth > 0).mean())
    max_err = 0.0
    for i in range(n_view):
        recon = depth_to_pointmap(depth[i], k_all[i])
        stored = np.transpose(pms[i], (1, 2, 0))
        max_err = max(max_err, float(np.nanmax(np.abs(recon - stored))))
    if max_err > DEPTH_RECON_ATOL:
        raise ValueError(f"pointmaps_sam3d disagrees with depth+K (max abs {max_err:g})")
    stats = {
        "n_view": float(n_view),
        "finite_frac": finite,
        "positive_frac": positive,
        "depth_min": float(np.nanmin(depth)),
        "depth_max": float(np.nanmax(depth)),
        "pointmap_recon_err": max_err,
    }
    print(
        f"da3 npz: N={n_view} depth=[{stats['depth_min']:.4g}, {stats['depth_max']:.4g}] "
        f"finite={finite:.3f} z>0={positive:.3f} recon_err={max_err:.3g}",
        flush=True,
    )
    return stats


def _file_for_stem(folder: Path | None, stem: str) -> Path | None:
    """Return ``folder/{stem}.png`` or a listed still with that stem."""
    if folder is None or not folder.is_dir():
        return None
    cand = folder / f"{stem}.png"
    if cand.is_file():
        return cand
    matches = [p for p in listed_images(folder) if p.stem == stem]
    return matches[0] if matches else None


def _rgb_for_view(stem: str, images_dir: Path | None, height: int, width: int) -> np.ndarray | None:
    """Return an RGB array sized to the depth map, or None."""
    path = _file_for_stem(images_dir, stem)
    if path is None:
        return None
    rgb = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(rgb)


def _view_stems(data: dict[str, np.ndarray], n_view: int) -> list[str]:
    """Stems from ``image_files`` or ``0..N-1``."""
    if "image_files" not in data:
        return [str(i) for i in range(n_view)]
    return [Path(str(name)).stem for name in data["image_files"]]


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Map a depth plane to an RGB uint8 image (turbo)."""
    valid = np.isfinite(depth) & (depth > 0)
    plane = np.zeros(depth.shape, dtype=np.float64)
    if np.any(valid):
        lo, hi = float(depth[valid].min()), float(depth[valid].max())
        span = max(hi - lo, 1e-9)
        plane[valid] = (depth[valid] - lo) / span
    cmap = plt.get_cmap("turbo")
    rgb = (cmap(plane)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 20
    return rgb


def _write_per_view(
    data: dict[str, np.ndarray],
    images_dir: Path | None,
    out_png: Path,
) -> Path:
    """Write a 3-column photo | depth grid with every view."""
    depth = np.asarray(data["depth"])
    n_view, _, _ = depth.shape
    stems = _view_stems(data, n_view)
    pairs: list[tuple[Image.Image, Image.Image, str, str]] = []
    for i in range(n_view):
        rgb_path = _file_for_stem(images_dir, stems[i])
        if rgb_path is not None:
            photo = Image.open(rgb_path).convert("RGB")
        else:
            photo = Image.fromarray(_colorize_depth(depth[i]))
        depth_im = Image.fromarray(_colorize_depth(depth[i]))
        pairs.append((photo, depth_im, f"{i} photo", f"{i} depth"))
    write_pair_preview(pairs, out_png)
    print(f"depth per-view → {out_png}", flush=True)
    return out_png


def overlay_depth_visibility(rgb: Image.Image, depth: np.ndarray, obj: np.ndarray) -> Image.Image:
    """Darken the photo; cyan = object and Z>0, red = object and Z=0."""
    height, width = depth.shape
    photo = rgb.convert("RGB")
    if photo.size != (width, height):
        photo = photo.resize((width, height), Image.Resampling.BILINEAR)
    arr = np.asarray(photo, dtype=np.float64)
    out = arr * 0.35
    valid = obj & np.isfinite(depth) & (depth > 0)
    miss = obj & (~np.isfinite(depth) | (depth <= 0))
    out[valid] = 0.25 * arr[valid] + np.array([0.0, 200.0, 220.0])
    out[miss] = 0.25 * arr[miss] + np.array([220.0, 40.0, 40.0])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def write_depth_visibility_preview(
    depth: np.ndarray, images: Sequence[Path], mask_dir: Path, out_png: Path,
) -> Path:
    """Write a 3-column photo | depth-visibility grid with every view."""
    n_view = int(np.asarray(depth).shape[0])
    if len(images) != n_view:
        raise ValueError(f"{len(images)} RGBs, depth has {n_view} views")
    pairs: list[tuple[Image.Image, Image.Image, str, str]] = []
    for i, rgb_path in enumerate(images):
        z = np.asarray(depth[i])
        height, width = z.shape
        rgb = Image.open(rgb_path).convert("RGB")
        keep = _mask_keep(rgb_path.stem, mask_dir, height, width)
        obj = keep if keep is not None else np.zeros((height, width), dtype=bool)
        pairs.append((rgb, overlay_depth_visibility(rgb, z, obj), f"{i} photo", f"{i} vis"))
    write_pair_preview(pairs, out_png)
    print(f"depth visibility → {out_png}", flush=True)
    return out_png


def _mask_keep(stem: str, mask_dir: Path | None, height: int, width: int) -> np.ndarray | None:
    """Boolean HxW object mask (alpha or L > 0), or None if this view has no mask."""
    if mask_dir is None:
        return None
    path = _file_for_stem(mask_dir, stem)
    if path is None:
        return None
    img = Image.open(path)
    alpha = img.getchannel("A") if img.mode in {"RGBA", "LA"} else img.convert("L")
    alpha = alpha.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(alpha) > 0


def _resolve_mask_dir(
    images_dir: Path | None, mask_dir: Path | None, object_only: bool,
) -> Path | None:
    """Object-mask folder for the orbit, or None for the full frame."""
    if not object_only:
        return None
    if mask_dir is not None:
        if not mask_dir.is_dir():
            raise FileNotFoundError(mask_dir)
        return mask_dir
    if images_dir is None:
        print("warning: orbit object-only with no --masks / --images; using full frame", flush=True)
        return None
    cand = images_dir.parent / "object"
    if cand.is_dir():
        return cand
    print(f"warning: no {cand}; orbit uses full frame", flush=True)
    return None


def _orbit_points(
    data: dict[str, np.ndarray],
    images_dir: Path | None,
    orbit_z_min: float | None,
    orbit_z_max: float | None,
    mask_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample camera-frame pointmaps, lift to world, optional RGB colors.

    ``orbit_z_min`` / ``orbit_z_max`` clip camera Z for display only.
    When ``mask_dir`` is set, only pixels with mask alpha (or L) > 0 are kept.
    """
    depth = np.asarray(data["depth"])
    pms = np.asarray(data["pointmaps_sam3d"])
    ext = np.asarray(data["extrinsics"])
    n_view, _, height, width = pms.shape
    stems = _view_stems(data, n_view)
    stride = max(1, int(np.ceil(np.sqrt(n_view * height * width / MAX_ORBIT_POINTS))))
    worlds: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for i in range(n_view):
        xyz = np.transpose(pms[i], (1, 2, 0))[::stride, ::stride].reshape(-1, 3)
        z = depth[i, ::stride, ::stride].reshape(-1)
        keep = np.isfinite(xyz).all(axis=1) & np.isfinite(z) & (z > 0)
        if orbit_z_min is not None:
            keep = keep & (z >= orbit_z_min)
        if orbit_z_max is not None:
            keep = keep & (z <= orbit_z_max)
        obj = _mask_keep(stems[i], mask_dir, height, width)
        if obj is not None:
            keep = keep & obj[::stride, ::stride].reshape(-1)
        xyz = xyz[keep]
        if xyz.size == 0:
            continue
        worlds.append(camera_to_world(xyz, ext[i]))
        rgb = _rgb_for_view(stems[i], images_dir, height, width)
        if rgb is None:
            colors.append(np.full((xyz.shape[0], 3), 0.65))
        else:
            pix = rgb[::stride, ::stride].reshape(-1, 3)[keep] / 255.0
            colors.append(pix)
    if not worlds:
        raise ValueError("no valid depth points to plot" + (" in object masks" if mask_dir is not None else ""))
    return np.concatenate(worlds, axis=0), np.concatenate(colors, axis=0)


def _write_orbit(
    data: dict[str, np.ndarray],
    images_dir: Path | None,
    out_png: Path,
    orbit_z_min: float | None,
    orbit_z_max: float | None,
    mask_dir: Path | None = None,
) -> Path:
    """Write front / side / top scatters of the fused world point cloud."""
    pts, cols = _orbit_points(data, images_dir, orbit_z_min, orbit_z_max, mask_dir=mask_dir)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    hi = lo + span
    fig, axes = plt.subplots(1, len(ORBIT_VIEWS), figsize=(4.2 * len(ORBIT_VIEWS), 4.4), subplot_kw={"projection": "3d"})
    for ax, (title, elev, azim) in zip(np.atleast_1d(axes), ORBIT_VIEWS):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=0.4, linewidths=0)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(span)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"depth orbit → {out_png}", flush=True)
    return out_png


def visualize_depth(
    npz_path: Path,
    output: Path,
    images_dir: Path | None = None,
    mode: Literal["per_view", "orbit", "both"] = "both",
    orbit_z_min: float | None = None,
    orbit_z_max: float | None = None,
    mask_dir: Path | None = None,
    object_only: bool = True,
) -> list[Path]:
    """Load ``da3_output.npz``, check it, and write preview PNGs.

    ``per_view`` is a 3-column photo | depth grid of every view.
    ``orbit`` is front/side/top of the fused world cloud, object-mask pixels
    only by default (``object/`` next to ``images/``, or ``mask_dir``).
    ``orbit_z_min`` / ``orbit_z_max`` clip camera Z for the orbit only (npz
    unchanged). When object masks are available, also writes
    ``depth_visibility_preview.png`` (cyan = object and Z>0, red = object and
    Z=0). ``output`` is a PNG path or a directory. With ``mode=both``
    and a PNG ``output``, per-view stays at that path and orbit is
    ``{stem}_orbit.png``.
    """
    data = load_da3_npz(npz_path)
    verify_da3_npz(data)
    resolved_masks = _resolve_mask_dir(images_dir, mask_dir, object_only)
    if images_dir is not None:
        n_rgb = len(listed_images(images_dir))
        n_depth = int(np.asarray(data["depth"]).shape[0])
        if n_rgb != n_depth:
            print(
                f"warning: {images_dir} has {n_rgb} RGBs, npz has {n_depth} views "
                f"(same stems from another scene will look aligned but wrong)",
                flush=True,
            )
    output = output.expanduser()
    if output.suffix.lower() == ".png":
        stem = output.with_suffix("")
        per_path = output
        orbit_path = Path(f"{stem}_orbit.png") if mode == "both" else output
        out_dir = output.parent
    else:
        out_dir = output
        per_path = out_dir / "depth_per_view.png"
        orbit_path = out_dir / "depth_orbit.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if mode in ("per_view", "both"):
        written.append(_write_per_view(data, images_dir, per_path))
    if mode in ("orbit", "both"):
        written.append(_write_orbit(
            data, images_dir, orbit_path, orbit_z_min, orbit_z_max, mask_dir=resolved_masks,
        ))
    if resolved_masks is not None and images_dir is not None:
        stems = _view_stems(data, int(np.asarray(data["depth"]).shape[0]))
        vis_images = [_file_for_stem(images_dir, stem) for stem in stems]
        if all(path is not None for path in vis_images):
            vis_path = out_dir / "depth_visibility_preview.png"
            written.append(write_depth_visibility_preview(
                np.asarray(data["depth"]), vis_images, resolved_masks, vis_path,
            ))
    return written


def parse_args() -> argparse.Namespace:
    """CLI for ``visualize_depth``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, type=Path, help="da3_output.npz")
    parser.add_argument("--output", required=True, type=Path, help="PNG path or directory")
    parser.add_argument("--images", type=Path, default=None, help="Folder of RGB stills")
    parser.add_argument("--masks", type=Path, default=None, help="RGBA object masks (default: sibling object/)")
    parser.add_argument("--mode", choices=("per_view", "orbit", "both"), default="both")
    parser.add_argument("--orbit-z-min", type=float, default=None, help="Orbit camera-Z clip low (meters)")
    parser.add_argument("--orbit-z-max", type=float, default=None, help="Orbit camera-Z clip high (meters)")
    parser.add_argument("--full-orbit", action="store_true", help="Orbit the full frame, ignore object masks")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_depth(
        args.npz, args.output, images_dir=args.images, mode=args.mode,
        orbit_z_min=args.orbit_z_min, orbit_z_max=args.orbit_z_max,
        mask_dir=args.masks, object_only=not args.full_orbit,
    )
