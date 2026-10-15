"""Run Depth Anything 3 with known OpenCV w2c poses and K (AprilTag / factory)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from helpers import listed_images
from vis import _as_w2c44

REPO = Path(__file__).resolve().parents[1]
DA3_ROOT = REPO / "Depth-Anything-3"
MVSAM_RUNNER = REPO / "MV-SAM3D" / "scripts" / "run_da3.py"


def run_da3_posed(images_dir: Path, pose_npz: Path, output: Path, process_res: int) -> Path:
    """Load PnP cameras, run DA3, write a SAM3D-compatible npz. Returns ``output``."""
    if not DA3_ROOT.is_dir():
        raise FileNotFoundError(DA3_ROOT)
    sys.path.insert(0, str(DA3_ROOT / "src"))
    sys.path.insert(0, str(MVSAM_RUNNER.parent))
    from depth_anything_3.api import DepthAnything3
    from run_da3 import _checkpoint_dir, depth_to_pointmap, pointmap_to_sam3d_format

    images = listed_images(images_dir)
    if not images:
        raise FileNotFoundError(f"no pngs in {images_dir}")
    raw = np.load(pose_npz)
    ext = _as_w2c44(raw["extrinsics"])
    k_mat = np.asarray(raw["intrinsics"], dtype=np.float64)
    if ext.shape[0] != len(images) or k_mat.shape[0] != len(images):
        raise ValueError(f"pose npz N={ext.shape[0]} K={k_mat.shape[0]} images={len(images)}")
    model_path = None
    for candidate in (
        DA3_ROOT / "checkpoints" / "DA3NESTED-GIANT-LARGE",
        Path.home() / ".cache" / "huggingface" / "hub" / "models--depth-anything--DA3NESTED-GIANT-LARGE",
    ):
        resolved = _checkpoint_dir(candidate)
        if resolved is not None:
            model_path = str(resolved)
            break
    if model_path is None:
        model_path = "depth-anything/DA3NESTED-GIANT-LARGE"
    print(f"Loading DA3 from {model_path}", flush=True)
    model = DepthAnything3.from_pretrained(model_path).to("cuda")
    print(f"DA3 posed N={len(images)} process_res={process_res}", flush=True)
    prediction = model.inference(
        image=[str(path) for path in images],
        extrinsics=ext.astype(np.float32),
        intrinsics=k_mat.astype(np.float32),
        align_to_input_ext_scale=True,
        process_res=process_res,
        export_dir=None,
    )
    depth = np.asarray(prediction.depth, dtype=np.float32)
    # Keep AprilTag / RS cameras (metric). DA3's align_to_input_ext_scale can leave
    # depth in a nonsense unit (~1e7). Rescale each view to RS Z when the pose npz
    # has depth — a single global scale still leaves ~2× per-view drift that breaks TSDF.
    extrinsics = ext.astype(np.float32)
    intrinsics = np.asarray(k_mat, dtype=np.float32)
    if "depth" in raw.files:
        rs_z = np.asarray(raw["depth"], dtype=np.float32)
        if rs_z.shape != depth.shape:
            raise ValueError(f"pose depth {rs_z.shape} != DA3 depth {depth.shape}")
        scales = np.full(depth.shape[0], np.nan, dtype=np.float64)
        for i in range(depth.shape[0]):
            keep = (
                (rs_z[i] > 0) & np.isfinite(rs_z[i]) & (rs_z[i] < 2.0)
                & (depth[i] > 0) & np.isfinite(depth[i])
            )
            if not np.any(keep):
                continue
            scales[i] = float(np.median(rs_z[i][keep] / np.maximum(depth[i][keep], 1e-6)))
        if not np.any(np.isfinite(scales)):
            raise RuntimeError("no overlap to scale da3_posed depth to RS")
        fallback = float(np.nanmedian(scales))
        scales = np.where(np.isfinite(scales), scales, fallback)
        depth = depth * scales.astype(np.float32)[:, None, None]
        print(
            f"da3_posed scale→RS per-view med={fallback:.6g} "
            f"min={float(np.min(scales)):.6g} max={float(np.max(scales)):.6g}",
            flush=True,
        )
    pointmaps = np.stack([depth_to_pointmap(depth[i], intrinsics[i]) for i in range(depth.shape[0])], axis=0)
    pointmaps_sam3d = np.stack([pointmap_to_sam3d_format(pointmaps[i]) for i in range(depth.shape[0])], axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        depth=depth,
        pointmaps=pointmaps.astype(np.float32),
        pointmaps_sam3d=pointmaps_sam3d.astype(np.float32),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_files=np.array([str(path) for path in images]),
        process_res=np.asarray(process_res),
    )
    valid = depth[np.isfinite(depth) & (depth > 0)]
    print(
        f"da3 posed → {output} depth={depth.shape} "
        f"range=[{float(valid.min()):.4g}, {float(valid.max()):.4g}]",
        flush=True,
    )
    return output


def parse_args() -> argparse.Namespace:
    """CLI for pose-conditioned DA3."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--pose-npz", required=True, type=Path, help="DA3-style npz with extrinsics + intrinsics")
    parser.add_argument("--output", required=True, type=Path, help="Output da3_output.npz")
    parser.add_argument("--process-res", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    """CLI wrapper around ``run_da3_posed``."""
    args = parse_args()
    run_da3_posed(args.images, args.pose_npz, args.output, args.process_res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
