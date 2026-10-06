"""Run Depth Anything 3 with known OpenCV w2c poses and K (AprilTag / factory)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from helpers import listed_images

REPO = Path(__file__).resolve().parents[1]
DA3_ROOT = REPO / "Depth-Anything-3"
MVSAM_RUNNER = REPO / "MV-SAM3D" / "scripts" / "run_da3.py"


def _as_w2c44(ext: np.ndarray) -> np.ndarray:
    """Pad (N, 3, 4) OpenCV w2c to (N, 4, 4)."""
    ext_n = np.asarray(ext, dtype=np.float64)
    if ext_n.ndim != 3:
        raise ValueError(f"extrinsics must be (N, 3, 4) or (N, 4, 4), got {ext_n.shape}")
    if ext_n.shape[1:] == (4, 4):
        return ext_n
    if ext_n.shape[1:] == (3, 4):
        out = np.zeros((ext_n.shape[0], 4, 4), dtype=np.float64)
        out[:, :3, :] = ext_n
        out[:, 3, 3] = 1.0
        return out
    raise ValueError(f"extrinsics must be (N, 3, 4) or (N, 4, 4), got {ext_n.shape}")


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
    extrinsics = np.asarray(prediction.extrinsics)
    intrinsics = np.asarray(prediction.intrinsics)
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
    print(
        f"da3 posed → {output} depth={depth.shape} "
        f"range=[{float(depth.min()):.4g}, {float(depth.max()):.4g}]",
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
