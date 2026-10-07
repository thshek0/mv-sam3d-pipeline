#!/usr/bin/env python3
"""Compose a processed scene with one depth method, then one mesh method.

Depth methods: ``da3``, ``da3_posed``, ``rs`` (existing npz), ``hybrid``
(aliases ``rs_da3``, ``fill_all_holes``). Mesh: SAM3D (default) or ``tsdf``.
Converters stay functions in ``helpers``. Optional ``--seal`` / ``--coacd``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

from helpers import (
    DEFAULT_MVSAM,
    DEFAULT_OUT,
    DEFAULT_WORK,
    env_python,
    listed_images,
    method_output_slug,
    run_output_dir,
    validate_scene,
)
from methods import depth_da3, depth_da3_posed, depth_rs_da3, mesh_sam3d, mesh_tsdf

DEPTH_METHODS = ("da3", "da3_posed", "rs", "hybrid", "rs_da3", "fill_all_holes")
MESH_METHODS = ("sam3d", "tsdf")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input", required=True, type=Path,
        help="Processed scene folder (images/ plus object/ RGBA masks)",
    )
    parser.add_argument("--scene", default=None, help="Scene stem (default: folder name)")
    parser.add_argument("--object", default="object", help="Mask folder name (--mask_prompt)")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK, help="Intermediate scene root")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT, help="Final output root")
    parser.add_argument("--mvsam-root", type=Path, default=DEFAULT_MVSAM, help="MV-SAM3D clone")
    parser.add_argument(
        "--da3-python",
        default=env_python("DA3_PYTHON", "SAM3D_PYTHON"),
        help="Python for DA3 (or set DA3_PYTHON / SAM3D_PYTHON)",
    )
    parser.add_argument(
        "--mvsam-python",
        default=env_python("MVSAM_PYTHON", "SAM3D_PYTHON"),
        help="Python for MV-SAM3D (or set MVSAM_PYTHON / SAM3D_PYTHON)",
    )
    parser.add_argument("--method", choices=DEPTH_METHODS, default="da3", help="How to build the depth npz")
    parser.add_argument("--mesh", choices=MESH_METHODS, default="sam3d", help="How to turn that npz into a mesh")
    parser.add_argument("--da3-npz", type=Path, default=None, help="Existing pose-free or any DA3-style npz")
    parser.add_argument("--rs-npz", type=Path, default=None, help="RealSense + PnP npz (``rs``, ``rs_da3``)")
    parser.add_argument("--pose-npz", type=Path, default=None, help="w2c + K for ``da3_posed``")
    parser.add_argument("--process-res", type=int, default=504, help="DA3 process_res for ``da3_posed``")
    parser.add_argument("--depth-only", action="store_true", help="Stop after writing the depth npz / previews")
    parser.add_argument("--voxel-length", type=float, default=0.002)
    parser.add_argument("--sdf-trunc", type=float, default=0.006)
    parser.add_argument("--depth-trunc", type=float, default=2.0)
    parser.add_argument("--keep-largest", action="store_true")
    parser.add_argument("--seal", action="store_true", help="Largest shell + fill holes")
    parser.add_argument("--coacd", action="store_true", help="CoACD t=0.05 (implies --seal)")
    parser.add_argument("--merge-da3-glb", action="store_true", help="Pass --merge_da3_glb to MV-SAM3D")
    return parser.parse_args(argv)


def _require_processed_scene(src: Path, object_name: str) -> None:
    """Raise if ``src`` is not ``images/`` plus a matching mask folder."""
    images_dir = src / "images"
    mask_dir = src / object_name
    if not images_dir.is_dir() or not mask_dir.is_dir():
        raise RuntimeError(
            f"{src} is not a processed scene (need images/ and {object_name}/). "
            "Build that layout with helpers.convert_stills_to_images, "
            "convert_video_to_frames, convert_labelme_to_masks, "
            "convert_rembg_to_masks, and crop_views_to_masks."
        )
    if not listed_images(images_dir):
        raise RuntimeError(f"No images in {images_dir}")


def _stage_processed_scene(src: Path, scene_dir: Path, object_name: str) -> None:
    """Copy ``images/`` and the mask folder into ``scene_dir`` unless already there."""
    if src.resolve() == scene_dir.resolve():
        return
    scene_dir.mkdir(parents=True, exist_ok=True)
    for name in ("images", object_name):
        dest = scene_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src / name, dest)


def resolve_depth(
    method: str,
    *,
    images_dir: Path,
    mask_dir: Path,
    work_dir: Path,
    mvsam_root: Path,
    da3_python: str | None,
    da3_npz: Path | None,
    rs_npz: Path | None,
    pose_npz: Path | None,
    process_res: int,
) -> Path:
    """Return a DA3-style npz for ``method``. Does not run a mesher."""
    if method == "da3":
        if da3_npz is not None:
            path = da3_npz.expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        if not da3_python:
            raise RuntimeError("Pass --da3-npz or --da3-python (DA3_PYTHON / SAM3D_PYTHON)")
        return depth_da3(da3_python, mvsam_root, images_dir, work_dir / "da3")
    if method == "da3_posed":
        if da3_npz is not None:
            path = da3_npz.expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        if pose_npz is None:
            raise RuntimeError("da3_posed needs --pose-npz (or pass --da3-npz)")
        return depth_da3_posed(images_dir, pose_npz, work_dir / "da3_output.npz", process_res)
    if method == "rs":
        if rs_npz is None and da3_npz is None:
            raise RuntimeError("rs needs --rs-npz (or --da3-npz)")
        path = (rs_npz or da3_npz).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if method in ("hybrid", "rs_da3", "fill_all_holes"):
        if rs_npz is None or da3_npz is None:
            raise RuntimeError("hybrid needs --rs-npz and --da3-npz")
        return depth_rs_da3(rs_npz, da3_npz, images_dir, mask_dir, work_dir / "da3_output.npz")
    raise ValueError(f"unknown depth method {method!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Stage a scene, build depth, then SAM3D or TSDF unless ``--depth-only``."""
    args = parse_args(argv)
    src = args.input.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    _require_processed_scene(src, args.object)
    scene = args.scene or src.name
    scene_dir = (args.work_dir / scene).resolve()
    images_dir = scene_dir / "images"
    slug = method_output_slug(args.method, args.mesh)
    run_dir = run_output_dir(args.output_dir.resolve(), scene, slug)

    print(
        f"scene_dir={scene_dir} src={src} out={run_dir} method={args.method} mesh={args.mesh}",
        flush=True,
    )
    _stage_processed_scene(src, scene_dir, args.object)
    validate_scene(scene_dir, args.object)

    npz = resolve_depth(
        args.method,
        images_dir=images_dir,
        mask_dir=scene_dir / args.object,
        work_dir=scene_dir,
        mvsam_root=args.mvsam_root,
        da3_python=args.da3_python,
        da3_npz=args.da3_npz,
        rs_npz=args.rs_npz,
        pose_npz=args.pose_npz,
        process_res=args.process_res,
    )
    if args.depth_only:
        print(f"DONE depth → {npz}", flush=True)
        return 0
    if args.mesh == "tsdf":
        mesh_tsdf(
            npz, images_dir, run_dir / "mesh.glb",
            voxel_length=args.voxel_length, sdf_trunc=args.sdf_trunc, depth_trunc=args.depth_trunc,
            mask_dir=scene_dir / args.object, keep_largest=args.keep_largest,
        )
        print(f"DONE → {run_dir / 'mesh.glb'}", flush=True)
        return 0
    mesh_sam3d(
        scene_dir, args.object, npz, run_dir, args.mvsam_python, args.mvsam_root,
        merge_da3_glb=args.merge_da3_glb, seal=args.seal, coacd=args.coacd,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
