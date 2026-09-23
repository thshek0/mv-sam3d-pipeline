#!/usr/bin/env python3
"""Run MV-SAM3D on a processed scene (``images/`` + RGBA masks).

Video, stills, LabelMe, rembg, and crop live in ``helpers`` as converters.
Optional ``--seal`` / ``--coacd``.
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
    run_da3,
    run_mvsam,
    validate_scene,
    visualize_depth,
)


def run_output_dir(out_dir: Path, scene: str) -> Path:
    """Return ``output/<scene>/`` and create it."""
    path = out_dir / scene
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    parser.add_argument("--da3-npz", type=Path, default=None, help="Existing DA3 npz (skip run_da3)")
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


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a processed scene, then DA3 or npz, then MV-SAM3D. Optional S6/S7."""
    args = parse_args(argv)
    src = args.input.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    _require_processed_scene(src, args.object)
    scene = args.scene or src.name
    scene_dir = (args.work_dir / scene).resolve()
    images_dir = scene_dir / "images"
    da3_out = scene_dir / "da3"
    run_dir = run_output_dir(args.output_dir.resolve(), scene)

    print(f"scene_dir={scene_dir} src={src} out={run_dir}", flush=True)
    _stage_processed_scene(src, scene_dir, args.object)
    validate_scene(scene_dir, args.object)

    if args.da3_npz is not None:
        npz = args.da3_npz.expanduser().resolve()
        if not npz.is_file():
            raise FileNotFoundError(npz)
    elif args.da3_python:
        npz = run_da3(args.da3_python, args.mvsam_root, images_dir, da3_out)
    else:
        raise RuntimeError("Pass --da3-npz or --da3-python (DA3_PYTHON / SAM3D_PYTHON)")
    visualize_depth(npz, run_dir / "depth_preview.png", images_dir=images_dir, mode="per_view")
    if not args.mvsam_python:
        raise RuntimeError("Pass --mvsam-python or set MVSAM_PYTHON / SAM3D_PYTHON")
    glb = run_mvsam(args.mvsam_python, args.mvsam_root, scene_dir, args.object, npz, args.merge_da3_glb)
    raw = run_dir / "mesh.glb"
    shutil.copy2(glb, raw)
    from post import preview_mesh, seal_mesh, write_coacd

    preview_mesh(raw, run_dir / "mesh.png")
    if args.seal or args.coacd:
        closed = seal_mesh(raw)
        closed.export(run_dir / "sealed.stl")
        preview_mesh(run_dir / "sealed.stl", run_dir / "sealed.png")
        if args.coacd:
            write_coacd(closed, run_dir)
    print(f"DONE → {raw}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
