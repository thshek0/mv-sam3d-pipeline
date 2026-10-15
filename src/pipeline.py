#!/usr/bin/env python3
"""Compose a processed scene with one depth method, then one mesh method.

Depth: ``da3``, ``da3_posed``, ``rs``, ``hybrid``. Mesh: ``sam3d`` or ``tsdf``.
If both DA3 and RS exist, hybrid is always written as a side product.

Layout under ``output/<name>/``: ``scene/`` (images + masks), ``depth/{source}.npz``,
``meshes/{depth}_{mesh}_mesh.glb``, boards at the dataset root.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

from helpers import (
    DEFAULT_MVSAM,
    DEFAULT_OUT,
    dataset_depth_npz,
    dataset_name_from_input,
    dataset_root,
    dataset_scene_dir,
    env_python,
    listed_images,
    load_realsense_depth_stack,
    mesh_glb_path,
    method_output_slug,
    place_depth_npz,
    realsense_dump_stems,
    validate_scene,
    write_unposed_rs_npz,
)
from methods import (
    depth_da3,
    depth_da3_posed,
    depth_hybrid,
    depth_npz_is_unposed,
    mesh_sam3d,
    mesh_tsdf,
)
from vis import write_dataset_outputs


DEPTH_METHODS = ("da3", "da3_posed", "rs", "hybrid")
MESH_METHODS = ("sam3d", "tsdf")
DEFAULT_MASKS = "masks"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input", required=True, type=Path,
        help="Processed scene (images/ + masks/), or output/<name>/ with scene/",
    )
    parser.add_argument(
        "--masks", default=DEFAULT_MASKS,
        help=f"Mask folder name / MV-SAM3D --mask_prompt (default: {DEFAULT_MASKS})",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT, help="Results root (default: output/)")
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
    parser.add_argument("--rs-npz", type=Path, default=None, help="RealSense npz (``rs``, hybrid); posed or unposed")
    parser.add_argument(
        "--rs-dump", type=Path, default=None,
        help="RealSense dump folder; write unposed RS depth npz (no AprilTags needed)",
    )
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


def _require_file(path: Path | None) -> Path | None:
    """Resolve ``path`` and require it to exist when given."""
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def resolve_input_scene(src: Path, masks_name: str) -> tuple[Path, str]:
    """Return ``(scene_dir_with_images, dataset_name)`` for a processed input."""
    src = src.expanduser().resolve()
    if (src / "images").is_dir() and (src / masks_name).is_dir():
        return src, dataset_name_from_input(src)
    nested = src / "scene"
    if (nested / "images").is_dir() and (nested / masks_name).is_dir():
        return nested, src.name
    raise RuntimeError(
        f"{src} is not a processed scene (need images/ and {masks_name}/, "
        f"or scene/images + scene/{masks_name}). "
        "Build that layout with helpers.convert_stills_to_images, "
        "convert_video_to_frames, convert_labelme_to_masks, "
        "convert_rembg_to_masks, and crop_views_to_masks."
    )


def _stage_processed_scene(src: Path, scene_dir: Path, masks_name: str) -> None:
    """Copy ``images/`` and the mask folder into ``scene_dir`` unless already there."""
    if src.resolve() == scene_dir.resolve():
        return
    scene_dir.mkdir(parents=True, exist_ok=True)
    for name in ("images", masks_name):
        dest = scene_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src / name, dest)


def resolve_depth(
    method: str,
    *,
    images_dir: Path,
    out_dir: Path,
    dataset: str,
    mvsam_root: Path,
    da3_python: str | None,
    da3_npz: Path | None,
    rs_npz: Path | None,
    pose_npz: Path | None,
    process_res: int,
) -> Path:
    """Return a DA3-style npz for ``method`` under ``depth/{source}.npz``."""
    if method == "da3":
        dest = dataset_depth_npz(out_dir, dataset, "da3")
        existing = _require_file(da3_npz)
        if existing is not None:
            return place_depth_npz(existing, dest)
        if not da3_python:
            raise RuntimeError("Pass --da3-npz or --da3-python (DA3_PYTHON / SAM3D_PYTHON)")
        run_dir = dataset_depth_npz(out_dir, dataset, "da3").parent / "_da3_run"
        return place_depth_npz(depth_da3(da3_python, mvsam_root, images_dir, run_dir), dest)
    if method == "da3_posed":
        dest = dataset_depth_npz(out_dir, dataset, "da3_posed")
        existing = _require_file(da3_npz)
        if existing is not None:
            return place_depth_npz(existing, dest)
        if pose_npz is None:
            raise RuntimeError("da3_posed needs --pose-npz (or pass --da3-npz)")
        return depth_da3_posed(images_dir, pose_npz, dest, process_res)
    if method == "rs":
        dest = dataset_depth_npz(out_dir, dataset, "rs")
        path = _require_file(rs_npz) or _require_file(da3_npz)
        if path is None:
            raise RuntimeError("rs needs --rs-npz (or --da3-npz)")
        return place_depth_npz(path, dest)
    if method == "hybrid":
        if rs_npz is None or da3_npz is None:
            raise RuntimeError("hybrid needs --rs-npz and --da3-npz")
        dest = dataset_depth_npz(out_dir, dataset, "hybrid")
        return depth_hybrid(rs_npz, da3_npz, images_dir, dest)
    raise ValueError(f"unknown depth method {method!r}")


def build_unposed_rs_npz(rs_dump: Path, images_dir: Path, out_npz: Path) -> Path:
    """Load a RealSense dump into an unposed DA3-style npz aligned to ``images/``."""
    dump = rs_dump.expanduser().resolve()
    if not dump.is_dir():
        raise FileNotFoundError(dump)
    images = listed_images(images_dir)
    stems = realsense_dump_stems(
        dump, rgb_suffix="_rgb.png", depth_suffix="_depth.png", meta_suffix=".json",
        labelme_suffix="_rgb.json", skip_incomplete=True,
    )
    if len(stems) != len(images):
        stems = realsense_dump_stems(
            dump, rgb_suffix="_rgb.png", depth_suffix="_depth.png", meta_suffix=".json",
        )
    if len(stems) != len(images):
        raise ValueError(f"{dump} has {len(stems)} views; {images_dir} has {len(images)} images")
    depths, ks = load_realsense_depth_stack(
        dump, stems, depth_suffix="_depth.png", meta_suffix=".json",
    )
    return write_unposed_rs_npz(out_npz, depths, ks, images, keep_stems=stems)


def main(argv: Sequence[str] | None = None) -> int:
    """Stage under ``output/<name>/``, build depth, then SAM3D or TSDF unless ``--depth-only``."""
    args = parse_args(argv)
    method = args.method
    src = args.input.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    input_scene, name = resolve_input_scene(src, args.masks)
    if not listed_images(input_scene / "images"):
        raise RuntimeError(f"No images in {input_scene / 'images'}")

    out_root = args.output_dir.resolve()
    root = dataset_root(out_root, name)
    scene_dir = dataset_scene_dir(out_root, name)
    images_dir = scene_dir / "images"
    mask_dir = scene_dir / args.masks
    slug = method_output_slug(method, args.mesh)
    glb = mesh_glb_path(out_root, name, slug)

    print(f"dataset={root} scene={scene_dir} src={src} out={glb} method={method} mesh={args.mesh}", flush=True)
    _stage_processed_scene(input_scene, scene_dir, args.masks)
    validate_scene(scene_dir, args.masks)

    rs_side = _require_file(args.rs_npz)
    if args.rs_dump is not None:
        rs_side = build_unposed_rs_npz(
            args.rs_dump, images_dir, dataset_depth_npz(out_root, name, "rs"),
        )
    da3_side = _require_file(args.da3_npz)
    pose_npz = args.pose_npz.expanduser().resolve() if args.pose_npz else None

    npz = resolve_depth(
        method,
        images_dir=images_dir,
        out_dir=out_root,
        dataset=name,
        mvsam_root=args.mvsam_root,
        da3_python=args.da3_python,
        da3_npz=da3_side,
        rs_npz=rs_side,
        pose_npz=pose_npz,
        process_res=args.process_res,
    )

    da3_posed_side: Path | None = None
    hybrid_side: Path | None = None
    if method == "da3":
        da3_side = npz
    elif method == "da3_posed":
        da3_posed_side = npz
    elif method == "rs":
        rs_side = npz
    else:
        hybrid_side = npz

    if da3_side is not None:
        da3_side = place_depth_npz(da3_side, dataset_depth_npz(out_root, name, "da3"))
    if rs_side is not None:
        rs_side = place_depth_npz(rs_side, dataset_depth_npz(out_root, name, "rs"))
    if da3_posed_side is not None:
        da3_posed_side = place_depth_npz(da3_posed_side, dataset_depth_npz(out_root, name, "da3_posed"))

    if hybrid_side is None and rs_side is not None and da3_side is not None:
        hybrid_side = depth_hybrid(
            rs_side, da3_side, images_dir, dataset_depth_npz(out_root, name, "hybrid"),
        )
    elif hybrid_side is not None:
        hybrid_side = place_depth_npz(hybrid_side, dataset_depth_npz(out_root, name, "hybrid"))

    if not args.depth_only:
        if args.mesh == "tsdf":
            if depth_npz_is_unposed(npz):
                raise RuntimeError(
                    f"{npz} is unposed (identity w2c). Multi-view TSDF needs AprilTag "
                    "(or other) poses; use --mesh sam3d, or --method da3 / da3_posed --mesh tsdf."
                )
            mesh_tsdf(
                npz, images_dir, glb,
                voxel_length=args.voxel_length, sdf_trunc=args.sdf_trunc, depth_trunc=args.depth_trunc,
                mask_dir=mask_dir, keep_largest=args.keep_largest,
            )
        else:
            mesh_sam3d(
                scene_dir, args.masks, npz, glb, args.mvsam_python, args.mvsam_root,
                merge_da3_glb=args.merge_da3_glb, seal=args.seal, coacd=args.coacd,
            )
    else:
        print(f"DONE depth → {npz}", flush=True)

    write_dataset_outputs(
        root, listed_images(images_dir), mask_dir,
        da3_npz=da3_side, rs_npz=rs_side,
        da3_posed_npz=da3_posed_side, hybrid_npz=hybrid_side,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
