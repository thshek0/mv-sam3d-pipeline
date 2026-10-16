#!/usr/bin/env python3
"""Build one depth and one mesh, then write the summary boards once.

.. code-block:: bash

    python src/run_dataset.py NAME --depth rs --mesh tsdf --tags
    python src/run_dataset.py NAME --all --rs --tags

A name alone prints the command table. Existing depth npz and mesh glb files
are kept; ``--refresh`` rebuilds them. Summaries are always rewritten at the end.

Steps: ingest | depth SOURCE | mesh DEPTH MESH | summarize
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from helpers import (
    DEFAULT_MVSAM,
    DEFAULT_OUT,
    convert_realsense_dump,
    crop_views_depth_to_masks,
    dataset_depth_npz,
    dataset_root,
    env_python,
    identity_w2c,
    listed_images,
    load_realsense_depth_stack,
    mesh_name,
    place_depth_npz,
    save_mask_preview,
    write_dataset_summary,
    write_labelme_masks,
)
from methods import depth_hybrid, depth_npz_is_unposed, mesh_sam3d, mesh_tsdf
from pipeline import resolve_depth
from vis import write_dataset_outputs

STEMS_FILE = "keep_stems.txt"
RGB_SUFFIX = "_rgb.png"
DEPTH_SUFFIX = "_depth.png"
META_SUFFIX = ".json"
LABELME_SUFFIX = "_rgb.json"
DEFAULT_MARGIN = 1.15
DEFAULT_PROCESS_RES = 504
DEFAULT_MASKS = "masks"
DEPTH_CHOICES = ("da3", "da3_posed", "rs", "hybrid")
MESH_CHOICES = ("sam3d", "tsdf")


@dataclass(frozen=True)
class DatasetLayout:
    """Paths for one dataset under ``dataset/`` → ``output/``."""

    name: str
    dump: Path
    out_root: Path
    mvsam_root: Path
    masks_name: str = DEFAULT_MASKS

    @property
    def root(self) -> Path:
        """``output/<name>/``."""
        return dataset_root(self.out_root, self.name)

    @property
    def scene(self) -> Path:
        """``output/<name>/scene/``."""
        return self.root / "scene"

    @property
    def images(self) -> Path:
        """Staged RGB views."""
        return self.scene / "images"

    @property
    def masks(self) -> Path:
        """Staged RGBA masks."""
        return self.scene / self.masks_name

    @property
    def labelme_raw(self) -> Path:
        """LabelMe stills copied for mask conversion."""
        return self.scene / "labelme_raw"

    @property
    def depth(self) -> Path:
        """``output/<name>/depth/``."""
        return self.root / "depth"

    @property
    def stems_path(self) -> Path:
        """Dump stems in view order (one per line)."""
        return self.scene / STEMS_FILE

    def depth_npz(self, source: str) -> Path:
        """``depth/<source>.npz``."""
        return dataset_depth_npz(self.out_root, self.name, source)


def layout_for(
    name: str,
    *,
    dump: Path | None = None,
    out_root: Path | None = None,
    mvsam_root: Path | None = None,
    masks_name: str = DEFAULT_MASKS,
) -> DatasetLayout:
    """Build paths for ``name`` (defaults: ``dataset/<name>``, ``output/``)."""
    return DatasetLayout(
        name=name,
        dump=(dump or Path("dataset") / name).expanduser().resolve(),
        out_root=(out_root or DEFAULT_OUT).expanduser().resolve(),
        mvsam_root=(mvsam_root or DEFAULT_MVSAM).expanduser().resolve(),
        masks_name=masks_name,
    )


def _write_stems(path: Path, stems: Sequence[str]) -> None:
    """Write dump stems, one per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(stems) + "\n", encoding="utf-8")


def _read_stems(path: Path) -> list[str]:
    """Read dump stems written by ``ingest``."""
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run ingest first")
    stems = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not stems:
        raise RuntimeError(f"{path} is empty")
    return stems


def _dump_rgb_paths(layout: DatasetLayout, stems: Sequence[str] | None = None) -> list[Path]:
    """Full-frame dump RGBs (not mask-cropped ``scene/images``)."""
    if stems is None:
        stems = _read_stems(layout.stems_path)
    paths = [layout.dump / f"{stem}{RGB_SUFFIX}" for stem in stems]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing dump RGBs: {', '.join(missing[:5])}")
    return paths


def _require_da3_python() -> str:
    """Resolve ``DA3_PYTHON`` / ``SAM3D_PYTHON`` or raise."""
    py = env_python("DA3_PYTHON", "SAM3D_PYTHON")
    if not py:
        raise RuntimeError("set SAM3D_PYTHON (or DA3_PYTHON)")
    return py


def _require_mvsam_python() -> str:
    """Resolve ``MVSAM_PYTHON`` / ``SAM3D_PYTHON`` or raise."""
    py = env_python("MVSAM_PYTHON", "SAM3D_PYTHON")
    if not py:
        raise RuntimeError("set SAM3D_PYTHON (or MVSAM_PYTHON)")
    return py


def mesh_jobs_for(
    *,
    use_rs: bool,
    tags: bool,
    depths: Sequence[str] | None = None,
    meshes: Sequence[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Allowed ``(depth, mesh)`` pairs. Tagged = full matrix; unposed = no posed / no RS·hybrid TSDF."""
    if not use_rs:
        allowed = {("da3", "sam3d"), ("da3", "tsdf")}
    elif tags:
        allowed = {
            (d, m)
            for d in ("da3", "da3_posed", "rs", "hybrid")
            for m in ("sam3d", "tsdf")
        }
    else:
        allowed = {
            ("da3", "sam3d"), ("da3", "tsdf"),
            ("rs", "sam3d"),
            ("hybrid", "sam3d"),
        }

    depth_f = set(depths) if depths is not None else None
    mesh_f = set(meshes) if meshes is not None else None
    jobs = [
        (d, m) for d, m in sorted(allowed)
        if (depth_f is None or d in depth_f) and (mesh_f is None or m in mesh_f)
    ]
    if not jobs:
        raise RuntimeError(
            f"no valid mesh jobs for use_rs={use_rs} tags={tags} "
            f"depths={depths} meshes={meshes}"
        )
    return tuple(jobs)


def validate_mesh_job(depth_method: str, mesh_method: str, *, use_rs: bool, tags: bool) -> None:
    """Raise if ``(depth, mesh)`` is illegal for the capability mode."""
    allowed = set(mesh_jobs_for(use_rs=use_rs, tags=tags))
    if (depth_method, mesh_method) not in allowed:
        raise RuntimeError(
            f"illegal combo depth={depth_method} mesh={mesh_method} "
            f"for use_rs={use_rs} tags={tags}; allowed={sorted(allowed)}"
        )


def ingest(layout: DatasetLayout, *, skip_incomplete: bool = True) -> list[Path]:
    """Convert the dump + LabelMe into ``scene/images`` and ``scene/masks``."""
    if not layout.dump.is_dir():
        raise FileNotFoundError(layout.dump)
    layout.root.mkdir(parents=True, exist_ok=True)
    for path in (layout.images, layout.masks, layout.labelme_raw):
        if path.exists():
            shutil.rmtree(path)
    images, _depths, _ks, stems = convert_realsense_dump(
        layout.dump, layout.images,
        rgb_suffix=RGB_SUFFIX, depth_suffix=DEPTH_SUFFIX, meta_suffix=META_SUFFIX,
        labelme_suffix=LABELME_SUFFIX, raw_dir=layout.labelme_raw, skip_incomplete=skip_incomplete,
    )
    write_labelme_masks(layout.labelme_raw, images, layout.masks)
    _write_stems(layout.stems_path, stems)
    imgs = listed_images(layout.images)
    save_mask_preview(imgs, [layout.masks / p.name for p in imgs], layout.root / "masks_preview.png")
    write_dataset_summary(_dump_rgb_paths(layout, stems), layout.root / "dataset_summary.png")
    print(f"ingest → {layout.scene} views={len(imgs)}", flush=True)
    return imgs


def ingest_processed(layout: DatasetLayout, processed: Path) -> list[Path]:
    """Stage an existing ``images/`` + masks scene (``--no-rs``)."""
    processed = processed.expanduser().resolve()
    if (processed / "images").is_dir() and (processed / layout.masks_name).is_dir():
        scene_src = processed
    elif (processed / "scene" / "images").is_dir():
        scene_src = processed / "scene"
    else:
        raise RuntimeError(
            f"{processed} needs images/ and {layout.masks_name}/ (or scene/...)"
        )
    if not (scene_src / layout.masks_name).is_dir():
        raise RuntimeError(f"missing {scene_src / layout.masks_name}")
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.scene.mkdir(parents=True, exist_ok=True)
    for name in ("images", layout.masks_name):
        dest = layout.scene / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(scene_src / name, dest)
    imgs = listed_images(layout.images)
    save_mask_preview(imgs, [layout.masks / p.name for p in imgs], layout.root / "masks_preview.png")
    write_dataset_summary(imgs, layout.root / "dataset_summary.png")
    print(f"ingest processed → {layout.scene} views={len(imgs)}", flush=True)
    return imgs


def _sync_kept_views(layout: DatasetLayout, keep: Sequence[int], stems: Sequence[str]) -> list[str]:
    """Renumber images/masks to ``keep`` indices; return kept stems."""
    if list(keep) == list(range(len(stems))):
        return list(stems)
    tmp_images = layout.scene / "_images_keep"
    tmp_masks = layout.scene / "_masks_keep"
    if tmp_images.exists():
        shutil.rmtree(tmp_images)
    if tmp_masks.exists():
        shutil.rmtree(tmp_masks)
    tmp_images.mkdir(parents=True)
    tmp_masks.mkdir(parents=True)
    kept_stems: list[str] = []
    for new_i, old_i in enumerate(keep):
        shutil.copy2(layout.images / f"{old_i}.png", tmp_images / f"{new_i}.png")
        mask_src = layout.masks / f"{old_i}.png"
        if mask_src.is_file():
            shutil.copy2(mask_src, tmp_masks / f"{new_i}.png")
        kept_stems.append(stems[old_i])
    shutil.rmtree(layout.images)
    shutil.rmtree(layout.masks)
    tmp_images.rename(layout.images)
    tmp_masks.rename(layout.masks)
    _write_stems(layout.stems_path, kept_stems)
    write_dataset_summary(_dump_rgb_paths(layout, kept_stems), layout.root / "dataset_summary.png")
    print(f"kept {len(keep)}/{len(stems)} tagged views", flush=True)
    return kept_stems


def build_rs(
    layout: DatasetLayout,
    *,
    tags: bool,
    process_res: int = DEFAULT_PROCESS_RES,
    margin: float = DEFAULT_MARGIN,
    neigh: int = 5,
    grow_steps: int = 8,
    min_corners: int = 4,
    min_posed: int = 3,
    refine: bool = True,
) -> Path:
    """Write cropped ``depth/rs.npz`` (AprilTag poses or unposed identity)."""
    stems = _read_stems(layout.stems_path)
    imgs = listed_images(layout.images)
    if len(imgs) != len(stems):
        raise RuntimeError(f"{len(imgs)} images vs {len(stems)} stems; re-run ingest")
    depths, ks = load_realsense_depth_stack(
        layout.dump, stems, depth_suffix=DEPTH_SUFFIX, meta_suffix=META_SUFFIX,
    )
    layout.depth.mkdir(parents=True, exist_ok=True)
    out_npz = layout.depth_npz("rs")

    if tags:
        import cv2

        from tags import convert_apriltag_to_da3_npz

        rs_full = layout.depth / "_rs_full.npz"
        _, keep = convert_apriltag_to_da3_npz(
            imgs, list(depths), list(ks), rs_full,
            neigh=neigh, grow_steps=grow_steps, min_corners=min_corners, min_posed=min_posed,
            pnp_flags=int(cv2.SOLVEPNP_ITERATIVE), refine=refine,
            keep_stems=stems, preview_png=layout.root / "tags_preview.png",
        )
        _sync_kept_views(layout, keep, stems)
        imgs = listed_images(layout.images)
        full = np.load(rs_full)
        crop_views_depth_to_masks(
            imgs, layout.masks, list(full["depth"]), list(full["intrinsics"]), full["extrinsics"],
            out_npz, margin=margin, process_res=process_res,
        )
        if depth_npz_is_unposed(out_npz):
            raise RuntimeError(f"{out_npz} still marked unposed after --tags")
    else:
        crop_views_depth_to_masks(
            imgs, layout.masks, list(depths), list(ks), identity_w2c(len(imgs)),
            out_npz, margin=margin, process_res=process_res,
        )
        raw = dict(np.load(out_npz))
        raw["unposed_rs"] = np.asarray(1, dtype=np.int8)
        np.savez(out_npz, **raw)
    print(f"rs → {out_npz} tags={tags} unposed={depth_npz_is_unposed(out_npz)}", flush=True)
    return out_npz


def build_da3(layout: DatasetLayout, *, process_res: int = DEFAULT_PROCESS_RES) -> Path:
    """Run pose-free DA3 into ``depth/da3.npz``."""
    npz = resolve_depth(
        "da3",
        images_dir=layout.images, out_dir=layout.out_root, dataset=layout.name,
        mvsam_root=layout.mvsam_root, da3_python=_require_da3_python(),
        da3_npz=None, rs_npz=None, pose_npz=None, process_res=process_res,
    )
    dest = place_depth_npz(npz, layout.depth_npz("da3"))
    print(f"da3 → {dest}", flush=True)
    return dest


def build_da3_posed(layout: DatasetLayout, *, process_res: int = DEFAULT_PROCESS_RES) -> Path:
    """DA3 with AprilTag cameras; pose comes from posed ``depth/rs.npz``."""
    from helpers import REPO, run

    rs_npz = layout.depth_npz("rs")
    if not rs_npz.is_file():
        raise FileNotFoundError(f"missing {rs_npz}; run rs --tags first")
    if depth_npz_is_unposed(rs_npz):
        raise RuntimeError(f"{rs_npz} is unposed; da3_posed needs --tags")
    out = layout.depth_npz("da3_posed")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Must use the GPU env (huggingface / DA3), not the driver .venv.
    run(
        [
            _require_da3_python(), str(REPO / "src" / "da3_posed.py"),
            "--images", str(layout.images),
            "--pose-npz", str(rs_npz),
            "--output", str(out),
            "--process-res", str(process_res),
        ],
        cwd=REPO,
    )
    if not out.is_file():
        raise FileNotFoundError(f"da3_posed did not write {out}")
    print(f"da3_posed → {out}", flush=True)
    return out


def build_hybrid(layout: DatasetLayout) -> Path:
    """Fuse RS + DA3 into ``depth/hybrid.npz``."""
    rs_npz = layout.depth_npz("rs")
    da3_npz = layout.depth_npz("da3")
    if not rs_npz.is_file():
        raise FileNotFoundError(f"missing {rs_npz}; run rs first")
    if not da3_npz.is_file():
        raise FileNotFoundError(f"missing {da3_npz}; run da3 first")
    out = depth_hybrid(rs_npz, da3_npz, layout.images, layout.depth_npz("hybrid"))
    print(f"hybrid → {out} unposed={depth_npz_is_unposed(out)}", flush=True)
    return out


def summarize(layout: DatasetLayout) -> list[Path]:
    """Write dataset/depth/mesh summaries for whatever currently exists."""
    def _existing(source: str) -> Path | None:
        path = layout.depth_npz(source)
        return path if path.is_file() else None

    scene_imgs = listed_images(layout.images)
    written: list[Path] = []
    if layout.stems_path.is_file() and layout.dump.is_dir():
        written.append(
            write_dataset_summary(_dump_rgb_paths(layout), layout.root / "dataset_summary.png")
        )
    else:
        written.append(write_dataset_summary(scene_imgs, layout.root / "dataset_summary.png"))
    written.extend(
        write_dataset_outputs(
            layout.root, scene_imgs, layout.masks,
            da3_npz=_existing("da3"), rs_npz=_existing("rs"),
            da3_posed_npz=_existing("da3_posed"), hybrid_npz=_existing("hybrid"),
        )
    )
    print(f"summarize → {layout.root}", flush=True)
    return written


def build_mesh(
    layout: DatasetLayout,
    depth_method: str,
    mesh_method: str,
    *,
    voxel_length: float = 0.002,
    sdf_trunc: float = 0.006,
    depth_trunc: float = 2.0,
    keep_largest: bool = True,
    seal: bool = False,
    coacd: bool = False,
    refresh_summary: bool = False,
) -> Path:
    """Build ``meshes/{depth}_{mesh}_mesh.glb``. Summaries stay a separate step."""
    npz = layout.depth_npz(depth_method)
    if not npz.is_file():
        raise FileNotFoundError(f"missing {npz}; run the depth step first")
    if mesh_method == "tsdf" and depth_npz_is_unposed(npz):
        raise RuntimeError(
            f"{npz} is unposed (identity w2c). Multi-view TSDF needs AprilTag poses; "
            "use sam3d, or da3/da3_posed TSDF."
        )
    from helpers import mesh_glb_path

    stem = mesh_name(depth_method, mesh_method)
    glb = mesh_glb_path(layout.out_root, layout.name, stem)
    print(f"mesh {stem} → {glb}", flush=True)
    if mesh_method == "tsdf":
        path = mesh_tsdf(
            npz, layout.images, glb,
            voxel_length=voxel_length, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc,
            mask_dir=layout.masks, keep_largest=keep_largest,
        )
    else:
        path = mesh_sam3d(
            layout.scene, layout.masks_name, npz, glb,
            _require_mvsam_python(), layout.mvsam_root, seal=seal, coacd=coacd,
        )
    if refresh_summary:
        summarize(layout)
    return path


USAGE = """\
Choose one depth and one mesh:

  python src/run_dataset.py NAME --depth da3 --mesh sam3d
  python src/run_dataset.py NAME --depth da3 --mesh tsdf
  python src/run_dataset.py NAME --depth da3_posed --mesh sam3d --tags
  python src/run_dataset.py NAME --depth rs --mesh tsdf --tags
  python src/run_dataset.py NAME --depth hybrid --mesh tsdf --tags
  python src/run_dataset.py NAME --depth hybrid --mesh sam3d

Every legal pair:

  python src/run_dataset.py NAME --all --rs --tags
  python src/run_dataset.py NAME --all --rs
  python src/run_dataset.py NAME --all

--refresh rebuilds depth and mesh files that already exist.
Summaries are written once at the end.
Steps: ingest | depth da3|da3_posed|rs|hybrid | mesh DEPTH MESH | summarize
"""

_DEPTH_ORDER = ("rs", "da3", "da3_posed", "hybrid")


def depths_required(depth: str) -> tuple[str, ...]:
    """Depth npz files ``depth`` needs, parents first."""
    if depth == "da3":
        return ("da3",)
    if depth == "rs":
        return ("rs",)
    if depth == "da3_posed":
        return ("rs", "da3_posed")
    if depth == "hybrid":
        return ("rs", "da3", "hybrid")
    raise ValueError(f"unknown depth {depth!r}")


def require_recipe(depth: str, mesh: str, *, tags: bool) -> None:
    """Raise if this depth/mesh pair is illegal for ``tags``."""
    if depth == "da3" and tags:
        raise RuntimeError("--tags is only for RealSense depths (rs, da3_posed, hybrid)")
    use_rs = depth != "da3"
    try:
        validate_mesh_job(depth, mesh, use_rs=use_rs, tags=tags)
    except RuntimeError:
        if mesh == "tsdf" and depth in {"rs", "hybrid"} and not tags:
            raise RuntimeError(
                f"{depth} + tsdf needs AprilTag poses; pass --tags, or use --mesh sam3d"
            ) from None
        if depth == "da3_posed" and not tags:
            raise RuntimeError("da3_posed needs --tags") from None
        raise


def _dump_is_realsense(dump: Path) -> bool:
    """True when ``dump`` has RealSense ``*_rgb.png`` frames."""
    return dump.is_dir() and any(dump.glob(f"*{RGB_SUFFIX}"))


def _keep_file(path: Path, *, refresh: bool) -> bool:
    """True when ``path`` should be left as it is."""
    if refresh or not path.is_file():
        return False
    print(f"skip {path.name} (exists)", flush=True)
    return True


def ensure_scene(
    layout: DatasetLayout, *, refresh: bool, skip_incomplete: bool, processed: Path | None = None,
) -> None:
    """Ingest the dump unless staged images are already present."""
    if not refresh and layout.images.is_dir() and listed_images(layout.images):
        print(f"skip scene ({len(listed_images(layout.images))} views)", flush=True)
        return
    if _dump_is_realsense(layout.dump):
        ingest(layout, skip_incomplete=skip_incomplete)
        return
    ingest_processed(layout, processed or layout.dump)


def ensure_depth(
    layout: DatasetLayout,
    source: str,
    *,
    tags: bool,
    refresh: bool,
    process_res: int,
    margin: float,
) -> Path:
    """Build ``depth/<source>.npz`` unless it exists and ``refresh`` is false."""
    npz = layout.depth_npz(source)
    if _keep_file(npz, refresh=refresh):
        return npz
    if source == "rs":
        return build_rs(layout, tags=tags, process_res=process_res, margin=margin)
    if source == "da3":
        return build_da3(layout, process_res=process_res)
    if source == "da3_posed":
        return build_da3_posed(layout, process_res=process_res)
    if source == "hybrid":
        return build_hybrid(layout)
    raise ValueError(f"unknown depth {source!r}")


def _mesh_glb(layout: DatasetLayout, depth: str, mesh: str) -> Path:
    """``meshes/{depth}_{mesh}_mesh.glb`` without creating the folder."""
    stem = mesh_name(depth, mesh)
    return layout.root / "meshes" / f"{stem}_mesh.glb"


def ensure_mesh(layout: DatasetLayout, depth: str, mesh: str, *, refresh: bool, mesh_kw: dict[str, object]) -> Path:
    """Build one mesh unless its glb exists and ``refresh`` is false."""
    glb = _mesh_glb(layout, depth, mesh)
    if _keep_file(glb, refresh=refresh):
        return glb
    return build_mesh(layout, depth, mesh, refresh_summary=False, **mesh_kw)  # type: ignore[arg-type]


def _run_jobs(
    layout: DatasetLayout,
    jobs: Sequence[tuple[str, str]],
    *,
    tags: bool,
    refresh: bool,
    process_res: int,
    margin: float,
    skip_incomplete: bool,
    processed: Path | None,
    mesh_kw: dict[str, object],
) -> None:
    """Ingest, build the depths those jobs need, mesh, then summarize once."""
    ensure_scene(layout, refresh=refresh, skip_incomplete=skip_incomplete, processed=processed)
    needed = {source for depth, _mesh in jobs for source in depths_required(depth)}
    for source in _DEPTH_ORDER:
        if source in needed:
            ensure_depth(
                layout, source, tags=tags, refresh=refresh, process_res=process_res, margin=margin,
            )
    for depth, mesh in jobs:
        ensure_mesh(layout, depth, mesh, refresh=refresh, mesh_kw=mesh_kw)
    summarize(layout)


def run_pair(
    layout: DatasetLayout,
    depth: str,
    mesh: str,
    *,
    tags: bool,
    refresh: bool = False,
    process_res: int = DEFAULT_PROCESS_RES,
    margin: float = DEFAULT_MARGIN,
    skip_incomplete: bool = True,
    processed: Path | None = None,
    mesh_kw: dict[str, object] | None = None,
) -> None:
    """Build one depth/mesh pair and write summaries once."""
    require_recipe(depth, mesh, tags=tags)
    print(f"run {layout.name} depth={depth} mesh={mesh} tags={tags}", flush=True)
    _run_jobs(
        layout, ((depth, mesh),), tags=tags, refresh=refresh, process_res=process_res, margin=margin,
        skip_incomplete=skip_incomplete, processed=processed, mesh_kw=mesh_kw or {},
    )
    print(f"DONE → {layout.root}", flush=True)


def run_all(
    layout: DatasetLayout,
    *,
    use_rs: bool,
    tags: bool,
    refresh: bool = False,
    process_res: int = DEFAULT_PROCESS_RES,
    margin: float = DEFAULT_MARGIN,
    skip_incomplete: bool = True,
    depths: Sequence[str] | None = None,
    meshes: Sequence[str] | None = None,
    processed: Path | None = None,
    mesh_kw: dict[str, object] | None = None,
) -> None:
    """Every legal depth × mesh for this capture, then one summary pass."""
    if tags and not use_rs:
        raise RuntimeError("--tags requires --rs")
    print(f"all {layout.name} use_rs={use_rs} tags={tags}", flush=True)
    jobs = mesh_jobs_for(use_rs=use_rs, tags=tags, depths=depths, meshes=meshes)
    _run_jobs(
        layout, jobs, tags=tags, refresh=refresh, process_res=process_res, margin=margin,
        skip_incomplete=skip_incomplete, processed=processed, mesh_kw=mesh_kw or {},
    )
    print(f"DONE all → {layout.root}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, epilog=USAGE)
    parser.add_argument("name", help="Dataset folder name under dataset/ and output/")
    parser.add_argument("--dump", type=Path, default=None, help="RealSense dump or processed scene")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mvsam-root", type=Path, default=DEFAULT_MVSAM)
    parser.add_argument("--masks", default=DEFAULT_MASKS)
    parser.add_argument("--process-res", type=int, default=DEFAULT_PROCESS_RES)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--voxel-length", type=float, default=0.002)
    parser.add_argument("--sdf-trunc", type=float, default=0.006)
    parser.add_argument("--depth-trunc", type=float, default=2.0)
    parser.add_argument("--keep-largest", action="store_true", default=True)
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--coacd", action="store_true")
    parser.add_argument("--all", action="store_true", help="Every legal depth × mesh")
    parser.add_argument("--rs", action="store_true", help="With --all: this capture is a RealSense dump")
    parser.add_argument("--tags", action="store_true", help="AprilTag poses (required for rs/hybrid TSDF and da3_posed)")
    parser.add_argument("--refresh", action="store_true", help="Rebuild depth and mesh files that already exist")
    parser.add_argument("--depth", choices=DEPTH_CHOICES, default=None)
    parser.add_argument("--mesh", choices=MESH_CHOICES, default=None)
    parser.add_argument("--no-skip-incomplete", action="store_true")

    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Dump → scene/images + masks")
    p_ingest.add_argument("--no-skip-incomplete", action="store_true")
    p_ingest.add_argument("--processed", action="store_true", help="Stage processed scene instead of RS dump")

    p_depth = sub.add_parser("depth", help="Build one depth npz")
    p_depth.add_argument("source", choices=DEPTH_CHOICES)
    p_depth.add_argument("--tags", action="store_true")

    p_mesh = sub.add_parser("mesh", help="Build one mesh from a depth npz already on disk")
    p_mesh.add_argument("depth", choices=DEPTH_CHOICES)
    p_mesh.add_argument("mesh", choices=MESH_CHOICES)

    sub.add_parser("summarize", help="Rewrite dataset, depth, and mesh summary boards")
    return parser.parse_args(argv)


def _mesh_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Voxel and post flags shared by mesh builds."""
    return {
        "voxel_length": args.voxel_length,
        "sdf_trunc": args.sdf_trunc,
        "depth_trunc": args.depth_trunc,
        "keep_largest": args.keep_largest,
        "seal": args.seal,
        "coacd": args.coacd,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry."""
    args = parse_args(argv)
    if args.cmd is None and not args.all and args.depth is None and args.mesh is None:
        print(USAGE, end="")
        return 0
    layout = layout_for(
        args.name, dump=args.dump, out_root=args.output_dir,
        mvsam_root=args.mvsam_root, masks_name=args.masks,
    )
    mesh_kw = _mesh_kwargs(args)
    if args.cmd == "ingest":
        if args.processed:
            ingest_processed(layout, layout.dump)
        else:
            ingest(layout, skip_incomplete=not args.no_skip_incomplete)
    elif args.cmd == "depth":
        if args.source == "da3" and args.tags:
            raise RuntimeError("--tags is only for RealSense depths (rs, da3_posed, hybrid)")
        ensure_depth(
            layout, args.source, tags=bool(args.tags), refresh=True,
            process_res=args.process_res, margin=args.margin,
        )
    elif args.cmd == "mesh":
        build_mesh(layout, args.depth, args.mesh, **mesh_kw)  # type: ignore[arg-type]
    elif args.cmd == "summarize":
        summarize(layout)
    elif args.cmd is None and args.all:
        if args.depth is not None or args.mesh is not None:
            raise RuntimeError("pass --all or --depth/--mesh, not both")
        run_all(
            layout, use_rs=bool(args.rs), tags=bool(args.tags), refresh=bool(args.refresh),
            process_res=args.process_res, margin=args.margin,
            skip_incomplete=not args.no_skip_incomplete, mesh_kw=mesh_kw,
        )
    elif args.cmd is None:
        if args.rs:
            raise RuntimeError("--rs is only used with --all; the depth name selects RealSense")
        if args.depth is None or args.mesh is None:
            raise RuntimeError("pass both --depth and --mesh\n" + USAGE)
        run_pair(
            layout, args.depth, args.mesh, tags=bool(args.tags), refresh=bool(args.refresh),
            process_res=args.process_res, margin=args.margin,
            skip_incomplete=not args.no_skip_incomplete, mesh_kw=mesh_kw,
        )
    else:
        raise RuntimeError(f"unknown cmd {args.cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
