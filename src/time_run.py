#!/usr/bin/env python3
"""Time ``da3`` → SAM3D on any processed scene (N repeats + summary)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from helpers import (
    DEFAULT_MVSAM,
    REPO,
    depth_rmse,
    env_python,
    listed_images,
    mean_sd,
    run_da3,
    run_mvsam,
    timed,
    valid_depth,
    validate_scene,
)
from post import load_triangle_mesh, preview_mesh
from vis import visualize_depth

DEFAULT_SCENE = REPO / "examples/gripper/processed"
DEFAULT_GRIPPER_OUT = REPO / "archive/gripper_timed"


def default_out_dir(src: Path) -> Path:
    """``archive/gripper_timed`` for the example scene, else ``archive/<name>_timed``."""
    resolved = src.expanduser().resolve()
    if resolved == DEFAULT_SCENE.resolve():
        return DEFAULT_GRIPPER_OUT
    return REPO / "archive" / f"{resolved.name}_timed"


def discard_run_dirs(root: Path) -> None:
    """Delete ``run_*`` and ``work/`` under ``root``. Leaves ``summary.json`` and sheets."""
    for path in root.glob("run_*"):
        if path.is_dir():
            shutil.rmtree(path)
    work = root / "work"
    if work.is_dir():
        shutil.rmtree(work)


def chamfer_m(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, n_pts: int = 4000) -> float:
    """Symmetric mean nearest-neighbor distance between ``n_pts`` samples (meters)."""
    pa = np.asarray(mesh_a.sample(n_pts), dtype=np.float64)
    pb = np.asarray(mesh_b.sample(n_pts), dtype=np.float64)
    d_ab = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(axis=2)).min(axis=1).mean()
    d_ba = np.sqrt(((pb[:, None, :] - pa[None, :, :]) ** 2).sum(axis=2)).min(axis=1).mean()
    return float(0.5 * (d_ab + d_ba))


def _stage(src: Path, work: Path, object_name: str) -> Path:
    """Copy a processed scene into ``work`` unless it is already there."""
    if (work / "images").is_dir() and (work / object_name).is_dir():
        validate_scene(work, object_name)
        return work
    work.mkdir(parents=True, exist_ok=True)
    for name in ("images", object_name):
        dest = work / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src / name, dest)
    validate_scene(work, object_name)
    return work


def run_once(
    run_dir: Path,
    scene_dir: Path,
    object_name: str,
    da3_python: str,
    mvsam_python: str,
    mvsam_root: Path,
    *,
    preview: bool = True,
) -> dict[str, Any]:
    """One da3 + optional visualize + SAM3D + mesh.png. Writes ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    images = scene_dir / "images"
    n_view = len(listed_images(images))
    da3_out = run_dir / "da3"
    npz, t_da3 = timed(lambda: run_da3(da3_python, mvsam_root, images, da3_out))
    shutil.copy2(npz, run_dir / "da3_output.npz")
    t_vis = 0.0
    if preview:
        _, t_vis = timed(lambda: visualize_depth(
            npz, run_dir / "depth_preview.png", images_dir=images, mode="both",
            mask_dir=scene_dir / object_name,
        ))
    glb, t_sam = timed(lambda: run_mvsam(mvsam_python, mvsam_root, scene_dir, object_name, npz, False))
    shutil.copy2(glb, run_dir / "mesh.glb")
    _, t_prev = timed(lambda: preview_mesh(run_dir / "mesh.glb", run_dir / "mesh.png"))
    mesh = load_triangle_mesh(run_dir / "mesh.glb")
    rec = {
        "n_view": n_view,
        "seconds": {
            "da3": t_da3,
            "visualize": t_vis,
            "sam3d": t_sam,
            "mesh_preview": t_prev,
            "total": t_da3 + t_vis + t_sam + t_prev,
        },
        "per_view": {
            "da3": t_da3 / n_view,
            "sam3d": t_sam / n_view,
            "total": (t_da3 + t_vis + t_sam + t_prev) / n_view,
        },
        "mesh": {"n_vert": int(mesh.vertices.shape[0]), "n_face": int(mesh.faces.shape[0])},
    }
    (run_dir / "timing.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"{run_dir.name} total={rec['seconds']['total']:.1f}s faces={rec['mesh']['n_face']}", flush=True)
    return rec


def summarize(root: Path) -> dict[str, Any]:
    """Mean±sd timings and depth/mesh spread across ``run_*/``."""
    runs = sorted(root.glob("run_*"))
    runs = [p for p in runs if (p / "timing.json").is_file() and (p / "da3_output.npz").is_file()]
    if not runs:
        raise FileNotFoundError(f"no completed runs in {root}")
    timings = [json.loads((p / "timing.json").read_text()) for p in runs]
    n_view = int(timings[0]["n_view"])
    steps = ("da3", "visualize", "sam3d", "mesh_preview", "total")
    seconds: dict[str, dict[str, float]] = {}
    per_view: dict[str, dict[str, float]] = {}
    for step in steps:
        vals = [float(t["seconds"].get(step, 0.0)) for t in timings]
        mean, sd = mean_sd(vals)
        seconds[step] = {"mean": mean, "sd": sd, "n": float(len(vals))}
        if step in ("da3", "sam3d", "total"):
            pv = [float(t["per_view"][step]) for t in timings]
            pmean, psd = mean_sd(pv)
            per_view[step] = {"mean": pmean, "sd": psd}
    depths = [np.asarray(np.load(p / "da3_output.npz")["depth"], dtype=np.float32) for p in runs]
    rmse_vs0 = [depth_rmse(depths[0], d) for d in depths]
    stack = np.stack(depths, axis=0)
    valid = valid_depth(stack)
    pixel_sd = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    enough = valid.sum(axis=0) >= 2
    if np.any(enough):
        pixel_sd[enough] = np.nanstd(np.where(valid, stack, np.nan), axis=0, ddof=1)[enough]
    faces = [int(t["mesh"]["n_face"]) for t in timings]
    verts = [int(t["mesh"]["n_vert"]) for t in timings]
    meshes = [load_triangle_mesh(p / "mesh.glb") for p in runs]
    chamfers = [0.0 if i == 0 else chamfer_m(meshes[0], meshes[i]) for i in range(len(meshes))]
    f_mean, f_sd = mean_sd([float(x) for x in faces])
    v_mean, v_sd = mean_sd([float(x) for x in verts])
    ch_mean, ch_sd = mean_sd(chamfers[1:] if len(chamfers) > 1 else [0.0])
    summary = {
        "n_runs": len(runs),
        "n_view": n_view,
        "seconds": seconds,
        "per_view_seconds": per_view,
        "depth": {
            "rmse_vs_run0": rmse_vs0,
            "rmse_vs_run0_mean": float(np.nanmean(rmse_vs0)),
            "pixel_sd_mean": float(np.nanmean(pixel_sd)),
            "pixel_sd_max": float(np.nanmax(pixel_sd)) if np.any(np.isfinite(pixel_sd)) else 0.0,
        },
        "mesh": {
            "n_face": faces,
            "n_face_mean": f_mean,
            "n_face_sd": f_sd,
            "n_vert_mean": v_mean,
            "n_vert_sd": v_sd,
            "chamfer_vs_run0_m": chamfers,
            "chamfer_vs_run0_mean_m": ch_mean,
            "chamfer_vs_run0_sd_m": ch_sd,
        },
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_contact_sheet(runs, root / "runs_mesh.png", "mesh.png")
    _write_contact_sheet(runs, root / "runs_depth.png", "depth_preview.png")
    print(json.dumps({k: summary[k] for k in ("n_runs", "seconds", "depth", "mesh")}, indent=2), flush=True)
    return summary


def _write_contact_sheet(runs: list[Path], out: Path, name: str) -> None:
    """One row of per-run stills. Skips a run if ``name`` is missing."""
    thumbs: list[Image.Image] = []
    for path in runs:
        src = path / name
        if not src.is_file():
            continue
        im = Image.open(src).convert("RGB")
        im.thumbnail((240, 240), Image.Resampling.LANCZOS)
        thumbs.append(im)
    if not thumbs:
        return
    w = sum(im.size[0] for im in thumbs)
    h = max(im.size[1] for im in thumbs)
    grid = Image.new("RGB", (w, h), (14, 14, 16))
    x = 0
    for im in thumbs:
        grid.paste(im, (x, 0))
        x += im.size[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI for timed ``da3`` → SAM3D repeats on a processed scene."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input", type=Path, default=DEFAULT_SCENE,
        help="Processed scene (images/ + object/). Default: examples/gripper/processed",
    )
    parser.add_argument("--object", default="object", help="Mask folder name")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None, help="Repeat folder (default: archive/<scene>_timed)")
    parser.add_argument(
        "--keep-runs", action=argparse.BooleanOptionalAction, default=True,
        help="Keep run_*/ meshes and npzs (default). --no-keep-runs leaves summary.json and sheets only",
    )
    parser.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=True,
        help="Time and write depth_preview.png (default). --no-preview skips that step",
    )
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run ``--n`` timed jobs, then write ``summary.json``."""
    args = parse_args(argv)
    src = args.input.expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(src)
    out = (args.out if args.out is not None else default_out_dir(src)).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        da3_python = env_python("DA3_PYTHON", "SAM3D_PYTHON")
        mvsam_python = env_python("MVSAM_PYTHON", "SAM3D_PYTHON")
        if not da3_python or not mvsam_python:
            raise RuntimeError("Set SAM3D_PYTHON")
        scene = _stage(src, out / "work", args.object)
        start = 0
        existing = list(out.glob("run_*"))
        if existing:
            start = max(int(p.name.split("_")[1]) for p in existing) + 1
        for i in range(start, args.n):
            run_once(
                out / f"run_{i:02d}", scene, args.object, da3_python, mvsam_python, DEFAULT_MVSAM,
                preview=args.preview,
            )
        print("DONE all runs", flush=True)
    summarize(out)
    if not args.keep_runs:
        discard_run_dirs(out)
        print("dropped run_* and work/ (--no-keep-runs)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
