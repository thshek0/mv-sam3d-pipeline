"""Capability matrix and sequential ``run_dataset`` steps."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from helpers import listed_images, write_dataset_summary
from methods import depth_npz_is_unposed
from run_dataset import (
    build_hybrid,
    build_mesh,
    build_rs,
    ingest,
    layout_for,
    mesh_jobs_for,
    summarize,
    validate_mesh_job,
)
from vis import write_da3_npz


def test_mesh_jobs_for_modes() -> None:
    """Allowed depth/mesh pairs follow the RS / tags capability matrix."""
    assert set(mesh_jobs_for(use_rs=False, tags=False)) == {("da3", "sam3d"), ("da3", "tsdf")}
    assert ("hybrid", "sam3d") in mesh_jobs_for(use_rs=True, tags=False)
    assert ("rs", "tsdf") not in mesh_jobs_for(use_rs=True, tags=False)
    assert ("hybrid", "tsdf") not in mesh_jobs_for(use_rs=True, tags=False)
    tagged = set(mesh_jobs_for(use_rs=True, tags=True))
    assert ("rs", "tsdf") in tagged
    assert ("hybrid", "tsdf") in tagged
    assert ("da3_posed", "sam3d") in tagged


def test_validate_mesh_job_rejects_unposed_rs_tsdf() -> None:
    """Unposed RS + TSDF is illegal."""
    try:
        validate_mesh_job("rs", "tsdf", use_rs=True, tags=False)
    except RuntimeError as exc:
        assert "illegal" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError")


def _crop_png_b64(arr: np.ndarray) -> str:
    """Encode a uint8 crop as a LabelMe-style base64 PNG."""
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _write_tiny_dump(dump: Path, *, n_view: int = 2) -> None:
    """Write ``n_view`` RGB/depth/meta/LabelMe files under ``dump``."""
    dump.mkdir(parents=True)
    crop = np.ones((4, 4), dtype=np.uint8)
    for i in range(n_view):
        stem = f"20260101_00000{i}_000000"
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[2:6, 2:6] = (40, 80, 120)
        Image.fromarray(rgb, mode="RGB").save(dump / f"{stem}_rgb.png")
        depth = np.zeros((8, 8), dtype=np.uint16)
        depth[2:6, 2:6] = 1000
        Image.fromarray(depth).save(dump / f"{stem}_depth.png")
        meta = {
            "fx": 4.0, "fy": 4.0, "cx": 3.5, "cy": 3.5,
            "depth_scale_m_per_unit": 0.001, "width": 8, "height": 8,
        }
        (dump / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")
        shapes = {
            "version": "5.0.1", "flags": {}, "imagePath": f"{stem}_rgb.png",
            "imageHeight": 8, "imageWidth": 8,
            "shapes": [{
                "label": "obj", "shape_type": "mask",
                "points": [[2, 2], [5, 5]], "mask": _crop_png_b64(crop),
            }],
        }
        (dump / f"{stem}_rgb.json").write_text(json.dumps(shapes), encoding="utf-8")


def test_ingest_unposed_and_dataset_summary(tmp_path: Path) -> None:
    """Ingest writes dataset_summary; unposed RS refuses TSDF."""
    dump = tmp_path / "dataset" / "toy"
    out_root = tmp_path / "output"
    _write_tiny_dump(dump)
    layout = layout_for("toy", dump=dump, out_root=out_root, mvsam_root=tmp_path / "mvsam")

    imgs = ingest(layout)
    assert len(imgs) == 2
    assert (layout.root / "dataset_summary.png").is_file()
    assert (layout.root / "masks_preview.png").is_file()

    rs_npz = build_rs(layout, tags=False, process_res=8, margin=1.0)
    assert depth_npz_is_unposed(rs_npz)
    z = np.asarray(np.load(rs_npz)["depth"], dtype=np.float32)
    rs = np.load(rs_npz)
    write_da3_npz(
        layout.depth_npz("da3"), z, rs["extrinsics"], rs["intrinsics"],
        listed_images(layout.images), process_res=int(z.shape[-1]),
    )
    hybrid = build_hybrid(layout)
    assert depth_npz_is_unposed(hybrid)

    try:
        build_mesh(layout, "rs", "tsdf", refresh_summary=False)
    except RuntimeError as exc:
        assert "unposed" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError for unposed rs TSDF")

    summarize(layout)
    assert (layout.root / "depth_summary.png").is_file()
    write_dataset_summary(listed_images(layout.images), layout.root / "dataset_summary2.png")
    assert (layout.root / "dataset_summary2.png").is_file()
