"""Ingest, masks, crop, and DA3 / MV-SAM3D subprocesses."""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO = Path(__file__).resolve().parent
DEFAULT_MVSAM = REPO / "MV-SAM3D"
DEFAULT_WORK = REPO / "work"
DEFAULT_OUT = REPO / "output"


def env_python(*names: str) -> str | None:
    """Return the first non-empty environment variable in ``names``."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess and raise on failure."""
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


STILL_SUFFIXES = {".png", ".jpg", ".jpeg"}


def still_image_paths(src_dir: Path) -> list[Path]:
    """Return stills in the same order as ``convert_stills_to_images`` (name sort)."""
    return sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in STILL_SUFFIXES
    )


def open_still_rgb(path: Path) -> Image.Image:
    """Open a still as RGB in display orientation (EXIF applied).

    Phone PNGs/JPEGs often store landscape pixels with ``Orientation`` 6/8.
    LabelMe annotates the upright view; ingest must use the same frame or the
    mask is stretched onto swapped axes.
    """
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _aspect(size: tuple[int, int]) -> float:
    """Return width/height, treating a degenerate size as 1."""
    return size[0] / float(max(size[1], 1))


def resize_mask_to_rgb(mask: Image.Image, rgb_size: tuple[int, int]) -> Image.Image:
    """Scale a full-frame LabelMe mask to the ingested RGB size.

    Raises if the mask is the transpose of the RGB (EXIF not applied on ingest).
    """
    if mask.size == rgb_size:
        return mask
    mask_aspect = _aspect(mask.size)
    rgb_aspect = _aspect(rgb_size)
    swapped_aspect = _aspect((rgb_size[1], rgb_size[0]))
    if abs(mask_aspect - swapped_aspect) + 1e-6 < abs(mask_aspect - rgb_aspect):
        raise ValueError(
            f"LabelMe mask {mask.size[0]}x{mask.size[1]} is transposed vs RGB "
            f"{rgb_size[0]}x{rgb_size[1]}. Apply EXIF orientation when ingesting stills."
        )
    return mask.resize(rgb_size, Image.Resampling.NEAREST)


def labelme_json_for_stills(src_dir: Path) -> list[Path] | None:
    """Return a JSON path per still, or None if the folder has no LabelMe files.

    If any still has a sidecar ``.json``, every still must have one. Mixed
    LabelMe / rembg in the same folder is not a mode.
    """
    jsons = [src.with_suffix(".json") for src in still_image_paths(src_dir)]
    present = [path for path in jsons if path.is_file()]
    if not present:
        return None
    missing = [path.name for path in jsons if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"LabelMe JSON missing for stills: {', '.join(missing)}")
    return jsons


def load_labelme(json_path: Path) -> dict[str, Any]:
    """Load a LabelMe annotation document."""
    return json.loads(json_path.read_text())


def canvas_size(data: dict[str, Any], json_path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` for the LabelMe canvas."""
    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    if width > 0 and height > 0:
        return width, height
    sidecar = json_path.parent / str(data.get("imagePath") or json_path.with_suffix(".png").name)
    if sidecar.is_file():
        return open_still_rgb(sidecar).size
    raise ValueError(f"{json_path} has no imageWidth/imageHeight and no sidecar still")


def decode_mask_crop(shape: dict[str, Any]) -> Image.Image:
    """Decode the base64 PNG crop stored on a LabelMe mask shape."""
    raw = shape.get("mask")
    if not raw:
        raise ValueError("mask shape is missing the base64 crop")
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("L")


def paste_mask_crop(canvas: np.ndarray, crop: Image.Image, points: Sequence[Sequence[float]]) -> None:
    """Paste a LabelMe mask crop into ``canvas`` using an inclusive bbox.

    Args:
        canvas: Full-frame ``uint8`` mask ``(H, W)``.
        crop: LabelMe PNG crop (pixels ``0/1`` or ``0/255``).
        points: ``[[x1, y1], [x2, y2]]`` inclusive bbox in image pixels.
    """
    if len(points) != 2:
        raise ValueError(f"mask shape needs a 2-point bbox, got {len(points)} points")
    height, width = canvas.shape
    x1, y1 = int(round(float(points[0][0]))), int(round(float(points[0][1])))
    x2, y2 = int(round(float(points[1][0]))), int(round(float(points[1][1])))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1
    piece = crop if crop.size == (box_w, box_h) else crop.resize((box_w, box_h), Image.Resampling.NEAREST)
    arr = np.asarray(piece)
    dst_x0, dst_y0 = max(0, x1), max(0, y1)
    dst_x1, dst_y1 = min(width, x1 + box_w), min(height, y1 + box_h)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    src_x0, src_y0 = dst_x0 - x1, dst_y0 - y1
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)
    fg = np.where(arr[src_y0:src_y1, src_x0:src_x1] > 0, np.uint8(255), np.uint8(0))
    region = canvas[dst_y0:dst_y1, dst_x0:dst_x1]
    np.maximum(region, fg, out=region)


def mask_from_labelme(data: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    """OR every ``shape_type=mask`` into one full-frame L image (0/255)."""
    width, height = size
    canvas = np.zeros((height, width), dtype=np.uint8)
    n_mask = 0
    for shape in data.get("shapes") or []:
        if shape.get("shape_type") != "mask":
            continue
        paste_mask_crop(canvas, decode_mask_crop(shape), shape.get("points") or [])
        n_mask += 1
    if n_mask == 0:
        raise ValueError("LabelMe JSON has no shape_type=mask annotations")
    return Image.fromarray(canvas, mode="L")


def rgba_from_labelme(json_path: Path, rgb: Image.Image) -> Image.Image:
    """Build an RGBA still whose alpha is the LabelMe mask, sized to ``rgb``."""
    data = load_labelme(json_path)
    mask = mask_from_labelme(data, canvas_size(data, json_path))
    mask = resize_mask_to_rgb(mask, rgb.size)
    out = rgb.convert("RGB").copy()
    out.putalpha(mask)
    return out


def write_labelme_masks(src_dir: Path, images: Sequence[Path], mask_dir: Path) -> list[Path]:
    """Write RGBA masks that pair 1:1 with ingested ``0.png``, ``1.png``, …

    Each ingested frame is the i-th still in ``src_dir``. The JSON is
    ``still.with_suffix('.json')``.
    """
    srcs = still_image_paths(src_dir)
    if len(srcs) != len(images):
        raise RuntimeError(f"still count {len(srcs)} != ingested count {len(images)}")
    mask_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for dest, src in zip(images, srcs):
        json_path = src.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(f"LabelMe JSON missing for {src.name}: {json_path}")
        rgb = Image.open(dest).convert("RGB")
        rgba = rgba_from_labelme(json_path, rgb)
        alpha = np.asarray(rgba.getchannel("A"))
        n_fg = int(np.count_nonzero(alpha))
        out = mask_dir / dest.name
        rgba.save(out)
        written.append(out)
        frac = n_fg / float(alpha.size)
        print(
            f"labelme {src.name} -> {out.name} {rgba.size[0]}x{rgba.size[1]} "
            f"fg={n_fg} ({100.0 * frac:.1f}%)",
            flush=True,
        )
    print(f"Wrote {len(written)} LabelMe RGBA masks → {mask_dir}", flush=True)
    return written


def _fit_thumb(im: Image.Image, thumb: int) -> Image.Image:
    """Return a copy fitted into a ``thumb x thumb`` RGB cell on dark gray."""
    cell = Image.new("RGB", (thumb, thumb), (22, 22, 24))
    work = im.convert("RGB")
    work.thumbnail((thumb - 8, thumb - 28), Image.Resampling.LANCZOS)
    x = (thumb - work.size[0]) // 2
    y = 8
    cell.paste(work, (x, y))
    return cell


def composite_rgba(rgba: Image.Image) -> Image.Image:
    """Paste an RGBA cutout onto a dark background."""
    bg = Image.new("RGB", rgba.size, (36, 36, 40))
    work = rgba.convert("RGBA")
    bg.paste(work, mask=work.split()[-1])
    return bg


def save_mask_preview(
    images: Sequence[Path],
    masks: Sequence[Path],
    out_png: Path,
    copies_dir: Path | None = None,
    thumb: int = 280,
) -> Path:
    """Write a photo | cutout grid and optional per-view RGBA copies."""
    if len(images) != len(masks):
        raise ValueError("images and masks must be the same length")
    n = len(images)
    cols = min(3, max(1, n))
    rows = (n + cols - 1) // cols
    cell_w, cell_h = thumb * 2, thumb
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (14, 14, 16))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    if copies_dir is not None:
        copies_dir.mkdir(parents=True, exist_ok=True)
    for i, (rgb_path, mask_path) in enumerate(zip(images, masks)):
        r, c = divmod(i, cols)
        rgb = Image.open(rgb_path).convert("RGB")
        rgba = Image.open(mask_path).convert("RGBA")
        if copies_dir is not None:
            shutil.copy2(mask_path, copies_dir / mask_path.name)
        left = _fit_thumb(rgb, thumb)
        right = _fit_thumb(composite_rgba(rgba), thumb)
        ImageDraw.Draw(left).text((8, thumb - 18), f"{i} photo", fill=(220, 220, 220), font=font)
        ImageDraw.Draw(right).text((8, thumb - 18), f"{i} labelme", fill=(220, 220, 220), font=font)
        x0, y0 = c * cell_w, r * cell_h
        grid.paste(left, (x0, y0))
        grid.paste(right, (x0 + thumb, y0))
        draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), outline=(60, 60, 64))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_png)
    print(f"labelme preview → {out_png}", flush=True)
    return out_png

# --- crop ---


def image_center_square(width: int, height: int) -> tuple[int, int, int]:
    """Return ``(cx, cy, side)`` for a square at the image center.

    ``side`` is ``min(width, height)`` so the crop stays inside the frame.
    """
    if width < 1 or height < 1:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    return width // 2, height // 2, max(min(width, height), 2)


def mask_square(mask: np.ndarray, margin: float = 1.15) -> tuple[int, int, int]:
    """Return ``(cx, cy, side)`` for a square around the foreground bbox.

    ``cx, cy`` are pixel centers. ``side`` may extend past the image; the caller pads.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be HxW, got {mask.shape}")
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("mask is empty")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = int(np.ceil(max(x1 - x0 + 1, y1 - y0 + 1) * margin))
    side = max(side, 2)
    return int(round(cx)), int(round(cy)), side


def crop_square(image: Image.Image, cx: int, cy: int, side: int, fill: tuple[int, ...] | int) -> Image.Image:
    """Crop a ``side x side`` window at ``(cx, cy)``, padding if it hangs off the frame."""
    width, height = image.size
    x0 = int(cx - side // 2)
    y0 = int(cy - side // 2)
    x1, y1 = x0 + side, y0 + side
    canvas = Image.new(image.mode, (side, side), fill)
    src_x0, src_y0 = max(0, x0), max(0, y0)
    src_x1, src_y1 = min(width, x1), min(height, y1)
    dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return canvas
    patch = image.crop((src_x0, src_y0, src_x1, src_y1))
    canvas.paste(patch, (dst_x0, dst_y0))
    return canvas


def _mask_alpha(mask_img: Image.Image) -> np.ndarray:
    """Return a 2-D uint8/bool-ready alpha (or L) array from a mask image."""
    work = mask_img.convert("RGBA") if mask_img.mode in {"RGBA", "LA"} else mask_img.convert("L")
    return np.asarray(work.getchannel("A") if work.mode == "RGBA" else work)


def crop_pair_to_mask(
    rgb: Image.Image, mask_img: Image.Image | None, margin: float = 1.15
) -> tuple[Image.Image, Image.Image | None]:
    """Crop RGB and optional mask with one square.

    Uses the mask bbox when the mask has foreground; otherwise an image-center square.
    """
    width, height = rgb.size
    alpha = _mask_alpha(mask_img) if mask_img is not None else None
    if alpha is not None and np.any(alpha > 0):
        cx, cy, side = mask_square(alpha > 0, margin=margin)
    else:
        cx, cy, side = image_center_square(width, height)
    rgb_out = crop_square(rgb.convert("RGB"), cx, cy, side, fill=(0, 0, 0))
    if mask_img is None:
        return rgb_out, None
    if mask_img.mode == "RGBA":
        mask_out = crop_square(mask_img.convert("RGBA"), cx, cy, side, fill=(0, 0, 0, 0))
    else:
        mask_out = crop_square(mask_img.convert("L"), cx, cy, side, fill=0)
    return rgb_out, mask_out


def crop_views_to_masks(images: Sequence[Path], mask_dir: Path, margin: float = 1.15) -> list[Path]:
    """Overwrite ingested RGB and RGBA files with squares.

    Mask-centered when a non-empty mask exists; image-centered otherwise.
    """
    written: list[Path] = []
    for rgb_path in images:
        mask_path = mask_dir / rgb_path.name
        mask_img = Image.open(mask_path) if mask_path.is_file() else None
        used_mask = mask_img is not None and np.any(_mask_alpha(mask_img) > 0)
        rgb, mask = crop_pair_to_mask(Image.open(rgb_path), mask_img, margin=margin)
        rgb.save(rgb_path)
        if mask is not None and mask_path.is_file():
            mask.save(mask_path)
        written.append(rgb_path)
        kind = "mask-center" if used_mask else "image-center"
        print(f"crop {kind} {rgb_path.name} -> {rgb.size[0]}x{rgb.size[1]}", flush=True)
    print(f"Cropped {len(written)} views to squares", flush=True)
    return written

def convert_video_to_frames(
    video: Path, images_dir: Path, *, fps: float, max_frames: int | None = None, ffmpeg: str = "ffmpeg",
) -> list[Path]:
    """Sample a video to ``images_dir`` as ``0.png``, ``1.png``, …"""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    images_dir.mkdir(parents=True, exist_ok=True)
    for old in images_dir.glob("*"):
        old.unlink()
    tmp_pattern = images_dir / "frame_%05d.png"
    run([ffmpeg, "-y", "-i", str(video), "-vf", f"fps={fps}", str(tmp_pattern)])
    frames = sorted(images_dir.glob("frame_*.png"))
    if max_frames is not None and max_frames > 0 and len(frames) > max_frames:
        idxs = [round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
        keep_set = {frames[i] for i in idxs}
        for path in frames:
            if path not in keep_set:
                path.unlink()
        frames = sorted(keep_set, key=lambda p: p.name)
    named: list[Path] = []
    for i, path in enumerate(frames):
        dest = images_dir / f"{i}.png"
        path.rename(dest)
        named.append(dest)
    if not named:
        raise RuntimeError(f"No frames extracted from {video}")
    print(f"Extracted {len(named)} frames at {fps} fps → {images_dir}", flush=True)
    return named


def extract_frames(ffmpeg: str, video: Path, images_dir: Path, fps: float, max_frames: int | None = None) -> list[Path]:
    """Alias of ``convert_video_to_frames`` (positional ffmpeg first)."""
    return convert_video_to_frames(video, images_dir, fps=fps, max_frames=max_frames, ffmpeg=ffmpeg)


def convert_stills_to_images(src_dir: Path, images_dir: Path, max_side: int = 1920) -> list[Path]:
    """Copy stills as ``0.png``, ``1.png``, … with EXIF applied and long edge capped."""
    srcs = still_image_paths(src_dir)
    if not srcs:
        raise RuntimeError(f"No stills in {src_dir}")
    images_dir.mkdir(parents=True, exist_ok=True)
    for old in images_dir.glob("*"):
        old.unlink()
    named: list[Path] = []
    for i, src in enumerate(srcs):
        im = open_still_rgb(src)
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        dest = images_dir / f"{i}.png"
        im.save(dest)
        named.append(dest)
        print(f"still {src.name} {w}x{h} -> {dest.name} {im.size[0]}x{im.size[1]}", flush=True)
    print(f"Ingested {len(named)} stills → {images_dir}", flush=True)
    return named


def ingest_stills(src_dir: Path, images_dir: Path, max_side: int = 1920) -> list[Path]:
    """Alias of ``convert_stills_to_images``."""
    return convert_stills_to_images(src_dir, images_dir, max_side=max_side)


def rembg_masks(rembg_python: str, images: list[Path], mask_dir: Path) -> None:
    """Write RGBA masks with rembg; alpha = foreground."""
    mask_dir.mkdir(parents=True, exist_ok=True)
    script = r"""
import sys
from pathlib import Path
from rembg import remove, new_session
from PIL import Image

images = [Path(p) for p in sys.argv[1:-1]]
mask_dir = Path(sys.argv[-1])
session = new_session("u2net")
for src in images:
    out = mask_dir / src.name
    img = Image.open(src).convert("RGB")
    cut = remove(img, session=session)
    cut.save(out)
    print(f"mask {src.name} -> {out}", flush=True)
"""
    run([rembg_python, "-c", script, *[str(p) for p in images], str(mask_dir)])


def write_scene_masks(
    src: Path, images: list[Path], mask_dir: Path, rembg_python: str | None, skip_rembg: bool, preview_png: Path,
) -> str:
    """Write RGBA masks: LabelMe JSON if present, otherwise rembg."""
    if src.is_dir() and labelme_json_for_stills(src) is not None:
        masks = write_labelme_masks(src, images, mask_dir)
        save_mask_preview(images, masks, preview_png)
        print("masks=labelme", flush=True)
        return "labelme"
    if skip_rembg:
        kind = "reuse"
    else:
        if not rembg_python:
            raise RuntimeError("Set REMBG_PYTHON (no LabelMe JSON)")
        rembg_masks(rembg_python, images, mask_dir)
        kind = "rembg"
    masks = [mask_dir / img.name for img in images]
    if all(m.is_file() for m in masks):
        save_mask_preview(images, masks, preview_png)
    print(f"masks={kind}", flush=True)
    return kind


def run_da3(da3_python: str, mvsam_root: Path, images_dir: Path, da3_out: Path) -> Path:
    """Run Depth Anything 3 via MV-SAM3D's runner."""
    da3_out.mkdir(parents=True, exist_ok=True)
    runner = mvsam_root / "scripts" / "run_da3.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Missing {runner}")
    run(
        [da3_python, str(runner), "--image_dir", str(images_dir), "--output_dir", str(da3_out), "--no_vis"],
        cwd=mvsam_root,
    )
    npz = da3_out / "da3_output.npz"
    if not npz.is_file():
        raise FileNotFoundError(f"DA3 did not write {npz}")
    return npz


def run_mvsam(
    mvsam_python: str, mvsam_root: Path, scene_dir: Path, object_name: str, da3_npz: Path, merge_da3_glb: bool,
) -> Path:
    """Run weighted multi-view SAM 3D; return path to result.glb."""
    infer = mvsam_root / "run_inference_weighted.py"
    if not infer.is_file():
        raise FileNotFoundError(f"Missing {infer}")
    cmd = [
        mvsam_python, str(infer),
        "--input_path", str(scene_dir),
        "--mask_prompt", object_name,
        "--da3_output", str(da3_npz),
    ]
    if merge_da3_glb:
        cmd.append("--merge_da3_glb")
    run(cmd, cwd=mvsam_root)
    hits = sorted((mvsam_root / "visualization").rglob("result.glb"), key=lambda p: p.stat().st_mtime)
    if not hits:
        hits = sorted(scene_dir.rglob("result.glb"), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise FileNotFoundError("MV-SAM3D finished but result.glb was not found")
    return hits[-1]


def convert_labelme_json_to_rgba(json_path: Path, rgb: Image.Image) -> Image.Image:
    """Alias of ``rgba_from_labelme`` (one JSON + one RGB → one RGBA)."""
    return rgba_from_labelme(json_path, rgb)


def convert_labelme_to_masks(src_dir: Path, images: Sequence[Path], mask_dir: Path) -> list[Path]:
    """Write RGBA masks that pair 1:1 with ingested frames (folder API)."""
    return write_labelme_masks(src_dir, images, mask_dir)


def convert_rembg_to_masks(images: Sequence[Path], mask_dir: Path, rembg_python: str) -> list[Path]:
    """Write RGBA masks with rembg in another env. Alpha is foreground."""
    if not rembg_python:
        raise RuntimeError("Pass rembg_python (or set REMBG_PYTHON)")
    rembg_masks(rembg_python, list(images), mask_dir)
    written = [mask_dir / path.name for path in images]
    missing = [path.name for path in written if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"rembg did not write: {', '.join(missing)}")
    return written


def listed_images(images_dir: Path) -> list[Path]:
    """Return ``*.png`` in numeric-stem order."""
    return sorted(
        images_dir.glob("*.png"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )


def validate_scene(scene_dir: Path, object_name: str) -> list[Path]:
    """Check ``images/{i}.png`` and ``{object}/{i}.png`` match (RGBA, some alpha)."""
    images = listed_images(scene_dir / "images")
    if not images:
        raise RuntimeError(f"No images in {scene_dir / 'images'}")
    mask_dir = scene_dir / object_name
    for image in images:
        mask = mask_dir / image.name
        if not mask.is_file():
            raise FileNotFoundError(f"Missing mask {mask}")
        rgba = Image.open(mask)
        if rgba.mode != "RGBA":
            raise ValueError(f"{mask} must be RGBA, got {rgba.mode}")
        alpha = np.asarray(rgba.getchannel("A"))
        if not np.any(alpha > 0):
            raise ValueError(f"{mask} has empty alpha")
    return images

