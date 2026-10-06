"""Capture aligned RealSense RGB-D dumps for ``helpers.convert_realsense_dump``.

SPACE writes ``{stem}_rgb.png``, raw ``uint16`` ``{stem}_depth.png`` (0 = no Z),
and ``{stem}.json``. Default session folder is ``input/captures_%Y%m%d_%H%M``.
Needs ``pyrealsense2`` on the capture machine; it is not in the repo ``.venv``.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
VideoMode = tuple[int, int, int]

DEFAULT_LOCK = REPO / "env" / "camera_lock.json"
CAPTURE_DIR_FORMAT = os.environ.get("RS_CAPTURE_DIR_FORMAT", "captures_%Y%m%d_%H%M")
VISUAL_PRESET = os.environ.get("RS_VISUAL_PRESET", "High Accuracy")
SETTLE_FRAMES = int(os.environ.get("RS_SETTLE_FRAMES", "30"))
RGB_EXPOSURE_STEP = float(os.environ.get("RS_EXPOSURE_STEP", "100"))
RGB_GAIN_STEP = float(os.environ.get("RS_GAIN_STEP", "16"))
RGB_BRIGHTNESS_STEP = float(os.environ.get("RS_BRIGHTNESS_STEP", "8"))
PREVIEW_MARGIN = float(os.environ.get("RS_PREVIEW_MARGIN", "0.85"))
WINDOW_NAME = "RealSense RGB-D"
METADATA_UNIT_DIVISORS: tuple[float, ...] = (10.0, 100.0, 1000.0)


def _rs() -> Any:
    """Import ``pyrealsense2`` or raise a clear missing-SDK error."""
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is not installed. Use a capture env (not this repo .venv)."
        ) from exc
    return rs


def display_limit() -> tuple[int, int] | None:
    """Return a preview size cap from env, or the screen size via Tk."""
    width = os.environ.get("RS_PREVIEW_MAX_WIDTH")
    height = os.environ.get("RS_PREVIEW_MAX_HEIGHT")
    if width and height:
        return int(width), int(height)
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = (int(root.winfo_screenwidth() * PREVIEW_MARGIN), int(root.winfo_screenheight() * PREVIEW_MARGIN))
        root.destroy()
        return size
    except (ImportError, Exception):
        return None


def raw_depth_preview(
    depth: np.ndarray, depth_scale: float, z_min_m: float, z_max_m: float,
) -> tuple[np.ndarray, float]:
    """Colorize depth on a fixed meter range. Zeros stay black; jet is not stretched per frame."""
    valid = depth > 0
    valid_fraction = float(np.mean(valid)) if depth.size else 0.0
    vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    span = z_max_m - z_min_m
    if not np.any(valid) or span <= 0:
        return vis, valid_fraction
    meters = depth.astype(np.float32) * depth_scale
    gray = np.zeros(depth.shape, dtype=np.uint8)
    gray[valid] = np.clip((meters[valid] - z_min_m) / span * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    vis[~valid] = 0
    return vis, valid_fraction


def fit_preview(image: np.ndarray) -> np.ndarray:
    """Scale a preview to the display cap, or to a single frame width."""
    height, width = image.shape[:2]
    limit = display_limit()
    max_width, max_height = limit if limit is not None else (width // 2, height)
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA,
    )


def ranked_stream_pairs(
    depth_modes: list[VideoMode], color_modes: list[VideoMode],
) -> list[tuple[VideoMode, VideoMode]]:
    """Rank RGB-D pairs, preferring the same width/height, then larger size and fps."""
    color_fps_by_size: dict[tuple[int, int], list[int]] = {}
    for width, height, fps in color_modes:
        color_fps_by_size.setdefault((width, height), []).append(fps)
    matched: list[tuple[VideoMode, VideoMode]] = []
    for width, height, depth_fps in depth_modes:
        for color_fps in color_fps_by_size.get((width, height), []):
            matched.append(((width, height, depth_fps), (width, height, color_fps)))
    candidates = matched or [(depth, color) for depth in depth_modes for color in color_modes]
    return sorted(
        candidates,
        key=lambda pair: (pair[0][0] * pair[0][1], pair[1][2], pair[0][2]),
        reverse=True,
    )


def get_opt(sensor: Any, option: Any) -> float | None:
    """Return a sensor option, or None if it is not supported."""
    if not sensor.supports(option):
        return None
    return float(sensor.get_option(option))


def set_opt(sensor: Any, option: Any, value: float) -> bool:
    """Set a sensor option if supported, scaled and clamped to the device range."""
    if not sensor.supports(option):
        return False
    option_range = sensor.get_option_range(option)
    number = float(value)
    if number > option_range.max:
        for divisor in METADATA_UNIT_DIVISORS:
            scaled = number / divisor
            if option_range.min <= scaled <= option_range.max:
                number = scaled
                break
    number = min(max(number, option_range.min), option_range.max)
    if option_range.step > 0:
        steps = round((number - option_range.min) / option_range.step)
        number = option_range.min + steps * option_range.step
        number = min(max(number, option_range.min), option_range.max)
    sensor.set_option(option, number)
    return True


def frame_meta(frame: Any, key: Any) -> float | None:
    """Return frame metadata when the firmware provides it."""
    if frame.supports_frame_metadata(key):
        return float(frame.get_frame_metadata(key))
    return None


def set_preset(sensor: Any, name: str) -> str:
    """Apply a visual preset by firmware label, if the sensor has one."""
    rs = _rs()
    if not sensor.supports(rs.option.visual_preset):
        return "unsupported"
    preset_range = sensor.get_option_range(rs.option.visual_preset)
    target = name.lower()
    for index in range(int(preset_range.min), int(preset_range.max) + 1):
        try:
            label = sensor.get_option_value_description(rs.option.visual_preset, float(index))
        except RuntimeError:
            continue
        if label and label.lower() == target:
            sensor.set_option(rs.option.visual_preset, float(index))
            return label
    return "not found"


def video_modes(sensor: Any, stream: Any, fmt: Any) -> list[VideoMode]:
    """List unique (width, height, fps) modes for a stream format, highest first."""
    modes: set[VideoMode] = set()
    for profile in sensor.get_stream_profiles():
        if profile.stream_type() != stream or profile.format() != fmt:
            continue
        video = profile.as_video_stream_profile()
        modes.add((video.width(), video.height(), video.fps()))
    return sorted(modes, key=lambda mode: (mode[0] * mode[1], mode[2]), reverse=True)


def try_enable_pair(config: Any, wrapper: Any, depth: VideoMode, color: VideoMode) -> bool:
    """Enable a depth/color pair on config if the device can resolve it."""
    rs = _rs()
    depth_w, depth_h, depth_fps = depth
    color_w, color_h, color_fps = color
    trial = rs.config()
    trial.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, depth_fps)
    trial.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, color_fps)
    try:
        trial.resolve(wrapper)
    except RuntimeError:
        return False
    config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, depth_fps)
    config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, color_fps)
    return True


def enable_best_streams(pipeline: Any, config: Any) -> dict[str, Any]:
    """Enable the highest same-size RGB-D pair this device and USB link can resolve."""
    rs = _rs()
    devices = rs.context().query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device found.")
    device = devices[0]
    depth_modes = video_modes(device.first_depth_sensor(), rs.stream.depth, rs.format.z16)
    color_modes = video_modes(device.first_color_sensor(), rs.stream.color, rs.format.bgr8)
    wrapper = rs.pipeline_wrapper(pipeline)
    for depth, color in ranked_stream_pairs(depth_modes, color_modes):
        if not try_enable_pair(config, wrapper, depth, color):
            continue
        print(f"Streams: depth {depth[0]}x{depth[1]}@{depth[2]} + color {color[0]}x{color[1]}@{color[2]}")
        return {
            "depth": {"width": depth[0], "height": depth[1], "fps": depth[2]},
            "color": {"width": color[0], "height": color[1], "fps": color[2]},
        }
    raise RuntimeError("No supported RGB-D stream pair.")


def apply_lock(color: Any, depth: Any, lock: dict[str, Any]) -> None:
    """Disable auto controls and apply saved manual RGB/depth settings."""
    rs = _rs()
    set_preset(depth, lock.get("visual_preset", VISUAL_PRESET))
    color_lock = lock.get("color", {})
    depth_lock = lock.get("depth", {})
    color_auto_off = (
        (rs.option.enable_auto_exposure, 0.0),
        (rs.option.enable_auto_white_balance, 0.0),
        (rs.option.auto_exposure_priority, 0.0),
        (rs.option.backlight_compensation, 0.0),
    )
    depth_auto_off = (
        (rs.option.enable_auto_exposure, 0.0),
        (rs.option.emitter_enabled, float(depth_lock.get("emitter_enabled", 1))),
    )
    manual = (
        ("exposure", rs.option.exposure),
        ("gain", rs.option.gain),
        ("brightness", rs.option.brightness),
        ("white_balance", rs.option.white_balance),
        ("laser_power", rs.option.laser_power),
    )
    for sensor, values, extras in ((color, color_lock, color_auto_off), (depth, depth_lock, depth_auto_off)):
        for option, value in extras:
            set_opt(sensor, option, value)
        for key, option in manual:
            if key in values and values[key] is not None:
                set_opt(sensor, option, values[key])


def wait_frames(pipeline: Any, align: Any, count: int) -> tuple[Any, Any]:
    """Return the last aligned RGB-D pair after grabbing count valid framesets."""
    color_frame: Any | None = None
    depth_frame: Any | None = None
    got = 0
    while got < count:
        aligned = align.process(pipeline.wait_for_frames())
        next_depth = aligned.get_depth_frame()
        next_color = aligned.get_color_frame()
        if next_depth and next_color:
            depth_frame, color_frame = next_depth, next_color
            got += 1
    assert color_frame is not None and depth_frame is not None
    return color_frame, depth_frame


def read_lock_values(color: Any, depth: Any, color_frame: Any, depth_frame: Any, preset: str) -> dict[str, Any]:
    """Build a lock dict from live metadata, falling back to current option values."""
    rs = _rs()
    return {
        "visual_preset": preset,
        "color": {
            "exposure": frame_meta(color_frame, rs.frame_metadata_value.actual_exposure)
            or get_opt(color, rs.option.exposure),
            "gain": frame_meta(color_frame, rs.frame_metadata_value.gain_level) or get_opt(color, rs.option.gain),
            "white_balance": frame_meta(color_frame, rs.frame_metadata_value.white_balance)
            or get_opt(color, rs.option.white_balance),
        },
        "depth": {
            "exposure": frame_meta(depth_frame, rs.frame_metadata_value.actual_exposure)
            or get_opt(depth, rs.option.exposure),
            "gain": frame_meta(depth_frame, rs.frame_metadata_value.gain_level) or get_opt(depth, rs.option.gain),
            "emitter_enabled": get_opt(depth, rs.option.emitter_enabled) or 1,
            "laser_power": get_opt(depth, rs.option.laser_power),
        },
    }


def lock_camera(
    pipeline: Any, align: Any, color: Any, depth: Any, lock_path: Path,
) -> dict[str, Any]:
    """Load a saved lock, or settle once, freeze settings, and write JSON."""
    rs = _rs()
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        apply_lock(color, depth, lock)
        print(f"Loaded camera lock: {lock_path}")
        return lock
    preset = set_preset(depth, VISUAL_PRESET)
    set_opt(depth, rs.option.emitter_enabled, 1)
    print(f"Settling auto settings ({SETTLE_FRAMES} frames)...")
    color_frame, depth_frame = wait_frames(pipeline, align, SETTLE_FRAMES)
    lock = read_lock_values(color, depth, color_frame, depth_frame, preset)
    apply_lock(color, depth, lock)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"Saved camera lock: {lock_path}")
    return lock


def nudge_color_option(
    color: Any, lock: dict[str, Any], name: str, option: Any, delta: float, lock_path: Path,
) -> float | None:
    """Change a locked RGB option, keep auto off, and persist the new value."""
    rs = _rs()
    set_opt(color, rs.option.enable_auto_exposure, 0.0)
    stored = lock.get("color", {}).get(name)
    current = stored if stored is not None else get_opt(color, option)
    if current is None or not set_opt(color, option, float(current) + delta):
        return None
    applied = get_opt(color, option)
    if applied is None:
        return None
    lock.setdefault("color", {})[name] = applied
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return applied


def depth_vis_range_m(depth_sensor: Any) -> tuple[float, float]:
    """Return an absolute colormap range in meters from env, the sensor, or the SDK colorizer."""
    rs = _rs()
    env_min = os.environ.get("RS_DEPTH_VIS_MIN_M")
    env_max = os.environ.get("RS_DEPTH_VIS_MAX_M")
    z_min = float(env_min) if env_min else get_opt(depth_sensor, rs.option.min_distance)
    z_max = float(env_max) if env_max else get_opt(depth_sensor, rs.option.max_distance)
    if z_min is None or z_max is None or z_max <= z_min:
        colorizer = rs.colorizer()
        supported = colorizer.get_supported_options()
        if z_min is None and rs.option.min_distance in supported:
            z_min = float(colorizer.get_option(rs.option.min_distance))
        if z_max is None and rs.option.max_distance in supported:
            z_max = float(colorizer.get_option(rs.option.max_distance))
    if z_min is None or z_max is None or z_max <= z_min:
        raise RuntimeError("Set RS_DEPTH_VIS_MIN_M and RS_DEPTH_VIS_MAX_M for an absolute depth scale.")
    return z_min, z_max


def capture_metadata(
    device: Any,
    usb: str,
    streams: dict[str, Any],
    depth_scale: float,
    color_frame: Any,
    depth_frame: Any,
    camera_lock: dict[str, Any],
    depth_valid_fraction: float,
    depth_vis_min_m: float,
    depth_vis_max_m: float,
) -> dict[str, Any]:
    """Build per-capture metadata including intrinsics and the active lock."""
    rs = _rs()
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    return {
        "device": device.get_info(rs.camera_info.name),
        "usb": usb,
        "streams": streams,
        "depth_scale_m_per_unit": depth_scale,
        "depth_valid_fraction": depth_valid_fraction,
        "depth_vis_min_m": depth_vis_min_m,
        "depth_vis_max_m": depth_vis_max_m,
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.ppx,
        "cy": intr.ppy,
        "depth_timestamp_ms": depth_frame.get_timestamp(),
        "color_timestamp_ms": color_frame.get_timestamp(),
        "camera_lock": camera_lock,
    }


def write_capture_dump(
    out_dir: Path, stem: str, rgb_bgr: np.ndarray, depth_u16: np.ndarray, metadata: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Write ``{stem}_rgb.png``, uint16 ``{stem}_depth.png``, and ``{stem}.json``.

    JSON must include ``fx``, ``fy``, ``cx``, ``cy``, ``depth_scale_m_per_unit``
    so ``helpers.convert_realsense_dump`` can ingest the folder.
    """
    if rgb_bgr.ndim != 3 or rgb_bgr.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3, got {rgb_bgr.shape}")
    if depth_u16.shape != rgb_bgr.shape[:2]:
        raise ValueError(f"depth {depth_u16.shape} != rgb {rgb_bgr.shape[:2]}")
    if depth_u16.dtype != np.uint16:
        raise ValueError(f"depth must be uint16, got {depth_u16.dtype}")
    missing = [key for key in ("fx", "fy", "cx", "cy", "depth_scale_m_per_unit") if key not in metadata]
    if missing:
        raise KeyError(f"metadata missing {missing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = out_dir / f"{stem}_rgb.png"
    depth_path = out_dir / f"{stem}_depth.png"
    json_path = out_dir / f"{stem}.json"
    if not cv2.imwrite(str(rgb_path), rgb_bgr):
        raise RuntimeError(f"failed to write {rgb_path}")
    if not cv2.imwrite(str(depth_path), depth_u16):
        raise RuntimeError(f"failed to write {depth_path}")
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return rgb_path, depth_path, json_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI: optional dump root and lock path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Session folder (default: input/captures_YYYYMMDD_HHMM)",
    )
    parser.add_argument(
        "--lock", type=Path, default=Path(os.environ.get("RS_LOCK_PATH", str(DEFAULT_LOCK))),
        help="camera_lock.json (default: env/camera_lock.json)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Write a synthetic dump and ingest it (no camera / no pyrealsense2)",
    )
    return parser.parse_args(argv)


def self_test_dump(out_dir: Path) -> tuple[Path, Path, Path]:
    """Write one synthetic RGB-D dump and ingest it. No RealSense hardware.

    Proves SPACE output is what ``helpers.convert_realsense_dump`` reads.
    """
    from helpers import convert_realsense_dump

    rgb_bgr = np.zeros((8, 12, 3), dtype=np.uint8)
    rgb_bgr[3, 5] = (10, 20, 30)
    depth_u16 = np.zeros((8, 12), dtype=np.uint16)
    depth_u16[3, 5] = 1500
    metadata = {
        "fx": 10.0,
        "fy": 11.0,
        "cx": 6.0,
        "cy": 4.0,
        "depth_scale_m_per_unit": 0.001,
        "device": "self-test",
    }
    raw_dir = out_dir / "raw"
    rgb_path, depth_path, json_path = write_capture_dump(raw_dir, "frame0", rgb_bgr, depth_u16, metadata)
    frames, depths, ks, stems = convert_realsense_dump(
        raw_dir, out_dir / "images",
        rgb_suffix="_rgb.png", depth_suffix="_depth.png", meta_suffix=".json",
    )
    if stems != ["frame0"]:
        raise RuntimeError(f"self-test stems {stems} != ['frame0']")
    if abs(float(ks[0, 0, 0]) - 10.0) > 1e-9:
        raise RuntimeError(f"self-test fx {ks[0, 0, 0]} != 10")
    if abs(float(depths[0, 3, 5]) - 1.5) > 1e-6:
        raise RuntimeError(f"self-test depth {depths[0, 3, 5]} != 1.5 m")
    if not frames[0].is_file():
        raise RuntimeError(f"self-test missing ingested RGB {frames[0]}")
    print(f"self-test dump ok → {raw_dir} ingested {frames[0]}", flush=True)
    return rgb_path, depth_path, json_path


def run_capture(out_dir: Path, lock_path: Path) -> int:
    """Stream aligned RGB-D, lock camera settings, and save captures on SPACE."""
    rs = _rs()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_dir}")
    pipeline = rs.pipeline()
    config = rs.config()
    streams = enable_best_streams(pipeline, config)
    profile = pipeline.start(config)
    device = profile.get_device()
    usb = (
        device.get_info(rs.camera_info.usb_type_descriptor)
        if device.supports(rs.camera_info.usb_type_descriptor)
        else "unknown"
    )
    print(f"Device: {device.get_info(rs.camera_info.name)}  USB: {usb}")
    align = rs.align(rs.stream.color)
    depth_sensor = device.first_depth_sensor()
    color_sensor = device.first_color_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    z_min_m, z_max_m = depth_vis_range_m(depth_sensor)
    print(f"Depth colormap: {z_min_m:.2f}–{z_max_m:.2f} m (absolute jet)")
    camera_lock = lock_camera(pipeline, align, color_sensor, depth_sensor, lock_path)
    print(json.dumps(camera_lock, indent=2))
    print("SPACE=capture  1/2 exposure  3/4 brightness  9/0 gain  Q=quit")
    rgb_nudge = {
        ord("1"): ("exposure", rs.option.exposure, -RGB_EXPOSURE_STEP),
        ord("2"): ("exposure", rs.option.exposure, RGB_EXPOSURE_STEP),
        ord("3"): ("brightness", rs.option.brightness, -RGB_BRIGHTNESS_STEP),
        ord("4"): ("brightness", rs.option.brightness, RGB_BRIGHTNESS_STEP),
        ord("9"): ("gain", rs.option.gain, -RGB_GAIN_STEP),
        ord("0"): ("gain", rs.option.gain, RGB_GAIN_STEP),
    }
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            color_frame, depth_frame = wait_frames(pipeline, align, 1)
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            if color.shape[:2] != depth.shape[:2]:
                raise RuntimeError(f"RGB/depth size mismatch: {color.shape[:2]} vs {depth.shape[:2]}")
            depth_vis, valid_fraction = raw_depth_preview(depth, depth_scale, z_min_m, z_max_m)
            preview = fit_preview(np.hstack((color, depth_vis)))
            color_lock = camera_lock.get("color", {})
            label = (
                f"SPACE | 1/2 exp | 3/4 bright | 9/0 gain | Q"
                f" | valid={valid_fraction:.0%} | jet {z_min_m:.1f}-{z_max_m:.1f}m"
            )
            for name in ("exposure", "brightness", "gain"):
                value = color_lock.get(name)
                if value is not None:
                    label += f" | {name[:3]}={value:.0f}"
            cv2.putText(preview, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in rgb_nudge:
                name, option, delta = rgb_nudge[key]
                applied = nudge_color_option(color_sensor, camera_lock, name, option, delta, lock_path)
                if applied is not None:
                    print(f"RGB {name}: {applied:.0f}")
                continue
            if key != 32:
                continue
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            metadata = capture_metadata(
                device, usb, streams, depth_scale, color_frame, depth_frame, camera_lock,
                valid_fraction, z_min_m, z_max_m,
            )
            write_capture_dump(out_dir, timestamp, color, depth.astype(np.uint16, copy=False), metadata)
            print(f"Captured: {timestamp}  depth valid={valid_fraction:.0%}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse args and run the capture loop, or ``--self-test`` without a camera."""
    args = parse_args(argv)
    out = args.out
    if out is None:
        out = REPO / "input" / datetime.now().strftime(CAPTURE_DIR_FORMAT)
    out = out.expanduser().resolve()
    if args.self_test:
        self_test_dump(out)
        return 0
    return run_capture(out, args.lock.expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
