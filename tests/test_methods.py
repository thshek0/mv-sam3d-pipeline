"""RS+DA3 combine: enclosed wells stay empty; fill_wells plugs them."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods import classify_miss, combine_rs_da3, resize_depth  # noqa: E402


def _lid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """8x8 object: lid Z=1, 2x2 well, right-edge wall miss, DA3 plate + wall 1.3."""
    obj = np.zeros((8, 8), dtype=bool)
    obj[1:7, 1:7] = True
    rs = np.zeros((8, 8), dtype=np.float32)
    rs[obj] = 1.0
    rs[2:4, 2:4] = 0.0
    rs[1:7, 5:7] = 0.0
    da3 = np.zeros((8, 8), dtype=np.float32)
    da3[obj] = 1.0
    da3[1:7, 5:7] = 1.3
    return obj, rs, da3


def test_classify_miss_splits_well_and_wall() -> None:
    """An enclosed hole is a well; a miss on the object outline is a wall."""
    obj, rs, _da3 = _lid()
    well, wall = classify_miss(obj, rs)
    assert bool(well[2, 2]) and bool(well[3, 3])
    assert not bool(well[1, 5])
    assert bool(wall[1, 5]) and bool(wall[5, 6])
    assert not bool(wall[2, 2])


def test_combine_keeps_well_zero_and_fills_wall() -> None:
    """Default combine leaves the well at 0 and copies DA3 onto the wall."""
    obj, rs, da3 = _lid()
    depth, well, wall, scale = combine_rs_da3(rs, da3, obj, fill_wells=False)
    assert abs(scale - 1.0) < 1e-5
    assert float(depth[2, 2]) == 0.0
    assert bool(well[2, 2])
    assert abs(float(depth[1, 2]) - 1.0) < 1e-5
    assert abs(float(depth[3, 6]) - 1.3) < 1e-5
    assert bool(wall[3, 6])


def test_combine_fill_wells_plugs_the_hole() -> None:
    """fill_wells copies DA3 into the well (old fill-all-holes)."""
    obj, rs, da3 = _lid()
    depth, well, _wall, _scale = combine_rs_da3(rs, da3, obj, fill_wells=True)
    assert float(depth[2, 2]) == 1.0
    assert not bool(np.any(well))


def test_resize_depth_nearest() -> None:
    """A 2x2 depth map scales to 4x4 without mixing zeros into valid pixels."""
    src = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
    out = resize_depth(src, 4, 4)
    assert out.shape == (4, 4)
    assert float(out[0, 0]) == 1.0
    assert float(out[0, 3]) == 2.0
    assert float(out[3, 0]) == 0.0
    assert float(out[3, 3]) == 3.0
