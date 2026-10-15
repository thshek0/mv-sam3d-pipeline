"""Hybrid fuse: drop flying pixels, RANSAC scale, reject DA in free space."""

from __future__ import annotations

import numpy as np

from methods import drop_flying_pixels, fuse_hybrid, ransac_scale


def _cameras(n_view: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Identity w2c and a shared pinhole K for an 8x8 view."""
    k_mat = np.array([[8.0, 0.0, 3.5], [0.0, 8.0, 3.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    ext = np.zeros((n_view, 3, 4), dtype=np.float64)
    ext[:, 0, 0] = 1.0
    ext[:, 1, 1] = 1.0
    ext[:, 2, 2] = 1.0
    return np.stack([k_mat] * n_view), ext


def test_drop_flying_pixels_zeros_a_spike() -> None:
    """A 20 cm spike on a flat plane is removed; the plane stays."""
    depth = np.ones((5, 5), dtype=np.float32)
    depth[2, 2] = 1.2
    out = drop_flying_pixels(depth, jump_m=0.01)
    assert float(out[2, 2]) == 0.0
    assert float(out[0, 0]) == 1.0
    assert float(out[2, 1]) == 1.0


def test_ransac_scale_ignores_one_outlier() -> None:
    """A single bad ratio does not move ``a`` off the inlier scale."""
    rs = np.full((10, 10), 2.0, dtype=np.float32)
    da = np.ones((10, 10), dtype=np.float32)
    rs[0, 0] = 50.0
    scale, inlier_frac = ransac_scale(rs, da, np.ones((10, 10), dtype=bool), thresh_m=0.01)
    assert abs(scale - 2.0) < 1e-6
    assert inlier_frac < 1.0
    assert inlier_frac > 0.9


def test_fuse_hybrid_rejects_da_in_front_of_other_view() -> None:
    """A closer DA sample on an RS hole is dropped when the other view saw past it."""
    ks, ext = _cameras()
    rs = np.full((2, 8, 8), 2.0, dtype=np.float32)
    da = np.ones((2, 8, 8), dtype=np.float32)
    rs[0, 3, 3] = 0.0
    da[0, 3, 3] = 0.5
    obj = np.ones((2, 8, 8), dtype=bool)
    fused, scales = fuse_hybrid(rs, da, obj, ks, ext, jump_m=0.05)
    assert abs(float(scales[0]) - 2.0) < 1e-6
    assert float(fused[0, 3, 3]) == 0.0
    assert abs(float(fused[0, 3, 4]) - 2.0) < 1e-5


def test_fuse_hybrid_keeps_da_that_meets_the_other_view() -> None:
    """DA on a hole is kept when its depth lands on the other view's surface."""
    ks, ext = _cameras()
    rs = np.full((2, 8, 8), 2.0, dtype=np.float32)
    da = np.ones((2, 8, 8), dtype=np.float32)
    rs[0, 3, 3] = 0.0
    obj = np.ones((2, 8, 8), dtype=bool)
    obj[:, 0, 0] = False
    fused, _scales = fuse_hybrid(rs, da, obj, ks, ext, jump_m=0.05)
    assert abs(float(fused[0, 3, 3]) - 2.0) < 1e-5
    assert float(fused[0, 0, 0]) == 0.0


def test_fuse_hybrid_replaces_a_flying_pixel_with_scaled_da() -> None:
    """A spike is not kept; scaled DA may rewrite it when it is not free space."""
    ks, ext = _cameras()
    rs = np.full((2, 8, 8), 2.0, dtype=np.float32)
    da = np.ones((2, 8, 8), dtype=np.float32)
    rs[0, 4, 4] = 3.0
    obj = np.ones((2, 8, 8), dtype=bool)
    fused, _scales = fuse_hybrid(rs, da, obj, ks, ext, jump_m=0.01)
    assert abs(float(fused[0, 4, 4]) - 2.0) < 1e-5
    assert abs(float(fused[0, 4, 5]) - 2.0) < 1e-5
