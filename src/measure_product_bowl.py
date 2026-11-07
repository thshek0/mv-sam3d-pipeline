"""Reproduce the provisional bowl diameter/height estimate from local RGB-D."""
from pathlib import Path
import json
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'dataset/hestia-rl/hestia_bowl_v1/derived/rgbd'
    data = np.load(folder/'depth/rs.npz')
    stem = '20261013_153506_636313'
    index = list(data['keep_stems']).index(stem)
    depth = data['depth'][index]; k = data['intrinsics'][index]
    mask = np.asarray(Image.open(folder/f'scene/masks/{index}.png'))[:, :, -1]>0
    y, x = np.indices(depth.shape)
    rays = np.stack([(x-k[0, 2])/k[0, 0], (y-k[1, 2])/k[1, 1], np.ones_like(x)], axis=-1)
    points = rays*depth[..., None]
    outer = cv2.dilate(mask.astype('uint8'), np.ones((241, 241), 'uint8'))>0
    inner = cv2.dilate(mask.astype('uint8'), np.ones((11, 11), 'uint8'))>0
    table_mask = outer & ~inner & (depth>.38) & (depth<.58)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[table_mask]))
    o3d.utility.random.seed(7)
    plane, inliers = cloud.segment_plane(.0025, 3, 1000)
    plane = np.array(plane)
    if plane[2]>0: plane = -plane
    normal, offset = plane[:3], plane[3]
    object_points = points[mask & (depth>.3) & (depth<.65)]
    heights = object_points@normal+offset
    height = float(np.percentile(heights, 99))
    # A circle fitted at the rim plane ignores portions where the side/base
    # extends past the projected rim. The soft loss limits their influence.
    contour = max(cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)[:, 0, :]
    contour_rays = np.c_[contour, np.ones(len(contour))]@np.linalg.inv(k).T
    projected = contour_rays*((height-offset)/(contour_rays@normal))[:, None]
    u = np.cross(normal, [0, 1, 0]); u /= np.linalg.norm(u); v = np.cross(normal, u)
    xy = np.c_[projected@u, projected@v]
    fit = least_squares(lambda q: np.linalg.norm(xy-q[:2], axis=1)-q[2], [*xy.mean(axis=0), .06], loss='soft_l1', f_scale=.001)
    report = {'source_stem': stem, 'source_npz': str((folder/'depth/rs.npz').relative_to(ROOT)),
              'method': 'Local-table RANSAC plane; 99th percentile object height; robust circle fit of silhouette rays on rim plane.',
              'table_plane_camera_coordinates': plane.tolist(), 'table_points': int(table_mask.sum()),
              'table_inliers': len(inliers), 'estimated_height_mm': height*1000, 'estimated_diameter_mm': float(fit.x[2]*2000),
              'circle_absolute_residual_mm_p50_p90': np.percentile(abs(fit.fun)*1000, [50, 90]).tolist(),
              'authored_diameter_height_mm': [120, 60], 'provisional_uncertainty_mm': 10,
              'limitations': 'Single usable top view; sparse reflective-surface depth and silhouette/body occlusion. Other views are visual checks, not calibrated dimension confirmations. Wall and floor thickness are not measured.'}
    out = ROOT/'output/objects/hestia-rl/hestia_bowl_v1/codex/intermediate/references/bowl_measurement.json'; out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2))


if __name__=='__main__': main()
