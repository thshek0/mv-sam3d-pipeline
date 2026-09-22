"""Optional S6 seal and S7 CoACD, plus three-view PNG previews.

Seal keeps the largest shell and fills holes. It does not quadric-collapse
to 10k. CoACD uses t=0.05 only.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from trimesh import repair

VIEWS: tuple[tuple[str, float, float], ...] = (
    ("front", 20.0, -60.0),
    ("side", 15.0, 30.0),
    ("top", 75.0, -90.0),
)
COACD_T = 0.05
PREVIEW_MAX_FACES = 12000
PART_TINTS: tuple[np.ndarray, ...] = (
    np.array([0.62, 0.70, 0.82]),
    np.array([0.86, 0.62, 0.38]),
    np.array([0.42, 0.72, 0.52]),
    np.array([0.78, 0.52, 0.62]),
)


def load_triangle_mesh(path: Path) -> trimesh.Trimesh:
    """Load one triangle mesh, concatenating a scene if needed."""
    loaded = trimesh.load(path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"{path} did not load as a triangle mesh ({type(loaded).__name__})")
    if len(loaded.faces) == 0:
        raise ValueError(f"{path} has no faces")
    return loaded


def _require_open3d():
    """Import Open3D only when sealing, CoACD, or preview decimate runs."""
    import open3d as o3d
    return o3d


def _require_coacd():
    """Import CoACD after Open3D (TBB clash)."""
    _require_open3d()
    import coacd
    return coacd


def to_o3d(mesh: trimesh.Trimesh):
    """Convert a trimesh to Open3D."""
    o3d = _require_open3d()
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    out.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    return out


def n_open_edges(mesh: trimesh.Trimesh) -> int:
    """Count edges used by exactly one triangle."""
    if len(mesh.faces) == 0:
        return 0
    grouped = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    return int(len(grouped))


def _largest_shell(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Keep the connected component with the largest surface area."""
    parts = mesh.split(only_watertight=False)
    if not parts:
        raise RuntimeError("Mesh split produced no components")
    return max(parts, key=lambda part: float(part.area))


def _clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Merge coincident verts and drop degenerate / duplicate faces."""
    work = mesh.copy()
    work.merge_vertices()
    work.update_faces(work.nondegenerate_faces())
    work.update_faces(work.unique_faces())
    work.remove_unreferenced_vertices()
    return work


def _boundary_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    """Return edges used by exactly one triangle."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for face in faces:
        for i, j in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (int(min(i, j)), int(max(i, j)))
            counts[edge] += 1
    return [edge for edge, count in counts.items() if count == 1]


def _fan_cap(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fill leftover open loops with a planar fan."""
    vertices = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.faces).copy()
    edges = _boundary_edges(faces)
    if not edges:
        return mesh
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    used: set[int] = set()
    for start in list(adj):
        if start in used:
            continue
        loop = [start]
        used.add(start)
        prev: int | None = None
        cur = start
        while True:
            nxt = None
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == start or nb not in used:
                    nxt = nb
                    break
            if nxt is None:
                break
            if nxt == start:
                break
            loop.append(nxt)
            used.add(nxt)
            prev, cur = cur, nxt
            if len(loop) > len(edges) + 2:
                break
        if len(loop) < 3:
            continue
        idx = np.asarray(loop, dtype=np.int64)
        center = vertices[idx].mean(axis=0)
        new_i = len(vertices)
        vertices = np.vstack([vertices, center])
        caps = np.asarray(
            [[loop[i], loop[(i + 1) % len(loop)], new_i] for i in range(len(loop))], dtype=np.int32
        )
        faces = np.vstack([faces, caps])
    return trimesh.Trimesh(vertices, faces, process=True)


def seal_mesh(path: Path) -> trimesh.Trimesh:
    """Largest shell + hole fill. No face-count cap."""
    work = _largest_shell(_clean(load_triangle_mesh(path)))
    print(
        f"seal: {len(work.faces)} faces open={n_open_edges(work)} watertight={work.is_watertight}",
        flush=True,
    )
    if not work.is_watertight:
        repair.fill_holes(work, use_fan=True)
        print(f"after fan fill: open={n_open_edges(work)} watertight={work.is_watertight}", flush=True)
    if not work.is_watertight:
        work = _clean(_fan_cap(work))
        print(f"after planar fan: open={n_open_edges(work)} watertight={work.is_watertight}", flush=True)
    work = _largest_shell(_clean(work))
    repair.fix_normals(work)
    if work.volume < 0:
        work.invert()
        repair.fix_normals(work)
    if not work.is_watertight or not work.is_volume:
        raise RuntimeError(
            f"Mesh still not closed (watertight={work.is_watertight}, "
            f"is_volume={work.is_volume}, open={n_open_edges(work)})"
        )
    print(f"sealed: {len(work.faces)} faces volume={work.volume:.6f}", flush=True)
    return work


def coacd_parts(mesh: trimesh.Trimesh, threshold: float = COACD_T) -> list[trimesh.Trimesh]:
    """Approximate convex decomposition at concavity ``threshold``."""
    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    print(f"coacd t={threshold} faces_in={len(faces)}", flush=True)
    coacd = _require_coacd()
    raw = coacd.run_coacd(
        coacd.Mesh(verts, faces),
        threshold=threshold,
        max_convex_hull=-1,
        preprocess_mode="on",
        preprocess_resolution=50,
        merge=True,
        decimate=True,
        max_ch_vertex=64,
        seed=0,
    )
    parts: list[trimesh.Trimesh] = []
    for vs, fs in raw:
        part = _clean(trimesh.Trimesh(np.asarray(vs), np.asarray(fs), process=True))
        if len(part.faces) < 4:
            continue
        if not part.is_watertight:
            part = part.convex_hull
        if part.volume <= 0:
            continue
        parts.append(part)
    if not parts:
        raise RuntimeError("CoACD produced no hulls")
    print(f"coacd: {len(parts)} hulls", flush=True)
    return parts


def write_coacd(mesh: trimesh.Trimesh, run_dir: Path) -> None:
    """Write combined STL, per-hull STLs, and a tinted preview."""
    parts = coacd_parts(mesh)
    combined = trimesh.util.concatenate(parts)
    combined.export(run_dir / "convex_parts.stl")
    parts_dir = run_dir / "convex_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for old in parts_dir.glob("part_*.stl"):
        old.unlink()
    for i, part in enumerate(parts):
        part.export(parts_dir / f"part_{i:02d}.stl")
    render_parts(parts, run_dir / "convex_parts.png")
    print(f"wrote {len(parts)} hulls -> {parts_dir}", flush=True)


def _decimate_for_preview(mesh: trimesh.Trimesh, max_faces: int = PREVIEW_MAX_FACES) -> trimesh.Trimesh:
    """In-memory quadric for matplotlib only."""
    if max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh
    src = to_o3d(mesh)
    simplified = src.simplify_quadric_decimation(target_number_of_triangles=max_faces)
    simplified.remove_degenerate_triangles()
    return trimesh.Trimesh(np.asarray(simplified.vertices), np.asarray(simplified.triangles), process=False)


def face_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """Shade each face from one fixed light."""
    light = np.array([0.35, -0.55, 0.76], dtype=np.float64)
    light /= np.linalg.norm(light)
    shade = 0.22 + 0.78 * np.clip(mesh.face_normals @ light, 0.0, 1.0)
    return np.column_stack((shade * 0.72, shade * 0.76, shade * 0.82))


def preview_mesh(path: Path, output: Path) -> Path:
    """Three-view PNG of a GLB/STL. Dense meshes are thinned only for the PNG."""
    mesh = _decimate_for_preview(load_triangle_mesh(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = face_colors(mesh)
    lo, hi = mesh.bounds
    span = np.maximum(hi - lo, 1e-9)
    fig, axes = plt.subplots(1, len(VIEWS), figsize=(4.2 * len(VIEWS), 4.4), subplot_kw={"projection": "3d"})
    for ax, (title, elev, azim) in zip(np.atleast_1d(axes), VIEWS):
        ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors=colors, linewidths=0))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(span)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"preview -> {output}", flush=True)
    return output


def render_parts(parts: list[trimesh.Trimesh], output: Path) -> Path:
    """Draw convex parts with distinct tints."""
    output.parent.mkdir(parents=True, exist_ok=True)
    bounds = np.vstack([part.bounds for part in parts])
    lo, hi = bounds.min(axis=0), bounds.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    fig, axes = plt.subplots(1, len(VIEWS), figsize=(4.2 * len(VIEWS), 4.4), subplot_kw={"projection": "3d"})
    for ax, (title, elev, azim) in zip(np.atleast_1d(axes), VIEWS):
        for i, part in enumerate(parts):
            tint = PART_TINTS[i % len(PART_TINTS)]
            shade = face_colors(part)
            mixed = np.clip(0.45 * tint.reshape(1, 3) + 0.55 * shade, 0.0, 1.0)
            ax.add_collection3d(
                Poly3DCollection(
                    part.triangles,
                    facecolors=mixed,
                    edgecolors=(0.12, 0.12, 0.14, 0.55),
                    linewidths=0.25,
                )
            )
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(span)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output
