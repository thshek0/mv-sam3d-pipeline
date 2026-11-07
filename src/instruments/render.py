"""Instrument render: reusable implementation behind the instruments CLI."""
import numpy as np
import open3d as o3d
import trimesh


class Renderer:
    def __init__(self, parts, smooth=False):
        self.mesh = trimesh.util.concatenate(list(parts.values()))
        self.smooth = smooth
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(self.mesh.vertices, np.float32)),
                                 o3d.core.Tensor(np.asarray(self.mesh.faces, np.uint32)))

    def image(self, u, v, camera, extent, width=1600):
        """RGBA image, with pixel edges at the given millimetre plot coordinates."""
        u, v, camera = map(np.asarray, (u, v, camera))
        assert np.allclose(np.cross(u, v), camera)
        xmin, xmax, ymin, ymax = extent
        height = round(width * (ymax-ymin)/(xmax-xmin))
        xs = np.linspace(xmin, xmax, width, endpoint=False) + (xmax-xmin)/(2*width)
        ys = np.linspace(ymax, ymin, height, endpoint=False) - (ymax-ymin)/(2*height)
        x, y = np.meshgrid(xs / 1000, ys / 1000)
        starts = x[..., None]*u + y[..., None]*v + camera*3
        rays = np.concatenate([starts, np.broadcast_to(-camera, starts.shape)], axis=-1).astype(np.float32)
        hit = self.scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)
        valid = np.isfinite(hit['t_hit'].numpy())
        tri = hit['primitive_ids'].numpy()[valid].astype(int)
        normals = hit['primitive_normals'].numpy()[valid]
        if self.smooth:
            uv = hit['primitive_uvs'].numpy()[valid]
            weights = np.c_[1-uv.sum(axis=1), uv]
            normals = (self.mesh.vertex_normals[self.mesh.faces[tri]]*weights[..., None]).sum(axis=1)
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        light = camera + u*.35 + v*.65
        light /= np.linalg.norm(light)
        shade = .55 + .45*np.clip(normals @ light, 0, 1)
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[valid, :3] = np.clip(self.mesh.visual.face_colors[tri, :3]*shade[:, None], 0, 255)
        rgba[valid, 3] = 255
        interior = valid.copy()
        interior[1:, :] &= valid[:-1, :]
        interior[:-1, :] &= valid[1:, :]
        interior[:, 1:] &= valid[:, :-1]
        interior[:, :-1] &= valid[:, 1:]
        rgba[valid & ~interior, :3] = (57, 66, 73)
        return rgba
