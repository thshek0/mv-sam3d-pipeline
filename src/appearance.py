"""Shared helpers to preserve image UVs or bake GLB colours into OBJ/PNG.

Original GLBs are delivered separately, unchanged. Vertex colours are averaged
per triangle for this texture bake; material lighting is not reproduced.
"""
import numpy as np
from PIL import Image
import trimesh


def load_surface(path):
    scene = trimesh.load(path, force='scene')
    meshes, colors, normals, textures = [], [], [], []
    for node in scene.graph.nodes_geometry:
        transform, key = scene.graph[node]
        mesh = scene.geometry[key].copy()
        mesh.apply_transform(transform)
        visual = mesh.visual
        if visual.kind == 'texture':
            material = visual.material
            image = getattr(material, 'baseColorTexture', None)
            image = image if image is not None else getattr(material, 'image', None)
            base = getattr(material, 'baseColorFactor', None)
            base = base if base is not None else getattr(material, 'diffuse', [255]*4)
            base = np.asarray(base, dtype=np.uint8)
            if image is None:
                color = np.tile(base, (len(mesh.faces), 1))
            else:
                uv = np.asarray(visual.uv)
                if uv.shape != (len(mesh.vertices), 2) or not np.isfinite(uv).all():
                    raise ValueError('Image texture requires finite per-vertex UV coordinates')
                rgba = np.asarray(image.convert('RGBA'))
                if base[3] != 255 or np.any(rgba[:, :, 3] != 255):
                    raise ValueError('Transparent input requires a separate alpha-material export')
                # glTF factors multiply linear colour, while PNG stores sRGB.
                rgb = rgba[:, :, :3] / 255.
                linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)
                linear *= base[:3] / 255.
                rgb = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear**(1/2.4) - .055)
                pixels = np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8)
                textures.append({'uv': uv.copy(), 'image': Image.fromarray(pixels)})
                color = np.tile([255]*4, (len(mesh.faces), 1))
        else:
            color = mesh.visual.face_colors.copy()
        if np.any(color[:, 3] != 255):
            raise ValueError('Transparent input requires a separate alpha-material export')
        colors.append(color[:, :3])
        normals.append(mesh.vertex_normals.copy())
        meshes.append(trimesh.Trimesh(mesh.vertices, mesh.faces, process=False))
    if not meshes:
        raise ValueError(f'Empty scene: {path}')
    mesh = trimesh.util.concatenate(meshes)
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f'Non-finite geometry: {path}')
    if textures:
        if len(meshes) != 1:
            raise ValueError('Mixed or multiple image materials need an explicit texture atlas')
        return mesh, textures[0], np.vstack(normals)
    return mesh, np.vstack(colors), np.vstack(normals)


def write_color_obj(mesh, colors, normals, path):
    if isinstance(colors, dict):
        return write_image_obj(mesh, colors, normals, path)
    palette, indices = np.unique(colors, axis=0, return_inverse=True)
    side = 2**int(np.ceil(np.log2(max(1, np.sqrt(len(palette))))))
    tile = 4
    grid = np.zeros((side*side, 3), dtype=np.uint8)
    grid[:len(palette)] = palette
    pixels = np.repeat(np.repeat(grid.reshape(side, side, 3), tile, axis=0), tile, axis=1)
    texture = path.with_name(path.stem+'_colors.png')
    Image.fromarray(pixels).save(texture)
    slots = np.arange(len(palette))
    uv = np.c_[(slots % side+.5)/side, 1-(slots//side+.5)/side]
    # Verify all palette entries using the standard bottom-left OBJ UV convention.
    sampled = trimesh.visual.color.uv_to_color(uv, Image.open(texture))[:, :3]
    if not np.array_equal(sampled, palette):
        raise ValueError('Palette UV mapping failed')
    material = path.with_suffix('.mtl')
    material.write_text('newmtl base_color\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\n'
                        f'map_Kd {texture.name}\n')
    with path.open('w') as stream:
        stream.write(f'# Source coordinates preserved\nmtllib {material.name}\n')
        np.savetxt(stream, mesh.vertices, fmt='v %.10f %.10f %.10f')
        np.savetxt(stream, normals, fmt='vn %.10f %.10f %.10f')
        np.savetxt(stream, uv, fmt='vt %.10f %.10f')
        stream.write('usemtl base_color\n')
        for face, slot in zip(mesh.faces+1, indices+1):
            a, b, c = face
            stream.write(f'f {a}/{slot}/{a} {b}/{slot}/{b} {c}/{slot}/{c}\n')
    return {'texture': texture.name, 'material_file': material.name,
            'texture_size_px': list(Image.open(texture).size), 'palette_colors': len(palette),
            'palette_uv_colors_verified': True, 'mujoco_texture_vflip': False,
            'bake': 'Base material colours and per-triangle averages of GLB vertex colours',
            'limitations': 'Original smooth vertex-colour interpolation and PBR lighting remain in the unchanged GLB'}


def write_image_obj(mesh, appearance, normals, path):
    """Write the original UVs, including seams, rather than triangle averages."""
    texture = path.with_name(path.stem+'_colors.png')
    appearance['image'].save(texture)
    material = path.with_suffix('.mtl')
    material.write_text('newmtl base_color\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\n'
                        f'map_Kd {texture.name}\n')
    with path.open('w') as stream:
        stream.write(f'# Source coordinates and image UVs preserved\nmtllib {material.name}\n')
        np.savetxt(stream, mesh.vertices, fmt='v %.10f %.10f %.10f')
        np.savetxt(stream, normals, fmt='vn %.10f %.10f %.10f')
        np.savetxt(stream, appearance['uv'], fmt='vt %.10f %.10f')
        stream.write('usemtl base_color\n')
        for face in mesh.faces+1:
            stream.write('f '+' '.join(f'{v}/{v}/{v}' for v in face)+'\n')
    return {'texture': texture.name, 'material_file': material.name,
            'texture_size_px': list(appearance['image'].size),
            'uv_mapping_preserved': True, 'mujoco_texture_vflip': False,
            'bake': 'Original image texture and per-vertex UVs; base material factor applied',
            'limitations': 'PBR lighting is not reproduced by the MuJoCo material'}
