import numpy as np
import pytest
import trimesh
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from src.export import export_pair


def test_export_preserves_instance_transforms_and_metres(tmp_path):
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=[.02, .04, .06]),
                       node_name='first', geom_name='shared')
    scene.graph.update(frame_to='second', geometry='shared',
                       matrix=trimesh.transformations.translation_matrix([.1, 0, 0]))
    source = tmp_path/'instances.glb'
    scene.export(source)
    report = export_pair(source, tmp_path/'export', 'boxes_m')
    assert report['triangles'] == 24
    for suffix in ('obj', 'stl'):
        mesh = trimesh.load(tmp_path/'export'/f'boxes_m.{suffix}', force='mesh')
        np.testing.assert_allclose(mesh.extents, [.12, .04, .06], atol=1e-8)
        np.testing.assert_allclose(mesh.bounds, [[-.01, -.02, -.03], [.11, .02, .03]], atol=1e-8)
        assert len(mesh.split()) == 2


def test_collision_export_keeps_hulls_separate_and_resolves_visual_assets(tmp_path):
    from src.collision import export_collision
    source = tmp_path/'object.glb'
    parts = trimesh.Scene()
    for i in range(2):
        mesh = trimesh.creation.box(extents=[.01, .02, .03])
        mesh.apply_translation([i*.04, 0, .015])
        parts.add_geometry(mesh, geom_name=f'part_{i}', node_name=f'part_{i}')
    parts.export(source)
    parts.export(tmp_path/'object_collision.glb')
    visual = tmp_path/'export/object_m.obj'
    report = export_pair(source, visual.parent, visual.stem)
    directory = visual.parent/'collision'/visual.stem
    collision = export_collision(source, directory, visual, report['appearance'])
    assert collision['convex_parts'] == 2
    xml = ET.parse(directory/'model.xml').getroot()
    geoms = xml.findall('worldbody/body/geom')
    assert sum(g.get('group') == '3' for g in geoms) == 2
    visible = next(g for g in geoms if g.get('group') == '2')
    assert visible.get('contype') == '0'
    assert visible.get('material') == 'visual_material'
    assert xml.find('asset/texture').get('vflip') == 'false'
    for asset in xml.findall('asset/mesh') + xml.findall('asset/texture'):
        assert (directory/asset.get('file')).is_file()


def test_colour_palette_roundtrip_and_material_base_colour(tmp_path):
    from src.appearance import write_color_obj
    from trimesh.visual.material import PBRMaterial
    mesh = trimesh.creation.box()
    colors = np.array([[20*i, 255-20*i, 50] for i in range(len(mesh.faces))], dtype=np.uint8)
    obj = tmp_path/'palette.obj'
    write_color_obj(mesh, colors, mesh.vertex_normals, obj)
    loaded = trimesh.load(obj, force='mesh', process=False)
    sampled = trimesh.visual.color.uv_to_color(loaded.visual.uv, loaded.visual.material.image)
    np.testing.assert_array_equal(sampled[loaded.faces][:, 0, :3], colors)
    np.testing.assert_array_equal(sampled[loaded.faces][:, 1, :3], colors)
    # Material-only GLBs have no UV array; their base colour must not become grey.
    mesh.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(baseColorFactor=[30, 120, 210, 255]))
    source = tmp_path/'material.glb'; mesh.export(source)
    report = export_pair(source, tmp_path/'material', 'material_m')
    from PIL import Image
    palette = np.asarray(Image.open(tmp_path/'material'/report['appearance']['texture']))
    np.testing.assert_array_equal(palette[0, 0], [30, 120, 210])
    assert (tmp_path/'material/material_m.glb').read_bytes() == source.read_bytes()


def test_mujoco_compiled_palette_uvs_sample_the_original_face_colours(tmp_path):
    """A PNG flip after OBJ import used to render the balance as a black silhouette."""
    mujoco = pytest.importorskip('mujoco')
    from src.collision import export_collision
    mesh = trimesh.creation.box()
    hull = mesh.copy()
    # glTF carries vertex colours. Duplicate shared corners so per-face colours
    # survive GLB encoding without averaging across neighbouring faces.
    mesh.unmerge_vertices()
    # Three colours in a 2x2 palette leave a black fourth slot: a flip is observable.
    colours = np.repeat([[230, 20, 30, 255], [30, 220, 40, 255], [40, 50, 210, 255]], 4, axis=0)
    mesh.visual.face_colors = colours
    source = tmp_path/'colour_box.glb'; mesh.export(source)
    report = export_pair(source, tmp_path/'export', 'box')
    directory = tmp_path/'export/collision'
    export_collision(source, directory, tmp_path/'export/box.obj', report['appearance'],
                     parts={'hull': hull})
    model = mujoco.MjModel.from_xml_path(str(directory/'model.xml'))
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, 'visual')
    face_start = model.mesh_faceadr[mesh_id]
    faces = model.mesh_facetexcoord[face_start:face_start+model.mesh_facenum[mesh_id]]
    uv = model.mesh_texcoord[model.mesh_texcoordadr[mesh_id]+faces[:, 0]]
    height, width = int(model.tex_height[0]), int(model.tex_width[0])
    pixels = model.tex_data[:height*width*3].reshape(height, width, 3)
    xy = np.floor(uv*[width, height]).astype(int)
    sampled = pixels[xy[:, 1], xy[:, 0]]
    assert sorted(map(tuple, sampled)) == sorted(map(tuple, colours[:, :3]))


def test_image_texture_uvs_survive_glb_obj_export(tmp_path):
    from PIL import Image
    from trimesh.visual.material import PBRMaterial
    # Asymmetric texture reveals vertical flips, lost UVs and colour averaging.
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[:16, :16] = [230, 10, 20]
    pixels[:16, 16:] = [10, 220, 30]
    pixels[16:, :16] = [20, 30, 210]
    pixels[16:, 16:] = [240, 210, 15]
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=[[.1, .1], [.9, .1], [.9, .9], [.1, .9]],
        material=PBRMaterial(baseColorTexture=Image.fromarray(pixels), baseColorFactor=[255]*4))
    source = tmp_path/'image.glb'; mesh.export(source)
    report = export_pair(source, tmp_path/'export', 'image_m')
    restored = trimesh.load(tmp_path/'export/image_m.obj', force='mesh', process=False)
    np.testing.assert_array_equal(np.asarray(restored.visual.material.image), pixels)
    np.testing.assert_allclose(restored.visual.uv[restored.faces], mesh.visual.uv[mesh.faces], atol=1e-7)
    assert report['appearance']['uv_mapping_preserved']


@pytest.mark.parametrize('name', ['plate_maehata_pop_blue_23cm', 'plate_strimmig_pale_grey_green_21cm', 'plate_paradisisk_off_white_20cm'])
def test_plate_pattern_preserves_surface_and_leaves_underside_plain(tmp_path, name):
    from src.plates.appearance import apply_appearance
    mesh = trimesh.creation.cylinder(radius=.115, height=.037, sections=32)
    recipe_path = f'dataset/hestia-rl/{name}/appearance/pattern.json'
    recipe = json.loads((Path(__file__).resolve().parents[1]/recipe_path).read_text())
    params = {'rgba': recipe['background_rgb']+[255], 'appearance_recipe': recipe_path}
    textured, report = apply_appearance(mesh, name, params, tmp_path)
    np.testing.assert_array_equal(textured.triangles, mesh.triangles)
    np.testing.assert_allclose(textured.vertex_normals[textured.faces], mesh.vertex_normals[mesh.faces])
    assert report['images'] == report['textures'] == 1
    underside = textured.visual.uv[textured.faces[mesh.face_normals[:, 2] < 0]]
    np.testing.assert_allclose(underside, np.broadcast_to([.75, .5], underside.shape))
    texture = np.asarray(textured.visual.material.baseColorTexture)
    assert np.all(texture[:, 1100:] == params['rgba'][:3])
    assert np.any(texture[:, :1024] != params['rgba'][:3])  # Decoration exists.
    # Food area stays plain for the two plates with rim-only decoration.
    if name != 'plate_maehata_pop_blue_23cm':
        assert np.all(texture[450:574, 450:574] == params['rgba'][:3])
        assert 'blue/gold' not in report['material']
    textured.merge_vertices(merge_tex=True, merge_norm=True)
    assert textured.is_watertight
