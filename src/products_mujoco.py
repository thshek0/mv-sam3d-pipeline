"""Export existing product GLBs to portable MJCF/OBJ packages, without MuJoCo.

Run: .venv/bin/python -m src.products_mujoco
Geometry and XML checks do not constitute a MuJoCo compilation or dynamics test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image
import trimesh

from .output_layout import products_dir
from .products import ROOT, INPUTS, probe_collision, write_json

VARIANTS = {'bowl': ['bowl'], 'container': ['container_open'],
            'scale': ['electronic_balance_square', 'electronic_balance_round']}
DOCS = 'https://mujoco.readthedocs.io/en/stable/XMLreference.html'


def numbers(values):
    return ' '.join(f'{float(v):.12g}' for v in np.asarray(values).ravel())


def write_xml(path, root):
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def load_parts(path):
    """Bake scene transforms; each GLB node remains a separate shape."""
    scene = trimesh.load(path, force='scene')
    parts = {}
    for node in sorted(scene.graph.nodes_geometry):
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        mesh.apply_transform(transform)
        name = re.sub(r'[^a-zA-Z0-9_-]', '_', node)
        if name in parts:
            raise ValueError(f'Duplicate sanitized node name: {node}')
        parts[name] = mesh
    if not parts:
        raise ValueError(f'No geometry in {path}')
    return parts


def inertial_properties(parts, mass):
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('Mass must be finite and positive')
    meshes = list(parts.values())
    volumes = np.array([m.volume for m in meshes])
    masses = mass * volumes / volumes.sum()
    centers = np.array([m.center_mass for m in meshes])
    center = np.average(centers, axis=0, weights=masses)
    inertia = np.zeros((3, 3))
    for mesh, m, c in zip(meshes, masses, centers):
        offset = c - center
        inertia += mesh.moment_inertia * (m / mesh.volume)
        inertia += m * (np.dot(offset, offset)*np.eye(3) - np.outer(offset, offset))
    eigenvalues = np.linalg.eigvalsh(inertia)
    if not (np.all(eigenvalues > 0) and eigenvalues[-1] <= eigenvalues[:2].sum() + 1e-12):
        raise ValueError('Invalid inertia tensor')
    return center, inertia


def export_visual(mesh, path):
    """OBJ normals plus a small face-color palette; no external MTL dependency."""
    colors = mesh.visual.face_colors.copy()
    # MuJoCo computes mesh hulls even for visual assets. Give the flat scale
    # decals 0.02 mm inward thickness, without changing any collision geometry.
    planar = np.linalg.svd(mesh.vertices-mesh.vertices.mean(axis=0), compute_uv=False)[-1] < 1e-9
    if planar:
        normal = mesh.face_normals[0]
        mesh = trimesh.convex.convex_hull(np.vstack([mesh.vertices, mesh.vertices-normal*2e-5]))
        mesh.visual.face_colors = colors[0]
        colors = mesh.visual.face_colors
    palette, indices = np.unique(colors, axis=0, return_inverse=True)
    # Repeat palette texels to avoid bleeding at a face's constant UV.
    texture = None
    if len(palette) > 1:
        texture = path.with_suffix('.png')
        Image.fromarray(np.tile(np.repeat(palette[None, :, :3], 8, axis=1), (8, 1, 1))).save(texture)
    lines = ['# metres; object frame; colors specified by MJCF']
    lines += ['v '+numbers(v) for v in mesh.vertices]
    lines += ['vn '+numbers(n) for n in mesh.vertex_normals]
    lines += [f'vt {(i+.5)/len(palette):.12g} 0.5' for i in range(len(palette))]
    lines += ['f '+' '.join(f'{int(v)+1}/{int(c)+1}/{int(v)+1}' for v in face)
              for face, c in zip(mesh.faces, indices)]
    path.write_text('\n'.join(lines)+'\n')
    return texture, palette[0]/255, planar


def export_variant(source, variant, output, mass, physics):
    visual_path, collision_path = source/f'{variant}.glb', source/f'{variant}_collision.glb'
    visuals, collisions = load_parts(visual_path), load_parts(collision_path)
    for name, mesh in collisions.items():
        if not (mesh.is_volume and mesh.is_convex and mesh.is_winding_consistent):
            raise ValueError(f'Invalid collision piece: {name}')
    output.mkdir(parents=True, exist_ok=True)
    meshes = output/'meshes'; meshes.mkdir(exist_ok=True)
    root = ET.Element('mujoco', model=variant)
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='false')
    assets = ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    body = ET.SubElement(world, 'body', name=variant, pos='0 0 0')
    ET.SubElement(body, 'freejoint', name=variant+'_free')
    center, inertia = inertial_properties(collisions, mass)
    ET.SubElement(body, 'inertial', pos=numbers(center), mass=str(mass),
                  fullinertia=numbers([inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                       inertia[0, 1], inertia[0, 2], inertia[1, 2]]))
    thickened = []
    for name, mesh in visuals.items():
        key = f'{variant}_visual_{name}'
        path = meshes/f'{key}.obj'
        texture, color, planar = export_visual(mesh, path)
        if planar:
            thickened.append(name)
        ET.SubElement(assets, 'mesh', name=key, file=f'meshes/{path.name}', inertia='convex')
        attributes = {'rgba': numbers(color)}
        if texture:
            ET.SubElement(assets, 'texture', name=key+'_texture', type='2d', file=f'meshes/{texture.name}')
            ET.SubElement(assets, 'material', name=key+'_material', texture=key+'_texture',
                          texuniform='false', rgba='1 1 1 1')
            attributes = {'material': key+'_material'}
        ET.SubElement(body, 'geom', name=key, type='mesh', mesh=key, group='2',
                      contype='0', conaffinity='0', mass='0', **attributes)
    for name, mesh in collisions.items():
        key = f'{variant}_collision_{name}'
        path = meshes/f'{key}.obj'
        path.write_text(trimesh.exchange.obj.export_obj(mesh, include_color=False, include_texture=False))
        ET.SubElement(assets, 'mesh', name=key, file=f'meshes/{path.name}', inertia='convex')
        ET.SubElement(body, 'geom', name=key, type='mesh', mesh=key, group='3',
                      contype='1', conaffinity='1', rgba='0.3 0.7 0.9 0', condim='3',
                      friction=numbers(physics['friction']), solref=numbers(physics['solref']),
                      solimp=numbers(physics['solimp']))
    write_xml(output/'model.xml', root)
    scene = ET.Element('mujoco', model=variant+'_scene')
    ET.SubElement(scene, 'include', file='model.xml')
    ET.SubElement(scene, 'option', timestep='0.001', gravity='0 0 -9.81', integrator='implicitfast')
    ET.SubElement(scene, 'statistic', center='0 0 0.08', extent='0.5')
    world = ET.SubElement(scene, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -1 2', dir='0 0 -1', diffuse='0.8 0.8 0.8')
    ET.SubElement(world, 'geom', name='ground', type='plane', size='1 1 0.1',
                  rgba='0.8 0.82 0.85 1', friction=numbers(physics['friction']))
    write_xml(output/'scene.xml', scene)
    write_json(output/'physics.json', dict(physics, mass_kg=mass, center_of_mass_m=center.tolist(),
                                         inertia_kg_m2=inertia.tolist()))
    report = validate_package(output, collisions)
    report.update(source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [visual_path, collision_path]},
                  visual_decals_thickened_inward_0_02_mm=thickened,
                  mujoco_runtime_tested=False, physical_properties_calibrated=False,
                  documentation=DOCS+'#asset-mesh')
    write_json(output/'validation.json', report)
    (output/'README.txt').write_text(
        f'{variant}: MuJoCo MJCF + OBJ (target MuJoCo 3.3+).\n\n'
        'Open scene.xml for the object on a floor. model.xml contains only the free rigid object.\n'
        'Keep meshes/ beside the XML files. Metres, Z up, object bottom Z=0.\n'
        'For fixed placement, remove the freejoint from model.xml. Move the object using body pos/quat.\n'
        'When merging into a larger scene, adjust asset paths and retain every separate collision geom.\n'
        'Visual geoms: group 2, collisions disabled. Collision geoms: group 3, transparent.\n'
        'To inspect collision geometry, give group 3 geoms nonzero RGBA alpha.\n\n'
        f'Mass: {mass} kg (placeholder). Friction and solver contact values are also unmeasured.\n'
        'Inertia uses uniform density across collision pieces; it is not measured.\n'
        'Edit body/inertial and collision geom attributes in model.xml, or edit the source\n'
        'src/product_inputs/mujoco_physics.json and rerun python -m src.products_mujoco.\n'
        'The packaged physics.json is an export record, not a live MJCF configuration.\n'
        'Scale variants are rigid props: no load cell, weighing sensor or animated display.\n'
        'Planar visual decals have 0.02 mm inward thickness for mesh hull compatibility.\n'
        'Visual colors use per-face averages of the source GLB vertex colors.\n\n'
        'Validation: XML references, OBJ reload, geometry, convexity, cavities and inertia.\n'
        'NOT compiled or simulated in MuJoCo; physical behaviour is unvalidated.\n'
        f'Format reference: {DOCS}#asset-mesh\n')
    return report


def validate_package(output, original):
    """Static export checks, explicitly without a MuJoCo runtime."""
    root = ET.parse(output/'model.xml').getroot()
    scene = ET.parse(output/'scene.xml').getroot()
    assert (output/scene.find('include').get('file')).is_file()
    loaded, all_assets = {}, {}
    max_error = 0.
    for element in root.findall('asset/*'):
        if element.get('file'):
            assert (output/element.get('file')).is_file()
        if element.tag != 'mesh':
            continue
        mesh = trimesh.load(output/element.get('file'), force='mesh', process=False)
        assert np.isfinite(mesh.vertices).all() and np.all(mesh.area_faces > 1e-14)
        assert np.linalg.svd(mesh.vertices-mesh.vertices.mean(axis=0), compute_uv=False)[-1] > 1e-9
        all_assets[element.get('name')] = mesh
    for geom in root.findall('worldbody/body/geom'):
        assert geom.get('mesh') in all_assets
        if geom.get('group') == '2':
            assert geom.get('contype') == geom.get('conaffinity') == '0'
            continue
        name = geom.get('name').split('_collision_', 1)[1]
        mesh = all_assets[geom.get('mesh')]
        assert mesh.is_volume and mesh.is_convex and mesh.is_winding_consistent
        error = float(np.max(np.abs(mesh.vertices-original[name].vertices)))
        assert error < 1e-7
        assert np.array_equal(mesh.faces, original[name].faces)
        max_error = max(max_error, error)
        loaded[name] = mesh
    assert loaded.keys() == original.keys()
    bounds = np.array([m.bounds for m in loaded.values()])
    lower, upper = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    assert abs(lower[2]) < 1e-7
    checks = {}
    variant = root.get('model')
    if variant in ('bowl', 'container_open'):
        floor_z = loaded['floor'].bounds[1, 2]*1000
        heights = np.linspace(floor_z+1, upper[2]*1000+10, 50)
        assert not probe_collision(loaded, [[0, 0, z] for z in heights]).any()
        assert probe_collision(loaded, [loaded['floor'].center_mass*1000])[0]
        checks = {'central_cavity_probes_free': 50, 'floor_probe_solid': True}
    else:
        assert probe_collision(loaded, [loaded['weighing_pan'].center_mass*1000])[0]
        checks = {'weighing_pan_probe_solid': True}
    return {'units': 'metres', 'collision_pieces': len(loaded), 'visual_meshes': len(all_assets)-len(loaded),
            'collision_max_vertex_error_m': max_error, 'bounds_m': [lower.tolist(), upper.tolist()],
            'static_geometry_checks': checks, 'xml_references_resolved': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['all', *VARIANTS], default='all',
                        type=lambda name: {'electronic_balance': 'scale', 'sand_container': 'container', 'sand_bucket': 'container',
                                           'hestia_bowl_v1': 'bowl', 'hestia_bowl': 'bowl'}.get(name, name))
    parser.add_argument('--physics', type=Path, default=INPUTS/'mujoco_physics.json')
    args = parser.parse_args()
    physics = json.loads(args.physics.read_text())
    names = list(VARIANTS) if args.model == 'all' else [args.model]
    for model in names:
        source = products_dir(ROOT/'output', model, 'codex')
        for variant in VARIANTS[model]:
            output = source/'mujoco'/variant
            report = export_variant(source, variant, output, physics['mass_kg'][model], physics)
            print(f'{variant}: {report["collision_pieces"]} convex pieces -> {output}', flush=True)
    # Repackage every available variant, including unchanged ones on partial builds.
    packages = ROOT/'output/packages'; packages.mkdir(exist_ok=True)
    with zipfile.ZipFile(packages/'mujoco_products.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for model, variants in VARIANTS.items():
            for variant in variants:
                folder = products_dir(ROOT/'output', model, 'codex')/'mujoco'/variant
                if folder.exists():
                    for path in sorted(folder.rglob('*')):
                        if path.is_file():
                            archive.write(path, f'{variant}/{path.relative_to(folder)}')
    print(packages/'mujoco_products.zip')


if __name__ == '__main__':
    main()
