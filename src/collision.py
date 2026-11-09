"""Collision construction and export helpers; no family selection or ZIP workflow."""
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from .products_mujoco import load_parts, write_xml

RECIPE = 1


def plate_parts(source, directory=None):
    from .plates.rebuild import keep_profile, read_profile, NAMES
    directory = source.parent if directory is None else directory
    name = source.stem.removesuffix('_m')
    if name in NAMES:
        params, dims, profile = read_profile(name, directory if directory.is_file() else None)
    else:
        directory = directory.parent if directory.is_file() else directory
        params = json.loads((directory/'parameters.json').read_text())
        dims = json.loads((directory/'dimensions.json').read_text())
        profile = np.loadtxt(directory/params['profile_csv'], delimiter=',', skiprows=1)
    r, top, bottom = profile.T
    r *= dims['outer_diameter_mm']/(2*r[-1])
    factor = dims['overall_height_mm']/top.max()
    top *= factor; bottom *= factor
    ti, _ = keep_profile(r, top, params['profile_tolerance_mm'])
    bi, _ = keep_profile(r, bottom, params['profile_tolerance_mm'])
    knots = np.unique(np.r_[r[ti], r[bi]])
    upper = np.interp(knots, r[ti], top[ti])
    lower = np.interp(knots, r[bi], bottom[bi])
    parts = {}
    for sector in range(params['angular_sections']):
        angles = 2*np.pi*np.array([sector, sector+1])/params['angular_sections']
        cells = []
        for k in range(len(knots)-1):
            vertices = [[knots[j]*np.cos(a), knots[j]*np.sin(a), z]
                        for j in (k, k+1) for a in angles for z in (lower[j], upper[j])]
            cell = trimesh.convex.convex_hull(np.asarray(vertices)/1000)
            # Join adjacent radial cells only when their union remains convex.
            if cells:
                combined = trimesh.convex.convex_hull(np.vstack([cells[-1].vertices, cell.vertices]))
                if np.isclose(combined.volume, cells[-1].volume+cell.volume, rtol=1e-7, atol=1e-14):
                    cells[-1] = combined
                    continue
            cells.append(cell)
        for j, cell in enumerate(cells):
            parts[f'sector_{sector:02}_band_{j:02}'] = cell
    visual = trimesh.load(source, force='mesh')
    volume_error = abs(sum(m.volume for m in parts.values())-visual.volume)/visual.volume
    if volume_error > 1e-5:
        raise ValueError(f'Plate collision volume differs from visual: {volume_error}')
    from .products import probe_collision
    probes = [[0, 0, z] for z in np.linspace(top[0]+.1, top.max()+10, 30)]
    if probe_collision(parts, probes).any():
        raise ValueError('Plate collision fills the dish')
    return parts, {'construction': 'Convex radial cells of the current simplified plate profile',
                   'visual_collision_volume_relative_error': volume_error,
                   'central_cavity_probes_free': len(probes), 'approximate': False}


def bird_parts(source):
    # Open3D must load before CoACD because both use native threading libraries.
    import open3d as o3d
    import coacd
    visual = trimesh.load(source, force='mesh')
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(visual.vertices),
                                   o3d.utility.Vector3iVector(visual.faces))
    simplified = mesh.simplify_quadric_decimation(12000)
    settings = dict(threshold=.03, preprocess_mode='on', preprocess_resolution=60,
                    resolution=2000, mcts_iterations=80, merge=True, decimate=True,
                    max_ch_vertex=64, seed=0)
    coacd.set_log_level('warn')
    raw = coacd.run_coacd(coacd.Mesh(np.asarray(simplified.vertices),
                                    np.asarray(simplified.triangles)), **settings)
    parts = {f'part_{i:03}': trimesh.Trimesh(v, f).convex_hull for i, (v, f) in enumerate(raw)}
    return parts, {'construction': 'CoACD on a decimated copy; source visual unchanged',
                   'approximate': True, 'decimated_input_triangles': len(simplified.triangles),
                   'coacd_settings': settings,
                   'units': 'Unverified source units; calibrate size before physical simulation'}


def bird_geometry_checks(source, parts):
    import open3d as o3d
    visual = trimesh.load(source, force='mesh')
    compound = trimesh.util.concatenate(list(parts.values()))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(compound.vertices),
                                 o3d.utility.Vector3iVector(compound.faces))))
    points = visual.vertices[np.linspace(0, len(visual.vertices)-1, 5000, dtype=int)]
    distances = scene.compute_distance(o3d.core.Tensor(points.astype('float32'))).numpy()
    span = float(visual.extents.max())
    bounds_error = float(np.abs(compound.bounds-visual.bounds).max())
    if bounds_error > span*.05 or np.percentile(distances, 95) > span*.05:
        raise ValueError('Bird collision differs excessively from visual geometry')
    return {'source_bounds': visual.bounds.tolist(), 'collision_bounds': compound.bounds.tolist(),
            'max_bounds_error_source_units': bounds_error,
            'sampled_visual_vertex_to_nearest_hull_surface': {
                'samples': len(points), 'p95_source_units': float(np.percentile(distances, 95)),
                'max_source_units': float(distances.max())},
            'limitations': 'Sampled nearest hull surfaces, not a Hausdorff bound or dynamics validation'}


def source_digest(source):
    digest = hashlib.sha256(source.read_bytes())
    digest.update(str(RECIPE).encode())
    if source.name.startswith('plate_'):
        from .plates.rebuild import INPUTS, NAMES
        if source.stem.removesuffix('_m') in NAMES:
            digest.update(INPUTS.read_bytes())
        else:
            for filename in ('parameters.json', 'dimensions.json'):
                digest.update((source.parent/filename).read_bytes())
            params = json.loads((source.parent/'parameters.json').read_text())
            digest.update((source.parent/params['profile_csv']).read_bytes())
    return digest.hexdigest()


def get_parts(source):
    """Reuse existing authored compounds; cache new compounds by source hash."""
    existing = source.with_name(source.stem+'_collision.glb')
    if existing.exists():
        return load_parts(existing), {'construction': 'Existing authored convex assembly',
                                      'source': str(existing), 'approximate': source.stem == 'bowl'}
    cache = source.parent/'collision'/source.stem
    key = source_digest(source)
    if (cache/'report.json').exists():
        report = json.loads((cache/'report.json').read_text())
        if report.get('source_recipe_sha256') == key and (cache/'parts.glb').exists():
            return load_parts(cache/'parts.glb'), report
    if source.name.startswith('plate_'):
        parts, report = plate_parts(source)
    else:
        parts, report = bird_parts(source)
        report['geometry_checks'] = bird_geometry_checks(source, parts)
    validate_parts(parts)
    cache.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for name, mesh in parts.items():
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    scene.export(cache/'parts.glb')
    report.update(source_recipe_sha256=key, recipe=RECIPE, convex_parts=len(parts),
                  mujoco_runtime_tested=False)
    (cache/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return load_parts(cache/'parts.glb'), report


def validate_parts(parts):
    if not parts:
        raise ValueError('No collision parts')
    for name, mesh in parts.items():
        if not (mesh.is_volume and mesh.is_convex and mesh.is_winding_consistent):
            raise ValueError(f'Invalid collision hull: {name}')


def export_collision(source, directory, visual_obj, appearance=None, *, parts=None, formats=('obj', 'stl')):
    """Write separate hulls and static MJCF; OBJ is required by the XML."""
    formats = set(formats)
    if 'obj' not in formats or formats - {'obj', 'stl'}:
        raise ValueError('Collision export needs obj, with optional stl')
    if parts is None:
        parts, report = get_parts(source)
    else:
        report = {'construction': 'Explicit authored convex assembly'}
    validate_parts(parts)
    directory.mkdir(parents=True, exist_ok=True)
    model = ET.Element('mujoco', model=source.stem+'_collision')
    assets = ET.SubElement(model, 'asset')
    # Paths are relative to model.xml. The assembled visual OBJ remains one file.
    import os
    ET.SubElement(assets, 'mesh', name='visual', file=os.path.relpath(visual_obj, directory), inertia='convex')
    if appearance:
        texture = visual_obj.with_name(appearance['texture'])
        # MuJoCo converts OBJ's bottom-origin V coordinates on import. Flipping
        # the PNG as well samples the opposite rows (often empty palette cells).
        ET.SubElement(assets, 'texture', name='visual_texture', type='2d',
                      file=os.path.relpath(texture, directory), vflip='false')
        ET.SubElement(assets, 'material', name='visual_material', texture='visual_texture',
                      rgba='1 1 1 1', texuniform='false', specular='.2', shininess='.2')
    world = ET.SubElement(model, 'worldbody')
    body = ET.SubElement(world, 'body', name=source.stem)
    visual_attributes = {'material': 'visual_material'} if appearance else {'rgba': '.7 .72 .75 1'}
    ET.SubElement(body, 'geom', name='visual', type='mesh', mesh='visual', contype='0',
                  conaffinity='0', group='2', mass='0', **visual_attributes)
    errors = []
    for name, mesh in parts.items():
        obj, stl = directory/f'{name}.obj', directory/f'{name}.stl'
        obj.write_text(trimesh.exchange.obj.export_obj(mesh, include_color=False,
                       include_texture=False, include_normals=False, digits=10))
        if 'stl' in formats:
            mesh.export(stl)
        for path in ([obj, stl] if 'stl' in formats else [obj]):
            loaded = trimesh.load(path, force='mesh')
            if not (loaded.is_convex and loaded.is_volume):
                raise ValueError(f'Exported collision hull invalid: {path}')
            error = float(np.max(np.abs(loaded.bounds-mesh.bounds)))
            if error > max(1e-8, float(np.abs(mesh.vertices).max())*2e-7):
                raise ValueError(f'Export changed collision bounds: {path}')
            errors.append(error)
        ET.SubElement(assets, 'mesh', name=name, file=obj.name, inertia='convex')
        ET.SubElement(body, 'geom', name=name, type='mesh', mesh=name, group='3',
                      rgba='.3 .7 .9 0', contype='1', conaffinity='1', friction='.6 .005 .0001')
    write_xml(directory/'model.xml', model)
    report.update(convex_parts=len(parts), export_max_bounds_error=max(errors),
                  visual_appearance=appearance,
                  mujoco_runtime_tested=False, assembly='Static rigid body; no freejoint or calibrated inertia')
    (directory/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    (directory/'README.txt').write_text(
        'Each OBJ/STL pair is ONE convex collision hull. Keep every hull separate under one rigid body.\n'
        'model.xml assembles the OBJ hulls and the single coloured visual OBJ/PNG in MuJoCo (target 3.3+).\n'
        'Keep the complete object folder together: the visual mesh and PNG are referenced relatively.\n'
        'The original GLB is also included for colour-capable viewers; MuJoCo uses OBJ/PNG directly.\n'
        'The body is static. Add a freejoint and appropriate mass/inertia for dynamic use.\n'
        'Group 2: visual only. Group 3: transparent collision hulls. Friction is a placeholder.\n'
        'Units match the visual file; bird source units are NOT calibrated physical dimensions.\n'
        'This assembly has not been compiled or simulated in MuJoCo.\n')
    return report
