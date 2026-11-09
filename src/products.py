"""Codex-authored simple products: evidence -> parameters -> GLB + convex parts.

Run: .venv/bin/python -m src.products [--model all|container|electronic_balance|bowl]
The historical recipe key and CLI alias "scale" remain supported.
Uses the existing instrument pipeline's CPU renderer. Geometry uses millimetres
internally and exports metres, X right / Y rear / Z up, bottom at Z=0.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import trimesh

ROOT = Path(__file__).resolve().parents[1]
INPUTS = Path(__file__).with_name('product_inputs')
METAL = [175, 188, 197, 255]
WHITE = [229, 230, 223, 255]
BLUE = [24, 139, 184, 255]
DARK = [38, 43, 48, 255]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def rounded_loop(width, depth, radius, steps=4):
    points = []
    for sx, sy, start in [(1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)]:
        center = np.array([sx*(width/2-radius), sy*(depth/2-radius)])
        for angle in np.linspace(start, start+90, steps+1):
            points.append(center + radius*np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]))
    return np.array(points)


def circle(radius, sections=32):
    angles = np.arange(sections)*2*np.pi/sections
    return radius*np.c_[np.cos(angles), np.sin(angles)]


def at_z(loop, z):
    return np.c_[loop, np.full(len(loop), z)]


def hull(vertices, color=METAL):
    mesh = trimesh.convex.convex_hull(np.asarray(vertices)/1000)
    mesh.visual.face_colors = color
    return mesh


def prism(loop, z0, z1, color=METAL):
    return hull(np.vstack([at_z(loop, z0), at_z(loop, z1)]), color)


def block(lo, hi, color=METAL):
    return hull([[x, y, z] for x in [lo[0], hi[0]] for y in [lo[1], hi[1]] for z in [lo[2], hi[2]]], color)


def shell(rings, color):
    """Closed shell from bottom-outside up around the rim and down to floor."""
    count = len(rings[0]); vertices = np.vstack(rings).tolist(); faces = []
    for j in range(len(rings)-1):
        for i in range(count):
            a, b = j*count+i, j*count+(i+1)%count
            faces.extend([[a, b, b+count], [a, b+count, a+count]])
    for ring_index, reverse in [(0, True), (len(rings)-1, False)]:
        center = np.mean(rings[ring_index], axis=0)
        c = len(vertices); vertices.append(center.tolist())
        for i in range(count):
            tri = [c, ring_index*count+i, ring_index*count+(i+1)%count]
            faces.append(tri[::-1] if reverse else tri)
    mesh = trimesh.Trimesh(np.asarray(vertices)/1000, faces, process=True)
    mesh.fix_normals(); mesh.visual.face_colors = color
    assert mesh.is_volume
    return mesh


def wall_parts(prefix, outer0, outer1, inner0, inner1, z0, z1, color):
    parts = {}
    rings = [at_z(outer0, z0), at_z(outer1, z1), at_z(inner0, z0), at_z(inner1, z1)]
    for i in range(len(outer0)):
        j = (i+1) % len(outer0)
        parts[f'{prefix}_{i:02}'] = hull([p[k] for p in rings for k in [i, j]], color)
    return parts


def container(p):
    w, d, h = p['outer_mm']; iw, id_ = p['opening_mm']; t = p['wall_mm']; rt = p['rim_thickness_mm']
    bottom = rounded_loop(*p['bottom_mm'], p['bottom_corner_mm'])
    inner_bottom = rounded_loop(p['bottom_mm'][0]-2*t, p['bottom_mm'][1]-2*t, p['bottom_corner_mm']-t)
    mouth = rounded_loop(iw, id_, p['corner_radius_mm'])
    outer_mouth = rounded_loop(iw+2*t, id_+2*t, p['corner_radius_mm']+t)
    lip = rounded_loop(w, d, p['corner_radius_mm']+10)
    rings = [at_z(bottom, 0), at_z(bottom, t), at_z(outer_mouth, h-rt),
             at_z(lip, h-rt), at_z(lip, h), at_z(mouth, h), at_z(mouth, h-rt), at_z(inner_bottom, t)]
    visual = {'container_shell': shell(rings, METAL)}
    collision = {'floor': prism(bottom, 0, t)}
    collision.update(wall_parts('wall', bottom, outer_mouth, inner_bottom, mouth, t, h-rt, METAL))
    collision.update(wall_parts('rim', lip, lip, mouth, mouth, h-rt, h, METAL))
    return {'container_open': (visual, collision)}


def bowl(p):
    n = p['sections']; nc = p['collision_sections']; profile = np.asarray(p['profile_z_outer_inner_mm'])
    assert np.all(np.diff(profile[:, 0])>0) and np.all(profile[:, 1]>profile[:, 2])
    assert np.isclose(profile[-1, 0], p['height_mm']) and np.isclose(profile[-1, 1]*2, p['diameter_mm'])
    assert np.isclose(profile[0, 0], p['floor_mm'])
    foot_r, foot_i, underside = p['foot_outer_radius_mm'], p['foot_inner_radius_mm'], p['underside_mm']
    color = [245, 242, 247, 255]
    rings = [at_z(circle(foot_r, n), underside)]
    rings += [at_z(circle(ro, n), z) for z, ro, ri in profile]
    rings += [at_z(circle(ri, n), z) for z, ro, ri in profile[::-1]]
    body = shell(rings, color)
    # Color is visual only; the lip remains a simple thin solid.
    lip_faces = body.triangles_center[:, 2] > (p['height_mm']-1)/1000
    body.visual.face_colors[lip_faces] = [239, 217, 231, 255]
    foot = trimesh.creation.revolve(np.array([[foot_i, 0], [foot_r, 0], [foot_r, underside],
                                             [foot_i, underside], [foot_i, 0]])/1000, sections=n)
    foot.visual.face_colors = color
    # Cache smooth vertex normals so GLB and CPU previews agree on curved walls.
    _ = body.vertex_normals
    _ = foot.vertex_normals
    visual = {'bowl_shell': body, 'foot_ring': foot}
    collision = {'floor': hull(np.vstack([at_z(circle(foot_r, nc), underside),
                                          at_z(circle(profile[0, 1], nc), p['floor_mm'])]), color)}
    collision.update(wall_parts('foot', circle(foot_r, nc), circle(foot_r, nc), circle(foot_i, nc),
                                circle(foot_i, nc), 0, underside, color))
    for j, (lower, upper) in enumerate(zip(profile[:-1], profile[1:])):
        z0, ro0, ri0 = lower; z1, ro1, ri1 = upper
        collision.update(wall_parts(f'wall_band_{j}', circle(ro0, nc), circle(ro1, nc),
                                    circle(ri0, nc), circle(ri1, nc), z0, z1, color))
    return {'bowl': (visual, collision)}


def scale(p):
    w, d, h = p['body_mm']; loop = rounded_loop(w, d, p['corner_radius_mm'])
    z0 = p['housing_bottom_mm']; zt = p['housing_top_mm']; yf = -d/2; ye = p['slope_end_y_mm']
    # Extra stations preserve the breakpoint of the sloping control panel.
    upper = np.vstack([loop, [[-w/2, ye], [w/2, ye]]])
    z = np.minimum(zt, p['housing_front_height_mm'] + (upper[:, 1]-yf)/(ye-yf)*(zt-p['housing_front_height_mm']))
    body = hull(np.vstack([at_z(loop, z0), np.c_[upper, z]]), WHITE)
    collision = {'housing': body}
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            foot = circle(10, 12)+[sx*74, sy*112]
            collision[f'foot_{sx}_{sy}'] = prism(foot, 0, p['foot_height_mm'], DARK)
    cy = p['pan_center_y_mm']
    collision['pan_support'] = prism(circle(24, 16)+[0, cy], zt, p['pan_bottom_mm'])
    result = {}
    for variant in ['square', 'round']:
        parts = dict(collision)
        pan_loop = rounded_loop(*p['square_pan_mm'], 7) if variant=='square' else circle(p['round_pan_diameter_mm']/2)
        parts['weighing_pan'] = prism(pan_loop+[0, cy], p['pan_bottom_mm'], h, BLUE if variant=='square' else METAL)
        visual = dict(parts)
        # Flush, flat visual markings never enter collision geometry.
        slope = (zt-p['housing_front_height_mm'])/(ye-yf)
        def panel(name, xa, xb, ya, yb, color):
            offset = .12 if name=='control_panel' else .24
            vertices = [[x, y, p['housing_front_height_mm']+(y-yf)*slope+offset]
                        for x, y in [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]]
            q = trimesh.Trimesh(np.array(vertices)/1000, [[0, 1, 2], [0, 2, 3]], process=False)
            q.visual.face_colors = color; visual[name] = q
        panel('control_panel', -78, 78, -126, -54, [181, 188, 193, 255])
        panel('display', -57, 57, -89, -59, [16, 57, 156, 255])
        for i, x in enumerate([-58, -29, 0, 29, 58]):
            panel(f'button_{i}', x-7, x+7, -118, -102, [48, 119, 175, 255] if i!=4 else [225, 155, 52, 255])
        result[f'electronic_balance_{variant}'] = (visual, parts)
    return result


def export_scene(parts, path, role):
    scene = trimesh.Scene()
    for name, mesh in parts.items():
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    scene.metadata = {'units': 'metres', 'axes': 'X right, Y rear, Z up', 'role': role,
                      'collision_usage': ('Each named geometry is one convex shape; preserve nodes. Do not hull the whole assembly.'
                                          if role!='visual' else 'Appearance mesh; use the separate collision GLB for convex physics shapes.')}
    scene.export(path)


def validate(parts, path, convex=False):
    loaded = trimesh.load(path, force='scene')
    assert set(loaded.geometry)==set(parts), path
    rows = []
    for name, mesh in loaded.geometry.items():
        assert np.isfinite(mesh.vertices).all() and mesh.area>0, name
        assert np.allclose(mesh.bounds, parts[name].bounds, atol=1e-7), name
        if convex:
            assert mesh.is_volume and mesh.is_winding_consistent and mesh.is_convex, name
        rows.append({'name': name, 'triangles': len(mesh.faces), 'watertight': bool(mesh.is_watertight), 'convex': bool(mesh.is_convex)})
    vertices = np.vstack([m.vertices for m in loaded.geometry.values()])
    return {'bounds_mm': (loaded.bounds*1000).round(4).tolist(), 'triangles': sum(x['triangles'] for x in rows),
            'geometry_count': len(rows), 'parts': rows, 'vertices': vertices}


def symmetry(vertices, axes):
    tree = cKDTree(vertices); errors = {}
    for axis in axes:
        reflected = vertices.copy(); reflected[:, axis] *= -1
        err = float(tree.query(reflected)[0].max())
        assert err < 1e-7, (axis, err)
        errors['xyz'[axis]+'_reflection_max_error_m'] = err
    return errors


def probe_collision(parts, points):
    # Half-space containment is robust for these small convex solids, without rtree.
    from scipy.spatial import ConvexHull
    points = np.asarray(points)/1000
    inside = np.zeros(len(points), dtype=bool)
    for mesh in parts.values():
        equations = ConvexHull(mesh.vertices).equations
        inside |= np.all(points@equations[:, :3].T+equations[:, 3] <= 1e-9, axis=1)
    return inside


def review(parts, path, title, smooth=False):
    from .instruments.render import Renderer
    renderer = Renderer(parts, smooth=smooth)
    vertices = np.vstack([m.vertices for m in parts.values()])*1000
    views = [(np.array([1., 0, 0]), np.array([0., 0, 1]), 'Front'),
             (np.array([1., 0, 0]), np.array([0., 1, 0]), 'Top')]
    camera = np.array([1., -1.5, 1.2]); camera /= np.linalg.norm(camera)
    u = np.cross([0, 0, 1], camera); u /= np.linalg.norm(u); v = np.cross(camera, u)
    views.insert(0, (u, v, 'Perspective'))
    canvas = Image.new('RGB', (1500, 560), '#eef2f5'); draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), title, fill='#172738')
    for i, (u, v, label) in enumerate(views):
        x, y = vertices@u, vertices@v
        span = max(np.ptp(x), np.ptp(y))*1.15
        cx, cy = (x.max()+x.min())/2, (y.max()+y.min())/2
        rgba = Image.fromarray(renderer.image(u, v, np.cross(u, v), [cx-span/2, cx+span/2, cy-span/2, cy+span/2], width=480))
        canvas.paste(rgba, (i*500+10, 50), rgba); draw.text((i*500+20, 535), label, fill='#172738')
    canvas.save(path)


def manifest(model):
    entries = json.loads((INPUTS/f'{model}_sources.json').read_text())
    for item in entries:
        path = ROOT/item['path']
        item['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    from .helpers import dataset_input_dir
    hint = dataset_input_dir(model)/'New Text Document.txt'
    return {'selection': entries, 'hint_text': hint.read_text() if hint.exists() else None,
            'originals_preserved': True, 'selection_method': 'Manual review of contact sheets; unrelated variants and redundant marketing images excluded.'}


def bowl_reference_review(parts, output):
    """Photo alongside a similar-elevation render; this is not calibrated fitting."""
    from .instruments.render import Renderer
    vertices = np.vstack([m.vertices for m in parts.values()])*1000
    angle = np.deg2rad(15)
    camera = np.array([0, -np.cos(angle), np.sin(angle)])
    u = np.array([1., 0, 0]); v = np.cross(camera, u)
    roll = np.deg2rad(7)
    u, v = np.cos(roll)*u-np.sin(roll)*v, np.sin(roll)*u+np.cos(roll)*v
    x, y = vertices@u, vertices@v
    cx, cy = (x.min()+x.max())/2, (y.min()+y.max())/2
    rgba = Image.fromarray(Renderer(parts, smooth=True).image(u, v, camera, [cx-71, cx+71, cy-52.5, cy+52.5], width=775))
    with Image.open(ROOT/'dataset/hestia-rl/hestia_bowl_v1/20261013_153452_094806_rgb.png') as original:
        photo = original.crop((460, 370, 770, 600)).resize((775, 575))
    canvas = Image.new('RGB', (1590, 650), '#eef2f5'); draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), 'Captured bowl: curved heel, narrow foot, thin rim', fill='#172738')
    draw.text((810, 15), 'Revised symmetric mesh - similar camera elevation', fill='#172738')
    canvas.paste(photo, (10, 50)); canvas.paste(rgba, (805, 50), rgba)
    draw.text((20, 630), 'Shape comparison only; diameter and height remain provisional RGB-D estimates.', fill='#172738')
    canvas.save(output/'reference_comparison.png')


def build(model, params, output, parameter_source=None):
    output.mkdir(parents=True, exist_ok=True)
    variants = {'container': container, 'scale': scale, 'bowl': bowl}[model](params)
    display_name = 'electronic_balance' if model == 'scale' else model
    report = {'model': display_name, 'units': 'metres', 'parameters_mm': params, 'variants': {},
              'physical_contact_validated': False, 'mass_friction_restitution': 'Not estimated; configure in simulator.'}
    write_json(output/'parameters.json', params)
    write_json(output/'input_manifest.json', manifest(model))
    if model=='bowl':
        measurement = ROOT/'work/hestia-rl/hestia_bowl_v1/codex/references/bowl_measurement.json'
        if measurement.exists():
            write_json(output/'measurement.json', json.loads(measurement.read_text()))
    for name, (visual, collision) in variants.items():
        export_scene(visual, output/f'{name}.glb', 'visual')
        export_scene(collision, output/f'{name}_collision.glb', 'convex compound collision')
        vr = validate(visual, output/f'{name}.glb'); vr.pop('vertices')
        cr = validate(collision, output/f'{name}_collision.glb', convex=True)
        sym = symmetry(cr.pop('vertices'), [0] if model=='scale' else [0, 1])
        assert abs(cr['bounds_mm'][0][2]) < 1e-5
        checks = {}
        if name in ['container_open', 'bowl']:
            h = params['outer_mm'][2] if model=='container' else params['height_mm']
            opening = min(params['opening_mm'])/2 if model=='container' else params['diameter_mm']/2-params['wall_mm']
            probes = [[0, 0, z] for z in np.linspace(5, h+10, 30)]
            probes += [[r*np.cos(a), r*np.sin(a), h-1] for r in [opening*.5, opening*.9] for a in np.linspace(0, 2*np.pi, 32, endpoint=False)]
            assert not probe_collision(collision, probes).any(), name
            floor_probe = .5 if model=='container' else (params['underside_mm']+params['floor_mm'])/2
            assert probe_collision(collision, [[0, 0, floor_probe]])[0]
            checks = {'interior_and_entry_probes_free': len(probes), 'floor_probe_solid': True}
            visual_volume = sum(m.volume for m in visual.values())
            collision_volume = sum(m.volume for m in collision.values())
            assert np.isclose(visual_volume, collision_volume, rtol=.01 if model=='bowl' else 1e-8)
            checks['visual_collision_volume_relative_error'] = abs(visual_volume-collision_volume)/visual_volume
        if name=='bowl':
            vertices = np.vstack([m.vertices for m in collision.values()])
            angle = 2*np.pi/params['collision_sections']
            rotation = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])[:3, :3]
            error = float(cKDTree(vertices).query(vertices@rotation.T)[0].max())
            assert error < 1e-7
            checks['rotational_symmetry_max_error_m'] = error
            checks['maximum_radial_chord_error_mm'] = params['diameter_mm']/2*(1-np.cos(np.pi/params['collision_sections']))
            assert not probe_collision(collision, [[0, 0, params['underside_mm']/2]])[0]
            assert probe_collision(collision, [[(params['foot_outer_radius_mm']+params['foot_inner_radius_mm'])/2, 0, .5]])[0]
            checks['raised_foot_and_underside_clearance'] = True
        if model=='scale':
            expected = np.array(params['body_mm'])
            assert np.allclose((np.array(cr['bounds_mm'][1])-cr['bounds_mm'][0]), expected, atol=.001)
            assert probe_collision(collision, [[0, params['pan_center_y_mm'], expected[2]-1]])[0]
            checks['pan_contact_probe_solid'] = True
        report['variants'][name] = {'visual': vr, 'collision': cr, 'symmetry': sym, 'contact_checks': checks}
        review(visual, output/f'{name}_preview.png', f'{name} | metres / Z up | {vr["triangles"]} visual triangles', smooth=model=='bowl')
        review(collision, output/f'{name}_collision_preview.png', f'{name} | {cr["geometry_count"]} convex collision pieces')
    write_json(output/'validation.json', report)
    note = ('Load visual GLB for appearance and collision GLB as a compound of its individually named convex meshes. '
            'Preserve node separation: one convex hull of the entire bowl/container fills the opening. '
            'All coordinates are metres, X right / Y rear / Z up, bottom at Z=0. ')
    if model=='container':
        note += 'Open container only. Visual and collision meshes have the same boundary; collision is split into convex pieces. '
    if model=='bowl':
        note += 'The curved bowl uses 48 visual sectors and 24 collision sectors (maximum radial chord error 0.514 mm), with a raised foot and thin flared lip. '
    if model=='scale': note += 'Electronic balance markings are visual only. '
    note += ('No cables, food, promotional props, or decorative microgeometry. '
             'Contact geometry checks passed, but dynamics and real-world contact accuracy remain unmeasured.')
    (output/'README.txt').write_text(note+'\n\n'+params['evidence']+'\n')
    sections = []
    if model=='bowl':
        bowl_reference_review(variants['bowl'][0], output)
        sections.append('<section><h2>Reference comparison</h2><img src="reference_comparison.png" alt="Captured bowl and revised symmetric mesh"></section>')
    for name in variants:
        sections.append(f'<section><h2>{name}</h2><a href="{name}.glb">Visual GLB</a> · <a href="{name}_collision.glb">Collision GLB</a><br><img src="{name}_preview.png"><details><summary>Collision geometry</summary><img src="{name}_collision_preview.png"></details></section>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+display_name+'</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:32px auto;padding:20px;background:#eef2f5;color:#172738}img{max-width:100%}section{background:white;padding:20px;margin:20px 0}a{color:#135e91}</style><h1>'+display_name+'</h1><p>'+html.escape(note)+'</p><p>'+html.escape(params['evidence'])+'</p><a href="parameters.json">Parameters</a> · <a href="validation.json">Validation</a> · <a href="input_manifest.json">Selected evidence</a>'+''.join(sections))
    if model=='container':
        # User explicitly removed the lid; remove only known retired exports.
        for stem in ['container_lid', 'container_closed']:
            for suffix in ['.glb', '_collision.glb', '_preview.png', '_collision_preview.png']:
                (output/(stem+suffix)).unlink(missing_ok=True)
    from .assets import save_manifest
    save_manifest(output/'asset.json', kind='recipe',
                  sources=[Path(__file__), parameter_source or output/'parameters.json'],
                  variants={name: {'visual': output/f'{name}.glb',
                                   'collision': output/f'{name}_collision.glb',
                                   'physics': INPUTS/'mujoco_physics.json', 'physics_key': model}
                            for name in variants}, units='m', frame='X right, Y rear, Z up; support Z=0',
                  rebuild=['python', '-m', 'src.products', '--model', display_name,
                           '--parameters', str(parameter_source or INPUTS/'designs.json')])
    return report


def review_products(destinations):
    """Refresh product review pages from the current models."""
    package = ROOT/'output/comparisons/products'; package.mkdir(exist_ok=True)
    images = [destinations['container']/'container_open_preview.png', destinations['scale']/'electronic_balance_square_preview.png',
              destinations['scale']/'electronic_balance_round_preview.png', destinations['bowl']/'bowl_preview.png']
    canvas = Image.new('RGB', (1500, 560*len(images)), 'white')
    for i, path in enumerate(images):
        with Image.open(path) as im: canvas.paste(im, (0, i*560))
    canvas.save(package/'all_products.png')
    links = ''.join(f'<li><a href="{Path("../..").joinpath(directory.relative_to(ROOT/"output"))}/index.html">{("electronic balance" if model == "scale" else model)}</a></li>' for model, directory in destinations.items())
    (package/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Collision products</title><style>body{font:18px/1.6 system-ui;max-width:1000px;margin:40px auto}img{max-width:100%}</style><h1>Simple collision products</h1><a href="../../packages/">Previously exported delivery snapshots</a><ul>'+links+'</ul><p>Symmetric geometry, metres, Z up. Keep individual convex meshes separate when importing collision GLBs.</p><img src="all_products.png" alt="Open container, both electronic balance variants, and revised bowl">')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['all', 'container', 'scale', 'bowl'], default='all',
                        type=lambda name: {'electronic_balance': 'scale', 'sand_container': 'container', 'sand_bucket': 'container',
                                           'hestia_bowl_v1': 'bowl', 'hestia_bowl': 'bowl'}.get(name, name))
    parser.add_argument('--parameters', type=Path, default=INPUTS/'designs.json')
    args = parser.parse_args()
    params = json.loads(args.parameters.read_text())
    destinations = {'container': ROOT/'output/hestia-rl/sand_container', 'scale': ROOT/'output/hestia-rl/electronic_balance',
                    'bowl': ROOT/'output/hestia-rl/hestia_bowl_v1'}
    names = list(destinations) if args.model=='all' else [args.model]
    for model in names:
        report = build(model, params[model], destinations[model], parameter_source=args.parameters)
        print(model, {k: {'triangles': v['visual']['triangles'], 'convex_parts': v['collision']['geometry_count']} for k, v in report['variants'].items()}, flush=True)
    if all((directory/'validation.json').exists() for directory in destinations.values()):
        review_products(destinations)
    from .output_layout import navigation
    navigation(ROOT/'output')


if __name__=='__main__':
    main()
