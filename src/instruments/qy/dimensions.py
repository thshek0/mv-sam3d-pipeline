"""Dimension sheets from the actual C13534 URDF meshes, in the closed pose.

Run with .venv/bin/python. Does not regenerate or modify the accepted geometry.
Dimensions are measured from transformed vertices; evidence is recorded separately.
"""
from pathlib import Path
import hashlib
import html
import json
import os
import xml.etree.ElementTree as ET

os.environ.setdefault('MPLCONFIGDIR', '/tmp/c13534_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image
import trimesh

from .urdf import xml_fk

from ..paths import ROOT
OUT = ROOT / 'output/objects/hkqai-instruments/c13534/codex/intermediate/.build_work'
PACKAGE = OUT / 'urdf'
DEST = OUT / 'dimensions'
DRAWING = ROOT / 'dataset/hkqai-instruments/c13534/online_sources/pdf_images/qy_dimension_drawing.png'
COLORS = dict(M='#1967b4', U='#087e62', E='#a85e10', G='#566674')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def origin(element):
    if element is None:
        return np.eye(4)
    t = trimesh.transformations.euler_matrix(*np.fromstring(element.get('rpy', '0 0 0'), sep=' '))
    t[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    return t


def read_parts():
    robot = ET.parse(PACKAGE / 'robot.urdf').getroot()
    fk = xml_fk(robot, {})
    parts = {}
    inputs = [PACKAGE / 'robot.urdf']
    for link in robot.findall('link'):
        collisions = {c.get('name'): c for c in link.findall('collision')}
        for v in link.findall('visual'):
            name = v.get('name')
            m = v.find('geometry/mesh')
            path = PACKAGE / m.get('filename')
            inputs.append(path)
            mesh = trimesh.load(path, force='mesh', process=False)
            mesh.apply_scale(np.fromstring(m.get('scale', '1 1 1'), sep=' '))
            mesh.apply_transform(fk[link.get('name')] @ origin(v.find('origin')))
            # The drawings must also represent the collision geometry, not just visuals.
            c = collisions[name]
            assert m.attrib == c.find('geometry/mesh').attrib
            assert np.allclose(origin(v.find('origin')), origin(c.find('origin')), atol=1e-12)
            rgba = np.fromstring(v.find('material/color').get('rgba'), sep=' ')
            mesh.visual.face_colors = np.rint(rgba * 255).astype(np.uint8)
            parts[name] = mesh
    accepted = trimesh.load(OUT / 'meshes/collision_closed.glb', force='scene')
    assert set(parts) == set(accepted.graph.nodes_geometry)
    errors = []
    for name, mesh in parts.items():
        t, g = accepted.graph[name]
        vertices = trimesh.transform_points(accepted.geometry[g].vertices, t)
        assert vertices.shape == mesh.vertices.shape
        assert np.array_equal(accepted.geometry[g].faces, mesh.faces)
        errors.append(float(np.linalg.norm(vertices - mesh.vertices, axis=1).max()))
    assert max(errors) < 1e-7, 'URDF differs from accepted closed GLB'
    return parts, inputs, max(errors)


def bounds(parts, names):
    vertices = np.vstack([parts[n].vertices for n in names]) * 1000
    return np.array([vertices.min(axis=0), vertices.max(axis=0)])


def measure(parts):
    p = json.loads((OUT / 'collision_doors.json').read_text())['parameters']
    user = json.loads((OUT / 'confirmed_measurements.json').read_text())
    research = json.loads((ROOT / 'dataset/hkqai-instruments/c13534/measurements.json').read_text())['manufacturer']
    published = research['catalog_dimensions_and_proxy_caveats']['c13534']
    body_names = [n for n in parts if n in {
        'right_housing', 'rear_housing', 'base_support', 'tower_middle_band',
        'tower_rear_wall', 'chamber_floor'} or n.startswith(('upper_shell__', 'lower_shell__', 'control_'))]
    b = bounds(parts, body_names)
    all_bounds = bounds(parts, parts)
    tower = bounds(parts, [n for n in body_names if n not in {'right_housing', 'rear_housing', 'base_support'}])
    door = bounds(parts, [n for n in body_names if n.startswith('upper_')])
    strip = bounds(parts, [n for n in body_names if n.startswith('control_')])
    strip_vertices = np.vstack([q.vertices for n,q in parts.items() if n.startswith('control_frame__')]) * 1000
    strip_right_front_y = float(strip_vertices[np.isclose(strip_vertices[:, 0], strip[1, 0]), 1].min())
    strip_left_front_y = float(strip_vertices[np.isclose(strip_vertices[:, 0], strip[0, 0]), 1].min())
    # Recover the outer curved profile from vertices, not from its parameter values.
    vertices = np.vstack([parts[n].vertices for n in parts if n.startswith('lower_shell__')]) * 1000
    xs = np.unique(np.round(vertices[:, 0], 4))
    profile = np.array([(x, vertices[np.isclose(vertices[:, 0], x, atol=0.0002), 1].min()) for x in xs])
    reference_x = p['door_depth_reference_x_m'] * 1000  # Approximate ruler location from photo.
    local_front_y = float(np.interp(reference_x, profile[:, 0], profile[:, 1]))
    rows = []

    def add(key, label, actual, kind, target=None, note=''):
        error = None if target is None else float(actual - target)
        row = dict(key=key, label=label, model_mm=float(actual), evidence=kind,
                   target_mm=target, difference_mm=error, note=note)
        if target is not None:
            row['nominal_constraint_pass'] = abs(error) < .001
            # Parameter edits can intentionally depart from measured/published evidence.
            # Keep the real evidence and report the difference instead of relabeling it.
        rows.append(row)

    for i, key in enumerate(('width', 'depth', 'height')):
        add('body_' + key, 'Enclosure ' + key, b[1, i] - b[0, i], 'M', published['overall_wdh_mm'][i],
            'Published dimension applied to enclosure/support; projecting hardware measured separately.')
    add('lower_housing_top', 'Lower housing top above table', parts['right_housing'].bounds[1, 2]*1000,
        'M', published['lower_top_height_mm'])
    # These additional labels are directly readable in the manufacturer dimension drawing.
    base = bounds(parts, ['base_support'])
    add('base_height', 'Base/support height', base[1, 2] - base[0, 2], 'M', 33)
    add('base_width', 'Base/support width', base[1, 0] - base[0, 0], 'M', 596)
    add('base_depth', 'Base/support depth', base[1, 1] - base[0, 1], 'M', 470)
    add('left_hardware_projection', 'Left hinge projection beyond enclosure', -all_bounds[0,0], 'M', 11,
        'PDF page 8 dimension; hinge axis follows projection minus modeled pin radius.')
    for key, label, value, field in [
        ('tower_width', 'Tower width', tower[1, 0]-tower[0, 0], 'tower_including_control_strip_mm'),
        ('door_width', 'Door projected width', door[1, 0]-door[0, 0], 'door_projected_width_mm'),
        ('strip_width', 'Control strip width', strip[1, 0]-strip[0, 0], 'control_strip_width_mm')]:
        add(key, label, value, 'U', user['widths'][field],
            'User width allowance +/-5 mm due to partly open door; not a manufacturing tolerance.')
    add('strip_depth', 'Strip depth at right edge', strip[1, 1]-strip_right_front_y, 'U', user['depths']['right_side_mm'],
        'Local span at the outer right edge; not the maximum bounding-box depth of the tapered strip.')
    add('door_local_depth', 'Door-region local depth', tower[1, 1]-local_front_y, 'U', user['depths']['door_region_mm'],
        'At approximate ruler X=211.5 mm. Not maximum depth, arc length or curve radius; depth uncertainty unspecified.')
    for level in ['upper', 'lower']:
        q = bounds(parts, [n for n in body_names if n.startswith(level + '_')])
        add(level + '_door_height', level.capitalize() + ' door height', q[1, 2]-q[0, 2], 'E',
            note='Image/model estimate; no confirmed ruler reading.')
    for i, key in enumerate(('width', 'depth', 'height')):
        add('complete_' + key, 'Closed extent including hardware: ' + key,
            all_bounds[1, i]-all_bounds[0, i], 'G', note='Generated extent, not independent evidence of the real hardware envelope.')
    add('tower_max_depth', 'Tower maximum body depth', tower[1, 1]-tower[0, 1], 'G',
        note='Generated curvature extent; do not confuse with the measured local 280 mm span.')
    add('door_strip_front_step', 'Door/strip front depth step', abs(strip_left_front_y-profile[-1, 1]), 'G',
        note='User correction: no artificial front-to-back step at the closed seam. Detailed seal and tangent curvature remain unmeasured.')
    if p.get('control_strip_front_shape') == 'tapered_flush':
        assert abs(strip_left_front_y-profile[-1, 1]) < .001
    return dict(rows=rows, body=b, complete=all_bounds, tower=tower, door=door,
                strip=strip, strip_right_front_y=strip_right_front_y,
                profile=profile, local_front_y=local_front_y, reference_x=reference_x)


class Renderer:
    def __init__(self, parts):
        self.mesh = trimesh.util.concatenate(list(parts.values()))
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(self.mesh.vertices, np.float32)),
                                 o3d.core.Tensor(np.asarray(self.mesh.faces, np.uint32)))

    def image(self, u, v, camera, extent, width=1600):
        # Image edge coordinates are the exact millimetre coordinates used by annotations.
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
        ids = hit['primitive_ids'].numpy()
        valid = np.isfinite(hit['t_hit'].numpy())
        tri = ids[valid].astype(int)
        normals = hit['primitive_normals'].numpy()[valid]
        light = camera + u*.35 + v*.65
        light = light/np.linalg.norm(light)
        shade = .55 + .45*np.clip(normals @ light, 0, 1)
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[valid, :3] = np.clip(self.mesh.visual.face_colors[tri, :3]*shade[:, None], 0, 255)
        rgba[valid, 3] = 255
        # Silhouette only: preserve geometric surface shading, avoid fake part seams.
        interior = valid.copy()
        interior[1:, :] &= valid[:-1, :]
        interior[:-1, :] &= valid[1:, :]
        interior[:, 1:] &= valid[:, :-1]
        interior[:, :-1] &= valid[:, 1:]
        rgba[valid & ~interior, :3] = (57, 66, 73)
        return rgba


def label(ax, x, y, s, kind='G', **kwargs):
    return ax.text(x, y, s, color=COLORS[kind], ha='center', va='center', zorder=5,
                   bbox=dict(facecolor='white', edgecolor='none', alpha=.94, pad=1.8), **kwargs)


def dimension(ax, a, b, offset, text, kind, vertical=False, fontsize=11):
    """a,b are the measured 2D endpoints; offset is the dimension-line coordinate."""
    c = COLORS[kind]
    if vertical:
        for pt in [a, b]:
            ax.plot([pt[0], offset], [pt[1], pt[1]], color=c, lw=.7, alpha=.65)
        aa, bb = (offset, a[1]), (offset, b[1])
        tx, ty = offset-9, (a[1]+b[1])/2
    else:
        for pt in [a, b]:
            ax.plot([pt[0], pt[0]], [pt[1], offset], color=c, lw=.7, alpha=.65)
        aa, bb = (a[0], offset), (b[0], offset)
        tx, ty = (a[0]+b[0])/2, offset+11
    ax.annotate('', xy=aa, xytext=bb, arrowprops=dict(arrowstyle='<->', color=c, lw=1.1,
                shrinkA=0, shrinkB=0, mutation_scale=11), zorder=4)
    label(ax, tx, ty, text+' ['+kind+']', kind, rotation=90 if vertical else 0, fontsize=fontsize)


def sheet(renderer, name, title, orientation, u, v, camera, extent, limits, annotate, notes):
    fig = plt.figure(figsize=(10.6, 9.4), facecolor='white')
    fig.text(.055, .96, 'C13534  /  '+title, fontsize=19, weight='bold', color='#203647', va='top')
    fig.text(.055, .92, 'Actual URDF geometry • doors closed • selector IN • millimetres', fontsize=10.8, color='#526575')
    ax = fig.add_axes([.07, .20, .86, .68])
    ax.imshow(renderer.image(u, v, camera, extent), extent=extent, origin='upper', interpolation='nearest')
    ax.set(xlim=limits[:2], ylim=limits[2:], aspect='equal')
    ax.axis('off')
    annotate(ax)
    fig.text(.055, .178, orientation, fontsize=10.5, color='#526575')
    for x, kind, text in [(.055, 'M', 'Published'), (.245, 'U', 'Your measurement'),
                           (.505, 'E', 'Estimate'), (.705, 'G', 'Generated extent')]:
        fig.text(x, .14, '['+kind+'] '+text, fontsize=10.3, color=COLORS[kind], weight='bold')
    for i, note in enumerate(notes):
        fig.text(.055, .105-i*.025, note, fontsize=9.4, color='#3d4e5c')
    fig.savefig(DEST / (name+'.png'), dpi=180)
    plt.close(fig)
    print('Rendered '+name+'.png', flush=True)


def render_sheets(parts, m):
    renderer = Renderer(parts)
    values = {r['key']: r['model_mm'] for r in m['rows']}
    b, a = m['body'], m['complete']
    w, d, h = b[1]-b[0]
    tw, dw = m['tower'][1, 0], m['door'][1, 0]
    strip_front, tower_rear = m['strip_right_front_y'], m['strip'][1, 1]
    f = lambda key: f'{values[key]:.0f}'
    common = ['Dimensions come from exported vertices; nominal agreement does not validate all contact surfaces.',
              'Handles, hinges, wall thickness and internal contact dimensions still require measurement.']

    def front(ax):
        dimension(ax, (0, 0), (w, 33), -30, f('body_width')+' body', 'M')
        dimension(ax, (a[0, 0], 83), (a[1, 0], 33), -70, f"{values['complete_width']:.1f} with hardware", 'G')
        dimension(ax, (0, 0), (0, h), -52, f('body_height'), 'M', True)
        dimension(ax, (w, 0), (w, values['lower_housing_top']), w+45, f('lower_housing_top'), 'M', True)
        dimension(ax, (0, h), (dw, h), h+35, f('door_width')+' door', 'U')
        dimension(ax, (dw, h), (tw, h), h+35, f('strip_width')+' strip', 'U', fontsize=10)
        dimension(ax, (0, h), (tw, h), h+79, f('tower_width')+' tower', 'U')
        for level in ['upper', 'lower']:
            q = bounds(parts,[n for n in parts if n.startswith(level+'_shell__')])
            dimension(ax, (110, q[0, 2]), (110, q[1, 2]), 110, f(level+'_door_height'), 'E', True, 10)
        label(ax, 496, 190, 'Right housing', 'G', fontsize=10)

    sheet(renderer, 'front', 'Front view', '+X right • +Z up • viewing from the handle/selector side',
          (1,0,0), (0,0,1), (0,-1,0), (-15,680,-5,545), (-95,755,-100,645), front,
          ['Width readings: nominal 340 = 225 + 115; your allowance is ±5 mm (partly open door).']+common)

    def top(ax):
        dimension(ax, (0, d), (w, d), d+38, f('body_width'), 'M')
        dimension(ax, (w, 0), (w, d), w+42, f('body_depth')+' body', 'M', True)
        dimension(ax, (a[0, 0], a[0, 1]), (a[0, 0], a[1, 1]), -59,
                  f"{values['complete_depth']:.1f} with hardware", 'G', True)
        dimension(ax, (0, 0), (dw, 0), -44, f('door_width'), 'U')
        dimension(ax, (dw, 16), (tw, 16), -44, f('strip_width'), 'U')
        dimension(ax, (0, 0), (tw, 16), -84, f('tower_width')+' tower', 'U')
        x = m['reference_x']
        dimension(ax, (x, m['local_front_y']), (x, tower_rear), x, f('door_local_depth')+' local', 'U', True)
        dimension(ax, (tw, strip_front), (tw, tower_rear), tw+36, f('strip_depth')+' right edge', 'U', True)
        label(ax, 510, 120, 'FRONT', 'G', fontsize=11)
        label(ax, 500, 470, 'REAR', 'G', fontsize=11)
        label(ax, 78, 145, 'Door top', 'G', fontsize=10)

    sheet(renderer, 'top', 'Top view', '+X right • +Y toward rear (up on this sheet) • front at bottom',
          (1,0,0), (0,1,0), (0,0,1), (-15,680,-20,580), (-100,755,-112,650), top,
          ['280 is local at approximate ruler X = 211.5; 267 is at the strip RIGHT edge (X = 340).',
           f"The curved tower's generated maximum body depth is {values['tower_max_depth']:.1f}; 280 is not its maximum.",
           'Width allowance ±5 mm. Depth uncertainty, curvature and hardware projections remain unverified.'])

    def side(ax, left):
        sign = -1 if left else 1
        front, rear = 0, sign*d
        dimension(ax, (front, 21), (rear, 33), -31, f('body_depth')+' body', 'M')
        dimension(ax, (sign*a[0, 1], 105), (sign*a[1, 1], 33), -72,
                  f"{values['complete_depth']:.1f} with hardware", 'G')
        xheight = min(front, rear)-42
        dimension(ax, (rear, 0), (sign*tower_rear, h), xheight, f('body_height'), 'M', True)
        dimension(ax, (sign*strip_front, h), (sign*tower_rear, h), h+40, f('strip_depth')+' right edge', 'U')
        label(ax, sign*80, -105, 'FRONT', 'G')
        label(ax, sign*495, -105, 'REAR', 'G')

    for left in [True, False]:
        sign = -1 if left else 1
        name = 'left_side' if left else 'right_side'
        sheet(renderer, name, 'Left side' if left else 'Right side',
              ('Front is on the RIGHT • looking at the straight left door wall' if left else
               'Front is on the LEFT • looking at the control-strip side'),
              (0,sign,0), (0,0,1), (sign,0,0),
              (-580,20,-5,545) if left else (-20,580,-5,545),
              (-660,100,-130,630) if left else (-100,660,-130,630),
              lambda ax, left=left: side(ax, left),
              ['267 labels the strip right-edge depth; its tapered front meets the door without a depth step.']+common)

    def rear(ax):
        dimension(ax, (-w, 33), (0, 33), -33, f('body_width'), 'M')
        dimension(ax, (0, 0), (0, h), 51, f('body_height'), 'M', True)
        dimension(ax, (-w, 0), (-w, values['lower_housing_top']), -w-45, f('lower_housing_top'), 'M', True)
        base = parts['base_support'].bounds*1000
        dimension(ax, (-base[1, 0], 0), (-base[0, 0], 0), -75, f('base_width')+' support', 'M')
        label(ax, -410, 200, 'Simplified rear housing', 'G', fontsize=10)

    sheet(renderer, 'rear', 'Rear view', 'Model +X is LEFT on this sheet • +Z up • looking from the rear',
          (-1,0,0), (0,0,1), (0,1,0), (-680,15,-5,545), (-755,95,-110,635), rear,
          ['Rear connectors, fans and vents are not represented in this collision prototype.']+common)


def overview():
    files = ['front', 'left_side', 'top', 'right_side']
    thumb = (1378, 1222)
    canvas = Image.new('RGB', (thumb[0]*2, thumb[1]*2), 'white')
    for i, name in enumerate(files):
        im = Image.open(DEST / (name+'.png')).convert('RGB').resize(thumb, Image.Resampling.LANCZOS)
        canvas.paste(im, ((i%2)*thumb[0], (i//2)*thumb[1]))
    canvas.save(DEST / 'overview.png')


def report(measurements, validation, hashes):
    rows = measurements['rows']
    payload = dict(units='mm', pose='closed; handles seated; selector IN',
                   geometry='URDF visual and collision meshes, transformed through XML joint hierarchy',
                   source_sha256=hashes, checks=validation, measurements=rows,
                   evidence_codes=dict(M='Manufacturer drawing nominal', U='User ruler transcription',
                                       E='Unconfirmed image/model estimate', G='Computed model extent, not measured hardware'),
                   body_bounds_mm=measurements['body'].tolist(),
                   complete_bounds_mm=measurements['complete'].tolist(),
                   body_vs_overall='The published 670/573/537 dimension lines are applied to the body/support. Estimated projecting hardware is separately bounded.',
                   unmodeled_drawing_details='The drawing also shows a 12 mm top projection on the low housing, fans/ports and a different support outline; these are not detailed in this simplified collider.',
                   ruler_x_mm=measurements['reference_x'], ruler_x_status='Approximate photo-based location, not a ruler-confirmed X coordinate',
                   physical_contact_accuracy_validated=False)
    (DEST / 'measurements.json').write_text(json.dumps(payload, indent=2)+'\n')
    table = []
    for r in rows:
        target = '—' if r['target_mm'] is None else f"{r['target_mm']:g}"
        delta = '—' if r['difference_mm'] is None else f"{r['difference_mm']:+.4f}"
        table.append(f"<tr><td>{html.escape(r['label'])}</td><td>{r['model_mm']:.3f}</td><td>{target}</td><td>{delta}</td><td>[{r['evidence']}]</td><td>{html.escape(r['note'])}</td></tr>")
    views = [('front','Front'),('left_side','Left side'),('right_side','Right side'),('top','Top'),('rear','Rear')]
    cards = ''.join(f'<section><h2>{title}</h2><a href="{name}.png"><img loading="lazy" src="{name}.png" alt="Dimensioned {title.lower()} view"></a></section>' for name,title in views)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C13534 dimension check</title><style>body{font:16px/1.5 system-ui,sans-serif;background:#eef2f5;color:#203647;margin:0}main{max-width:1250px;margin:auto;padding:30px}section{background:white;padding:24px;margin:24px 0;border-radius:10px}img{width:100%;height:auto}a{color:#1967b4}table{border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #ddd}.scroll{overflow:auto}code{background:#e6ebef}</style><main><h1>C13534 — dimensioned model views</h1>
<p>Orthographic views of the actual exported URDF, in the closed pose. Dimensions are measured from transformed mesh vertices in millimetres. Visual and collision meshes match. Rendering itself makes no geometry changes.</p>
<p><a href="../index.html">Current model</a> · <a href="../urdf/index.html">URDF review</a> · <a href="overview.png">Four-view PNG sheet</a> · <a href="measurements.json">Numerical check and source hashes</a></p>
<section><h2>What matches, and what remains estimated</h2><p><b>Body: 670 × 573 × 537 mm. Tower: 340 = 225 + 115 mm. Local depths: 267 and 280 mm.</b> These nominal constraints agree with the exported mesh within 0.001 mm numerical precision. That is a software consistency check, not physical accuracy. Your width allowance remains ±5 mm; depth uncertainty was not specified.</p>
<p>The complete closed model, including estimated projecting handles and hinges, measures <b>678 × 584.8 × 537 mm</b>. Do not uniformly rescale that complete bounding box to 670 × 573 × 537: that would shrink the already constrained body. The brochure dimension lines are used for the enclosure/support; hardware projections need their own measurements.</p>
<p><b>[M]</b> Published drawing &nbsp; <b>[U]</b> Your measurement &nbsp; <b>[E]</b> Unconfirmed estimate &nbsp; <b>[G]</b> Computed geometry extent. Door heights 230/200 mm are estimates. The 280 mm span is local at an approximate ruler position; it is not a maximum depth or curve radius. The 267 mm strip span is at its outer right edge. The corrected strip tapers to meet the door without a front depth step; the exact profile remains approximate. Controls, hinge placement, shell thickness and interiors remain unverified.</p>
<p>The brochure also shows a 12 mm projection on the low housing and detailed feet, ports and vents not reproduced by this simplified collider. The nominal envelope match does not mean the entire drawing has been reproduced.</p></section>
<section><h2>Vertex measurements</h2><div class="scroll"><table><tr><th>Feature</th><th>Model (mm)</th><th>Reference</th><th>Difference</th><th>Evidence</th><th>Interpretation</th></tr>TABLE</table></div></section>
<section><h2>Source dimension drawing</h2><p><a href="../evidence/hamamatsu-c13534.pdf">Manufacturer PDF, page 8</a> · <a href="../confirmed_measurements.json">Your ruler transcriptions</a></p><a href="../evidence/qy_dimension_drawing.png"><img src="../evidence/qy_dimension_drawing.png" alt="Manufacturer dimension drawing"></a></section>
CARDS
<p>Recreate: <code>.venv/bin/python -m src.instruments build --model c13534</code>. Rebuilds the current model, dimension sheets and portable package.</p></main></html>'''
    (DEST / 'index.html').write_text(page.replace('TABLE', ''.join(table)).replace('CARDS', cards))
    main = OUT / 'index.html'
    page = main.read_text()
    start, end = '<!-- DIMENSION_SHEETS_START -->', '<!-- DIMENSION_SHEETS_END -->'
    block = start+'''<section id="dimension-sheets"><h2>Dimensioned front, side, top and rear views</h2><p><a href="dimensions/index.html">Dimension review and reference comparison</a> · <a href="dimensions/overview.png">Four-view PNG sheet</a> · <a href="dimensions/measurements.json">Measured mesh dimensions</a></p><p>Actual closed URDF geometry: body 670 × 573 × 537 mm; complete hardware envelope 678 × 584.8 × 537 mm. Published values, your ruler readings and remaining estimates are labeled separately.</p><a href="dimensions/overview.png"><img loading="lazy" src="dimensions/overview.png" alt="Actual URDF with front, side and top dimensions"></a></section>'''+end
    if start in page:
        a, tail = page.split(start, 1)
        _, b = tail.split(end, 1)
        page = a+block+b
    else:
        page = page.replace('<!-- URDF_EXPORT_START -->', block+'<!-- URDF_EXPORT_START -->')
    main.write_text(page)


def main():
    DEST.mkdir(exist_ok=True)
    parts, inputs, error = read_parts()
    inputs += [OUT / 'meshes/collision_closed.glb', OUT / 'collision_doors.json',
               OUT / 'collision_controls.json', OUT / 'confirmed_measurements.json',
               ROOT / 'dataset/hkqai-instruments/c13534/measurements.json', DRAWING]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    m = measure(parts)
    render_sheets(parts, m)
    overview()
    for path, before in hashes.items():
        assert digest(ROOT / path) == before, 'Input changed during rendering: '+path
    validation = dict(visual_collision_meshes_identical=True, geometry_parts=len(parts),
                      triangles=sum(len(p.faces) for p in parts.values()),
                      maximum_urdf_vs_closed_glb_vertex_error_mm=error*1000,
                      nominal_constraint_check_tolerance_mm=.001,
                      nominal_constraints_checked=sum(r['target_mm'] is not None for r in m['rows']),
                      all_nominal_constraints_pass=all(r.get('nominal_constraint_pass',True) for r in m['rows']), source_geometry_and_parameters_unchanged=True)
    report(m, validation, hashes)
    print(json.dumps(validation, indent=2))
