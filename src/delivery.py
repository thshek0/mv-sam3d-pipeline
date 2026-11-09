"""Create compact, relocatable handoff folders from registered assets.

python -m src.delivery --output output/packages/delivery
Family/variant selection lives in delivery_profiles.json, never in geometry code.
Existing sources, products and deliveries are preserved. No ZIP is required.
"""
from pathlib import Path
import argparse
import base64
import html
import json
import os
import re
import shutil
import tempfile
import textwrap
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import trimesh

from .assets import export_asset, load_manifest, resolve
from .export import sha256
from .output_layout import canonical_name

ROOT = Path(__file__).resolve().parents[1]
PROFILE = Path(__file__).with_name('delivery_profiles.json')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def local_file(folder, name):
    """Only file references contained in this portable folder are permitted."""
    if Path(name).is_absolute():
        raise ValueError(f'Absolute delivery reference: {name}')
    path = (folder/name).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise ValueError(f'Missing or external delivery reference: {name}')
    return path


def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError(f'Invalid delivery identifier: {value}')
    return value


def selections(profile):
    if profile.get('schema_version') != 1:
        raise ValueError('Unsupported delivery profile schema')
    for family, settings in profile['families'].items():
        safe_name(family)
        for name, override in settings['objects'].items():
            safe_name(name)
            config = {k: v for k, v in settings.items() if k != 'objects'} | override
            safe_name(config['method'])
            for variant in config['variants']:
                safe_name(variant)
                # A single variant uses the object name; multiple variants get separate folders.
                leaf = name if len(config['variants']) == 1 else variant
                yield family, name, variant, Path(family)/leaf, config


def compact_mujoco(source, destination):
    """Copy only MJCF runtime assets, assigning folders by their actual role.

    Node names/counts come from the registered collision scene. We never split a
    mesh by object name, filename convention, material colour or a fixed count.
    """
    xml = source/'mujoco/collision/model.xml'
    root = ET.parse(xml).getroot()
    collision_meshes = {geom.get('mesh') for geom in root.iter('geom')
                        if geom.get('group') == '3'}
    used = set()
    for element in root.findall('asset/*'):
        reference = element.get('file')
        if not reference:
            continue
        original = (xml.parent/reference).resolve()
        if not original.is_relative_to(source.resolve()) or not original.is_file():
            raise ValueError(f'Invalid exported asset: {reference}')
        if element.tag == 'texture':
            relative = Path('textures')/original.name
        elif element.tag == 'mesh':
            role = 'collision' if element.get('name') in collision_meshes else 'visual'
            relative = Path('meshes')/role/original.name
        else:
            raise ValueError(f'Unsupported MuJoCo asset type: {element.tag}')
        if relative in used:
            raise ValueError(f'Duplicate destination: {relative}')
        used.add(relative)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.suffix.lower() == '.obj':
            # MJCF defines the material explicitly; no redundant MTL files.
            with original.open() as src, target.open('w') as dst:
                for line in src:
                    if not line.lstrip().startswith(('mtllib ', 'usemtl ')):
                        dst.write(line)
        else:
            shutil.copy2(original, target)
        element.set('file', relative.as_posix())
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(destination/'model.xml', encoding='utf-8', xml_declaration=True)
    return 'model.xml'


def check_scad(folder):
    references = 0
    for path in folder.rglob('*.scad'):
        text = re.sub(r'/\*.*?\*/|//[^\n]*', '', path.read_text(), flags=re.S)
        refs = re.findall(r'(?:include|use)\s*<([^>]+)>', text)
        refs += re.findall(r'import\s*\(\s*(?:file\s*=\s*)?"([^"]+)"', text)
        for ref in refs:
            target = local_file(folder, (path.parent.relative_to(folder)/ref).as_posix())
            if target.is_symlink():
                raise ValueError('Source deliveries must contain regular files')
            references += 1
    return {'scad_references_resolved': references,
            'validation_scope': 'Source hashes and literal include/use/import paths; no CAD rebuild'}


def mesh_thumbnails(path, destination):
    """Ray-cast the actual STL without simplification or a display server."""
    import open3d as o3d
    mesh = trimesh.load(path, force='mesh')
    ray_scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    ray_scene.add_triangles(o3d.core.Tensor(mesh.vertices.astype('float32')),
                            o3d.core.Tensor(mesh.faces.astype('uint32')))
    center, span = mesh.bounds.mean(0), float(mesh.extents.max())*1.25
    size = 400
    u, v = np.meshgrid((np.arange(size)+.5-size/2)/size*span,
                       (size/2-np.arange(size)-.5)/size*span)
    destination.mkdir(parents=True, exist_ok=True)
    for i, direction in enumerate(((1, 1, .5), (0, 0, 1), (1, 0, .05)), 1):
        view = np.array(direction, dtype=float); view /= np.linalg.norm(view)
        right = np.cross([0, 1, 0] if abs(view[2]) > .95 else [0, 0, 1], view)
        right /= np.linalg.norm(right); up = np.cross(view, right)
        origin = center+view*span*2+u[:, :, None]*right+v[:, :, None]*up
        rays = np.concatenate([origin, np.broadcast_to(-view, origin.shape)], axis=2)
        hit = ray_scene.cast_rays(o3d.core.Tensor(rays.astype('float32')), nthreads=4)
        mask = np.isfinite(hit['t_hit'].numpy())
        normals = hit['primitive_normals'].numpy()
        light = view+up*.5-right*.4; light /= np.linalg.norm(light)
        shade = .3+.7*np.maximum(normals@light, 0)
        pixels = np.full((size, size, 3), 246, dtype=np.uint8)
        pixels[mask] = np.clip(shade[mask, None]*[170, 185, 200], 0, 255)
        Image.fromarray(pixels).save(destination/f'{i}.png')


def mujoco_thumbnails(model, data, bounds, destination):
    # Set MUJOCO_GL before importing MuJoCo for headless rendering.
    import mujoco
    camera = mujoco.MjvCamera()
    camera.lookat[:] = np.mean(bounds, axis=0)
    camera.distance = float(np.max(bounds[1]-bounds[0]))*1.7
    # Preview-only studio lighting; the delivered XML/physics is unchanged.
    model.vis.headlight.ambient[:] = .45
    model.vis.headlight.diffuse[:] = .75
    model.vis.headlight.specular[:] = .15
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    destination.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=400, width=480) as renderer:
        for i, (azimuth, elevation) in enumerate(((135, -35), (90, -89), (0, -10)), 1):
            camera.azimuth, camera.elevation = azimuth, elevation
            renderer.update_scene(data, camera=camera, scene_option=option)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = renderer.render()
            renderer.disable_depth_rendering()
            rgb[depth >= model.vis.map.zfar*model.stat.extent*.99] = 246
            Image.fromarray(rgb).save(destination/f'{i}.png')


def validate_relocated(folder, format, entrypoint, *, bounds=None, thumbnails=True):
    """Copy the package elsewhere and load only relative package resources."""
    for path in folder.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'Symlink in delivery: {path}')
    with tempfile.TemporaryDirectory(prefix='delivery_load_') as tmp:
        relocated = Path(tmp)/'object'
        shutil.copytree(folder, relocated)
        entry = local_file(relocated, entrypoint)
        if format == 'mujoco':
            os.environ.setdefault('MUJOCO_GL', 'egl')
            import mujoco
            root = ET.parse(entry).getroot()
            for asset in root.findall('asset/*'):
                if asset.get('file'):
                    local_file(relocated, asset.get('file'))
            model = mujoco.MjModel.from_xml_path(str(entry))
            data = mujoco.MjData(model); mujoco.mj_forward(model, data)
            if thumbnails:
                mujoco_thumbnails(model, data, bounds, folder/'thumbnails')
            # Exercise the portable model itself; this is not task/contact calibration.
            for _ in range(100):
                mujoco.mj_step(model, data)
                if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                    raise ValueError('Non-finite MuJoCo state')
            if any(w.number for w in data.warning):
                raise ValueError('MuJoCo raised a warning during the smoke test')
            report = {'compiled': True, 'mujoco_version': mujoco.__version__,
                      'steps': 100, 'finite_state': True, 'mesh_assets': model.nmesh,
                      'dynamic': bool(model.nv), 'physical_properties_calibrated': False,
                      'validation_scope': 'Compilation and finite-state smoke test; no task/contact validation'}
        elif format == 'scad':
            report = check_scad(relocated)
        elif format == 'stl':
            mesh = trimesh.load(entry, force='mesh')
            if not mesh.is_volume or not mesh.is_winding_consistent:
                raise ValueError('Printing requires a closed, consistently wound positive-volume mesh')
            report = {'watertight': True, 'winding_consistent': True,
                      'positive_volume': True, 'triangles': len(mesh.faces),
                      'extents_source_units': mesh.extents.tolist(),
                      'validation_scope': 'Mesh topology only; wall thickness/supports are not checked'}
            if thumbnails:
                mesh_thumbnails(entry, folder/'thumbnails')
        else:
            raise ValueError(f'Unknown delivery format: {format}')
    report['relocated_load'] = True
    return report


def copy_scad_previews(manifest, data, config, destination):
    """Use the published previews only when their source snapshot still matches."""
    product = manifest.parent
    published = json.loads((product/'manifest.json').read_text())['files']
    for name, ref in data['native_files'].items():
        if published[f'openscad/{name}']['sha256'] != ref['sha256']:
            raise ValueError('Instrument preview belongs to a different SCAD build')
    out = destination/'thumbnails'; out.mkdir()
    for i, relative in enumerate(config['thumbnails'], 1):
        source = local_file(manifest.parent, relative)
        if sha256(source) != published[source.relative_to(product).as_posix()]['sha256']:
            raise ValueError(f'Stale preview: {source}')
        with Image.open(source) as im:
            im.convert('RGB').save(out/f'{i}.png')


STYLE = '<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#243546}a{color:#165f99}img{max-width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:24px}article{border:1px solid #ddd;border-radius:8px;padding:16px}article img{width:100%;height:210px;object-fit:contain;background:#f6f6f6}.views img{max-width:480px}code{background:#f4f4f4;padding:2px 5px}</style>'


def inline_image(path):
    """HTML previews remain visible when opened without a local asset server."""
    return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')


def preview_sheet(items, target, columns=3):
    """A directly openable image, so browsing does not depend on HTML support."""
    width, height = 400, 340
    columns = min(columns, len(items))
    board = Image.new('RGB', (columns*width, ((len(items)+columns-1)//columns)*height), '#f6f6f6')
    draw = ImageDraw.Draw(board)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 17)
    except OSError:
        font = ImageFont.load_default(size=17)
    for i, (path, label) in enumerate(items):
        x, y = i % columns*width, i // columns*height
        with Image.open(path) as image:
            tile = ImageOps.contain(image.convert('RGB'), (width-16, height-70))
        board.paste(tile, (x+(width-tile.width)//2, y+8+(height-70-tile.height)//2))
        draw.multiline_text((x+14, y+height-58), '\n'.join(textwrap.wrap(label, width=36)[:2]),
                            fill='#243546', font=font, spacing=3)
    board.save(target, quality=92)


def document(folder, name, format, entrypoint, units, checks, notes):
    instructions = f'Open {entrypoint}. Keep this entire folder together; it can be copied or zipped.'
    if format == 'stl':
        instructions = (f'Open {entrypoint} in your slicer. Set the intended physical size before slicing. '
                        'Source coordinates are preserved, NOT calibrated millimetres. STL stores no unit metadata.')
    if format == 'scad':
        instructions += (' Dimensions are in millimetres. Edit openscad/parameters_native.scad. '
                         'Pose/motion controls are documented in the SCAD comments. '
                         'No Python or reconstruction software is needed to open these sources.')
    if format == 'mujoco':
        instructions += (' Mesh coordinates are metres. XML references the OBJ and PNG files directly; '
                         'no MTL is required. Group 2 is visual; group 3 contains separate convex colliders.')
        instructions += (' This is a free rigid body.' if checks['dynamic'] else
                         ' This is STATIC: no recorded mass/inertia is available. Configure physics before dynamic use.')
        instructions += ' Physical properties remain estimates; a smoke test is not physics calibration.'
    text = f'{name}\n\n{instructions}\n\nUnits: {units}\n\n'+'\n'.join(notes)+'\n'
    (folder/'README.txt').write_text(text)
    preview_sheet([(p, f'{name.replace("_", " ")} / view {p.stem}')
                   for p in sorted((folder/'thumbnails').glob('*.png'))], folder/'preview.jpg')
    views = ''.join(f'<img src="{inline_image(p)}" alt="{html.escape(name)} view {p.stem}">'
                    for p in sorted((folder/'thumbnails').glob('*.png')))
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8">'+
        f'<title>{html.escape(name)}</title>'+STYLE+f'<h1>{html.escape(name)}</h1>'+
        f'<p>{html.escape(instructions)}</p><p><a href="{entrypoint}">Open {entrypoint}</a> · '+
        '<a href="preview.jpg">Preview image</a> · <a href="README.txt">Instructions</a> · '+
        '<a href="asset.json">Metadata and checks</a></p>'+
        ''.join(f'<p>{html.escape(note)}</p>' for note in notes)+f'<div class="views">{views}</div>')


def build_one(manifest, variant, config, destination, name):
    manifest, data = load_manifest(manifest)
    format = config['format']
    if format not in ('scad', 'mujoco', 'stl'):
        raise ValueError(f'Unsupported delivery format: {format}')
    destination.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='delivery_export_') as tmp:
        raw = Path(tmp)/'export'
        exported = export_asset(manifest, raw, [format], variant)
        if format == 'mujoco':
            entrypoint = compact_mujoco(raw, destination)
        elif format == 'scad':
            shutil.copytree(raw/'openscad', destination/'openscad')
            entrypoint = 'openscad/assembly.scad'
            copy_scad_previews(manifest, data, config, destination)
        else:
            if data['units'] == 'source_units' and config.get('print_scale') != 'preserve_source_set_in_slicer':
                raise ValueError('Uncalibrated STL needs an explicit print scale policy')
            entrypoint = f'{name}.stl'
            shutil.copy2(raw/f'{variant}.stl', destination/entrypoint)
    assets = data['variants'][variant]
    bounds = None
    if format == 'mujoco':
        bounds = trimesh.load(resolve(manifest, assets['visual']), force='mesh').bounds
    checks = validate_relocated(destination, format, entrypoint, bounds=bounds)
    notes = []
    if format == 'stl':
        notes.append('Preserved the accepted reconstruction without scaling or automatic geometry repair. '
                     'Choose print dimensions, orientation and supports in the slicer.')
    if 'physics' in assets:
        physics = json.loads(resolve(manifest, assets['physics']).read_text())
        notes.append(physics.get('provenance', physics.get('note', 'Physical properties are not calibrated.')))
    document(destination, name, format, entrypoint, data['units'], checks, notes)
    files = {p.relative_to(destination).as_posix(): sha256(p)
             for p in sorted(destination.rglob('*')) if p.is_file()}
    metadata = {'schema_version': 1, 'name': name, 'format': format,
                'entrypoint': entrypoint, 'units': data['units'], 'frame': data['frame'],
                'source_kind': data['source_kind'], 'source_variant': variant,
                'source_manifest_sha256': sha256(manifest),
                'source_asset_hashes': {key: value['sha256'] for key, value in assets.items()
                                        if isinstance(value, dict) and 'sha256' in value},
                'exporter_sha256': exported['exporter_sha256'] | {'delivery.py': sha256(Path(__file__))},
                'versions': exported['versions'], 'checks': checks,
                'notes': notes, 'files': files}
    write_json(destination/'asset.json', metadata)
    return metadata


def gallery(destination, entries):
    cards = []
    for relative, data in entries:
        folder = relative.as_posix()
        label = data['format'].upper()
        if data['format'] == 'mujoco':
            label += ' · dynamic' if data['checks']['dynamic'] else ' · static'
        if data['format'] == 'stl':
            label += ' · set size in slicer'
        image_url = inline_image(destination/relative/'thumbnails/1.png')
        cards.append(f'<article><a href="{folder}/index.html"><img src="{image_url}" '+
                     f'alt="{html.escape(data["name"])}"><h2>{html.escape(data["name"])}</h2></a>'+
                     f'<p>{label}</p><code>{folder}/</code></article>')
    preview_sheet([(destination/relative/'thumbnails/1.png',
                    data['name'].replace('_', ' ')+' / '+data['format'].upper())
                   for relative, data in entries], destination/'preview.jpg', columns=4)
    (destination/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Asset deliveries</title>'+STYLE+
        '<h1>Asset deliveries</h1><p>Each object folder is self-contained: copy it or ZIP it to share. '+
        'Open an object for previews and loading instructions. '+
        '<a href="preview.jpg">Open the overview image directly</a>.</p><div class="grid">'+''.join(cards)+'</div>')
    (destination/'README.txt').write_text('Open preview.jpg for an overview image, or index.html to browse thumbnails. Copy or ZIP an entire object folder.\n'
        'QAI instruments: OpenSCAD. Hestia: MuJoCo, except both birds: STL (set size in slicer).\n'
        'HKQAI trial is intentionally excluded. Source recipes, captures and experiments stay in the workspace.\n')


def build(workspace, destination, profile=PROFILE, only=None):
    workspace, destination = Path(workspace).resolve(), Path(destination).resolve()
    config = json.loads(Path(profile).read_text())
    only = [canonical_name(name) for name in only] if only else None
    selected = [entry for entry in selections(config) if not only or entry[1] in only]
    if only and set(only)-{entry[1] for entry in selected}:
        raise ValueError('Unknown or excluded object selection')
    if not selected:
        raise ValueError('No objects selected')
    if len({entry[3] for entry in selected}) != len(selected):
        raise ValueError('Profile produces duplicate delivery folders')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError('Choose a new or empty delivery directory; existing deliveries are preserved')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.delivery_', dir=destination.parent) as tmp:
        stage = Path(tmp)/'models'; stage.mkdir()
        entries = []
        for family, name, variant, relative, settings in selected:
            manifest = (local_file(workspace, settings['manifest']) if 'manifest' in settings
                        else workspace/'output'/family/name/('' if settings['method']=='codex' else settings['method'])/'asset.json')
            print(f'{name} / {variant} -> {settings["format"]}', flush=True)
            data = build_one(manifest, variant, settings, stage/relative, relative.name)
            entries.append((relative, data))
        gallery(stage, entries)
        write_json(stage/'catalog.json', {'schema_version': 1, 'profile_sha256': sha256(Path(profile)),
                   'objects': [{'folder': p.as_posix(), 'name': d['name'], 'format': d['format'],
                                'asset_sha256': sha256(stage/p/'asset.json')} for p, d in entries]})
        if destination.exists():
            destination.rmdir()  # Verified empty above.
        stage.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True, help='New or empty directory')
    parser.add_argument('--profile', type=Path, default=PROFILE)
    parser.add_argument('--only', nargs='+', help='Optional object names from the profile')
    args = parser.parse_args()
    result = build(args.workspace, args.output, args.profile, args.only)
    print(f'Delivery gallery: {result / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
