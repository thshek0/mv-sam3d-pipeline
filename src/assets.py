"""Canonical asset manifests and explicit, on-demand delivery exports.

python -m src.assets inspect path/to/object/asset.json
python -m src.assets export path/to/object/asset.json --format obj --output /tmp/object

Manifests reference existing authoring files; they never replace them with meshes.
Export destinations must be new or empty, so stale files cannot enter a delivery.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET

from .export import export_visual, sha256

SCHEMA = 1
FORMATS = ('glb', 'obj', 'stl', 'scad', 'step', 'mujoco')
AUTHORED_KINDS = frozenset(('recipe', 'openscad', 'radial_profile', 'radial_profile_cad', 'accepted_mesh_spoon', 'parametric_spoon'))


def check_codex_dependencies(manifest, sources, kind=None):
    """Codex assets must not depend on another method's disposable outputs."""
    if Path(manifest).parent.name != 'codex' and kind not in AUTHORED_KINDS:
        return
    for source in sources:
        resolved = Path(source).resolve()
        if set(resolved.parts) & {'sam3d', 'tsdf', 'dualbrep'}:
            raise ValueError(f'Codex dependency must be a stable authoring input, not an experimental method output: {resolved}')


def save_manifest(path, *, kind, sources, variants, units, frame, rebuild,
                  native_files=None):
    """Record references and hashes without copying or relocating source assets.

    Variant entries contain Path values for visual/collision/physics/step assets,
    and optional physics_key for a shared mass table. Units describe visual meshes.
    """
    path = Path(path).resolve()
    sources = list(sources)
    check_codex_dependencies(path, sources + list((native_files or {}).values()) +
                             [value for entry in variants.values() for key, value in entry.items() if key != 'physics_key'], kind)

    def reference(source):
        source = Path(source).resolve()
        return {'path': os.path.relpath(source, path.parent), 'sha256': sha256(source)}

    data = dict(schema_version=SCHEMA, source_kind=kind, units=units, frame=frame,
                sources=[reference(p) for p in sources], rebuild=rebuild,
                variants={name: {key: reference(value) if key != 'physics_key' else value
                                 for key, value in entry.items()}
                          for name, entry in variants.items()},
                native_files={name: reference(p) for name, p in (native_files or {}).items()},
                export_directory='exports',
                policy='Preserve authoring sources; generate only requested delivery formats.')
    if path.parent.name == 'codex' or kind in AUTHORED_KINDS:
        data['dependency_policy'] = {'experimental_method_inputs_allowed': False,
                                     'registered_paths_checked_after_resolving_symlinks': True}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')
    return data


def load_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get('schema_version') != SCHEMA:
        raise ValueError('Unsupported asset manifest schema')
    return path, data


def resolve(path, reference):
    source = (path.parent/reference['path']).resolve()
    if sha256(source) != reference['sha256']:
        raise ValueError(f'Source changed: {source}. Rebuild and refresh the asset manifest first.')
    return source


def export_asset(manifest, destination, formats, variant=None):
    """Produce an isolated delivery; never modify source files or old deliveries."""
    path, data = load_manifest(manifest)
    formats = set(formats)
    if not formats or formats - set(FORMATS):
        raise ValueError('Unknown or empty format selection')
    variants = data['variants']
    if variant is None:
        if len(variants) != 1:
            raise ValueError('Choose --variant from: '+', '.join(variants))
        variant = next(iter(variants))
    if variant not in variants or Path(variant).name != variant or variant in ('', '.', '..'):
        raise ValueError(f'Unknown or invalid variant: {variant}')
    entry = variants[variant]
    # Detect stale preview/physics inputs rather than exporting a mismatched model.
    sources = [resolve(path, reference) for reference in data['sources']]
    assets = {key: resolve(path, value) for key, value in entry.items() if key != 'physics_key'}
    native = {name: resolve(path, value) for name, value in data['native_files'].items()}
    check_codex_dependencies(path, sources + list(assets.values()) + list(native.values()), data['source_kind'])
    if any(Path(name).name != name or name in ('', '.', '..') for name in native):
        raise ValueError('Native source names must be filenames')
    if 'scad' in formats and not native:
        raise ValueError('No editable OpenSCAD source is registered; a mesh cannot recover it.')
    if 'step' in formats and 'step' not in assets:
        raise ValueError('No solid STEP source is registered. Mesh-to-STEP is not lossless CAD recovery.')
    visual_formats = formats & {'glb', 'obj', 'stl'}
    native_stl = visual_formats == {'stl'} and 'visual' not in assets and 'assembly.scad' in native
    if visual_formats and 'visual' not in assets and not native_stl:
        raise ValueError('No visual GLB registered. Native OpenSCAD supports scad/stl export here.')
    has_collision = 'collision' in assets or (data['source_kind'] == 'radial_profile' and 'collision_recipe' in assets)
    if 'mujoco' in formats and ('visual' not in assets or not has_collision or data['units'] != 'm'):
        raise ValueError('MuJoCo needs an explicit visual GLB, separate convex collision GLB and metre units.')
    destination = Path(destination).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError('Choose a new or empty export directory; existing deliveries are preserved.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.asset_export_', dir=destination.parent) as temporary:
        stage = Path(temporary)/'delivery'
        stage.mkdir()
        checks = {}
        if 'scad' in formats or native_stl:
            folder = stage/'openscad'
            folder.mkdir()
            for name, source in native.items():
                shutil.copy2(source, folder/name)
            if native_stl:
                from .instruments.native_build import run_engine
                run_engine(folder/'assembly.scad', stage/f'{variant}.stl')
                (stage/f'{variant}.stl.log').unlink(missing_ok=True)
                checks['stl'] = {'units': data['units'], 'source': 'Native OpenSCAD tessellation',
                                 'appearance': 'Geometry only; no colours or materials'}
            if 'scad' not in formats:
                shutil.rmtree(folder)
        if visual_formats and not native_stl:
            checks['visual'] = export_visual(assets['visual'], stage, variant, visual_formats)
        if 'step' in formats:
            shutil.copy2(assets['step'], stage/f'{variant}.step')
        if 'mujoco' in formats:
            checks['mujoco'] = export_mujoco(assets, entry.get('physics_key'), stage/'mujoco', variant)
        files = {p.relative_to(stage).as_posix(): sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        # Include exporter dependencies and package versions for reproducibility.
        import numpy
        import PIL
        import trimesh
        implementation = Path(__file__).parent
        report = dict(schema_version=SCHEMA, variant=variant, formats=sorted(formats),
                      units=data['units'], frame=data['frame'], checks=checks, files=files,
                      source_manifest_sha256=sha256(path),
                      source_files={str(p): sha256(p) for p in sources+list(assets.values())+list(native.values())},
                      exporter_sha256={name: sha256(implementation/name) for name in
                                       ('assets.py', 'export.py', 'appearance.py', 'collision.py',
                                        'products_mujoco.py', 'instruments/native_build.py')},
                      versions={'numpy': numpy.__version__, 'trimesh': trimesh.__version__, 'Pillow': PIL.__version__},
                      limitations=['OBJ preserves supported base-colour appearance, not complete PBR/scene semantics.',
                                   'STL is geometry only. CAD tessellation is approximate.',
                                   'MuJoCo exports have offline checks only; no simulation validation.'])
        (stage/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        if destination.exists():
            destination.rmdir()  # Only the previously verified empty directory.
        stage.rename(destination)
    return report


def export_mujoco(assets, physics_key, destination, variant):
    """Use explicit authored hulls; preserve textured visuals through OBJ/PNG."""
    from .collision import export_collision
    from .products_mujoco import load_parts, inertial_properties, numbers, write_xml
    import numpy as np

    appearance = export_visual(assets['visual'], destination, variant, ('obj',))['appearance']
    construction = {}
    if 'collision' in assets:
        parts = load_parts(assets['collision'])
    else:
        from .collision import plate_parts
        parts, construction = plate_parts(assets['visual'], assets['collision_recipe'])
    report = export_collision(assets['visual'], destination/'collision',
                              destination/f'{variant}.obj', appearance,
                              parts=parts, formats=('obj',))
    report.update(construction)
    physics = json.loads(assets['physics'].read_text()) if 'physics' in assets else None
    if physics:
        mass = physics['mass_kg']
        if isinstance(mass, dict):
            mass = mass[physics_key]
        friction = np.asarray(physics['friction'], dtype=float)
        if friction.shape != (3,) or not np.isfinite(friction).all() or (friction < 0).any():
            raise ValueError('Friction must contain three finite nonnegative values')
        center, inertia = inertial_properties(parts, mass)
        if 'inertia_kg_m2' in physics:
            center = np.asarray(physics['center_of_mass_m'], dtype=float)
            inertia = np.asarray(physics['inertia_kg_m2'], dtype=float)
        if center.shape != (3,) or inertia.shape != (3, 3) or not np.isfinite(center).all() or not np.isfinite(inertia).all():
            raise ValueError('Invalid center of mass or inertia shape/value')
        eigenvalues = np.linalg.eigvalsh(inertia)
        if not np.allclose(inertia, inertia.T) or eigenvalues[0] <= 0 or eigenvalues[-1] > eigenvalues[:2].sum()+1e-12:
            raise ValueError('Inertia must be symmetric and physically valid')
        model_path = destination/'collision/model.xml'
        model = ET.parse(model_path).getroot()
        body = model.find('worldbody/body')
        ET.SubElement(body, 'freejoint', name=variant+'_free')
        ET.SubElement(body, 'inertial', mass=str(mass), pos=numbers(center),
                      fullinertia=numbers([inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                           inertia[0, 1], inertia[0, 2], inertia[1, 2]]))
        for geom in body.findall('geom'):
            if geom.get('group') == '3':
                geom.set('friction', numbers(friction))
                for key in ('solref', 'solimp'):
                    if key in physics:
                        geom.set(key, numbers(physics[key]))
                for key in ('priority', 'margin'):
                    if key in physics:
                        geom.set(key, str(physics[key]))
        write_xml(model_path, model)
        report.update(assembly='Free rigid body; explicit mass and contact settings',
                      inertia_source='Configured tensor' if 'inertia_kg_m2' in physics else 'Uniform density across collision pieces')
        (destination/'physics.json').write_text(json.dumps(physics, indent=2)+'\n')
    model_path = destination/'collision/model.xml'
    model = ET.parse(model_path).getroot()
    if 'contact_settings' in assets:
        contacts = json.loads(assets['contact_settings'].read_text())
        for geom in model.findall('worldbody/body/geom'):
            if geom.get('group') == '3':
                for key in ('solref', 'solimp'):
                    if key in contacts:
                        geom.set(key, numbers(contacts[key]))
                for key in ('priority', 'margin'):
                    if key in contacts:
                        geom.set(key, str(contacts[key]))
        write_xml(model_path, model)
        (destination/'contact_settings.json').write_text(json.dumps(contacts, indent=2)+'\n')
        report['contact_settings'] = contacts
    for asset in model.findall('asset/*'):
        if asset.get('file') and not (model_path.parent/asset.get('file')).is_file():
            raise ValueError('Unresolved MuJoCo asset: '+asset.get('file'))
    (destination/'collision/validation.json').write_text(json.dumps(report, indent=2)+'\n')
    (destination/'collision/README.txt').write_text(
        'Load model.xml. Keep the entire mujoco folder together, including its parent-level OBJ/MTL/PNG.\n'
        'Metres; coordinate frame is recorded in the delivery manifest.\n'
        +report['assembly']+'.\n'
        'Each collision mesh is a separate convex piece. Visual appearance uses base-colour textures.\n'
        'Offline geometry and asset-path checks only; not compiled or simulated in MuJoCo.\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect')
    inspect.add_argument('manifest', type=Path)
    export = commands.add_parser('export')
    export.add_argument('manifest', type=Path)
    export.add_argument('--variant')
    export.add_argument('--format', choices=FORMATS, nargs='+', required=True)
    export.add_argument('--output', type=Path, required=True, help='New or empty delivery directory')
    args = parser.parse_args()
    try:
        if args.command == 'inspect':
            _, data = load_manifest(args.manifest)
            print(json.dumps(data, indent=2))
        else:
            report = export_asset(args.manifest, args.output, args.format, args.variant)
            print(f'Exported {report["variant"]}: {", ".join(report["formats"])} -> {args.output}')
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
