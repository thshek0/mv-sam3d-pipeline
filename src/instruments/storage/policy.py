"""Compact deliverables with explicit optional reviews; never prune unknown files."""
import hashlib
import json
from PIL import Image
from .diagnostics import capture

VIEWS = ('front', 'left_side', 'right_side', 'top', 'rear')


def dimension_sheet(stage):
    folder = stage / 'dimensions'
    sheet = folder / 'overview.png'
    sources = [folder / (name + '.png') for name in VIEWS]
    if not all(path.exists() for path in sources):
        if not sheet.exists() and not (stage / 'overview.png').exists():
            raise FileNotFoundError('No complete dimension views or combined sheet')
        return
    images = []
    for path in sources:
        with Image.open(path) as source:
            image = source.convert('RGB')
            image.thumbnail((1350, 1440))
            images.append(image)
    width, height = max(i.width for i in images), max(i.height for i in images)
    board = Image.new('RGB', (width * 3, height * 2), 'white')
    for i, image in enumerate(images):
        board.paste(image, ((i % 3) * width, (i // 3) * height))
    board.save(sheet)


def apply(stage, extras=False):
    """Keep URDF dependencies and compact numerical reports in every profile."""
    dimension_sheet(stage)
    for name in ('parameters.json', 'reports/validation.json',
                 'reports/provenance.json', 'openscad/parameters_native.scad',
                 'openscad/assembly.scad', 'openscad/common.scad'):
        capture(stage / name, 'inputs_and_reports/' + name)
    checks = stage / 'openscad/csg_validation.json'
    summary_path = stage / 'reports/cad_summary.json'
    if checks.exists():
        data = json.loads(checks.read_text())
        shared = data.get('shared_backend_checks', {})
        summary = {key: value for key, value in data.items()
                   if key not in ('checks', 'shared_backend_checks')}
        summary.update(cases=len(data.get('checks', {})),
                       shared_backend_checks={k: v for k, v in shared.items() if k != 'parts_checked'},
                       parts_checked=len(shared.get('parts_checked', {})),
                       native_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in (stage / 'openscad').glob('*.scad')})
        # Keep tested tolerances even when the per-part diagnostic is omitted.
        summary['part_tolerances'] = [dict(items) for items in sorted({
            tuple(sorted((k, v) for k, v in part.items() if 'tolerance' in k))
            for part in shared.get('parts_checked', {}).values()})]
        summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    elif not summary_path.exists():
        summary_path.write_text(json.dumps({'status': 'not run',
            'run': 'build --check-cad; add --keep-validation for diagnostics'}, indent=2) + '\n')
    candidates = [*(stage / 'openscad').glob('csg_*'),
                  *(stage / 'previews').glob('*'),
                  *(stage / 'meshes').glob('collision_*.glb'),
                  *(stage / 'meshes').glob('visual_*.glb'),
                  *(stage / 'dimensions').glob('*.png'), stage / 'dimensions/index.html',
                  stage / 'urdf/review.json', stage / 'textures/sources.png']
    kept = {'previews/assembly.png', 'meshes/visual_closed.glb',
            'meshes/visual_open.glb', 'dimensions/overview.png'}
    optional = []
    for path in candidates:
        relative = str(path.relative_to(stage))
        if not path.is_file() or relative in kept:
            continue
        # Restrict preview pruning to known generated names.
        if path.parent.name == 'previews' and path.name not in {
            'interior.png', 'assembly_collision.png', 'interior_collision.png',
            'motion.gif', 'closed.png', 'open.png', 'front.png',
            'front_reference_comparison.png', 'source_review.png'}:
            continue
        capture(path, 'review/' + relative)
        optional.append(relative)
        if not extras:
            path.unlink()
    (stage / 'reports/output_policy.json').write_text(json.dumps({
        'profile': 'extras' if extras else 'compact',
        'optional_artifacts': sorted(optional),
        'regenerate_optional': 'build --extras',
        'retain_diagnostics': 'build --check-cad --keep-validation',
        'collision_geometry': 'Separate convex pieces are always retained in urdf/meshes.',
    }, indent=2) + '\n')
