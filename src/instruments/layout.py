"""Normalize staged deliverables; builders may keep temporary working filenames."""
import json

STATES = ('closed', 'open', 'upper_open', 'lower_open', 'handles_released')
RELOCATIONS = {
    'preview.png': 'previews/assembly.png',
    'interior.png': 'previews/interior.png',
    'preview_collision.png': 'previews/assembly_collision.png',
    'interior_collision.png': 'previews/interior_collision.png',
    'motion.gif': 'previews/motion.gif',
    'closed.png': 'previews/closed.png',
    'open.png': 'previews/open.png',
    'front_textured.png': 'previews/front.png',
    'front_reference_comparison.png': 'previews/front_reference_comparison.png',
    'sources.png': 'previews/source_review.png',
    'review.json': 'reports/validation.json',
    'parts.json': 'reports/parts.json',
    'provenance.json': 'reports/provenance.json',
    'limitations.json': 'reports/limitations.json',
    'collision_controls.json': 'reports/assembly_controls.json',
    'collision_doors.json': 'reports/door_configuration.json',
    'confirmed_measurements.json': 'reports/confirmed_measurements.json',
}


def normalize(stage, model):
    """Idempotently organize a private staging copy, before hashes/ZIP are made."""
    for old, new in RELOCATIONS.items():
        source, target = stage / old, stage / new
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
    for duplicate, canonical in {
        'preview_textured.png': 'previews/assembly.png',
        'interior_textured.png': 'previews/interior.png',
        'validation.json': 'reports/validation.json',
        **{f'meshes/{state}.glb': f'meshes/visual_{state}.glb' for state in STATES},
        **{f'meshes/{kind}_controls_{state}.glb': f'meshes/{kind}_{state}.glb'
           for kind in ('visual', 'collision') for state in STATES},
    }.items():
        source, target = stage / duplicate, stage / canonical
        if source.exists():
            if not target.exists() or source.read_bytes() != target.read_bytes():
                raise ValueError(f'Refusing to discard differing/missing canonical asset: {duplicate}')
            source.unlink()
    # Previous-iteration comparisons are not part of the latest-only deliverable.
    (stage / 'comparison.png').unlink(missing_ok=True)
    named = stage / 'urdf' / f'{model}.urdf'
    if named.exists():
        canonical = stage / 'urdf/robot.urdf'
        if canonical.exists() and named.read_bytes() != canonical.read_bytes():
            raise ValueError('Refusing to discard differing URDF aliases')
        named.replace(stage / 'urdf/robot.urdf')
    manifest = stage / 'urdf/manifest.json'
    if manifest.exists():
        data = json.loads(manifest.read_text())
        data['urdf'] = 'robot.urdf'
        manifest.write_text(json.dumps(data, indent=2) + '\n')
    page = stage / 'dimensions/index.html'
    if page.exists():
        page.write_text(page.read_text().replace('../confirmed_measurements.json',
                                                '../reports/confirmed_measurements.json'))
