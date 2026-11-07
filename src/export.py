"""Shared visual mesh export helpers. See README.md's asset storage section.

Call with an explicit source and destination. This module does not select datasets,
create ZIPs, or define object-specific command-line flags.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def export_visual(source, destination, stem, formats=('glb',)):
    """Export selected visual formats; GLB is copied without re-encoding.

    OBJ retains supported base-colour appearance, not full PBR materials or scene
    semantics. STL retains geometry only. Neither is an editable CAD source.
    """
    from .appearance import load_surface, write_color_obj
    source, destination = Path(source), Path(destination)
    formats = set(formats)
    if not formats or formats - {'glb', 'obj', 'stl'}:
        raise ValueError('Choose one or more of glb, obj, stl')
    if source.suffix.lower() != '.glb':
        raise ValueError('Visual master must be a GLB')
    if Path(stem).name != stem or stem in ('', '.', '..'):
        raise ValueError('Export stem must be a filename, not a path')
    if any((destination/f'{stem}.{suffix}').resolve() == source.resolve() for suffix in formats):
        raise ValueError('Export destination must not overwrite the source')
    destination.mkdir(parents=True, exist_ok=True)
    if formats == {'glb'}:
        glb = destination/f'{stem}.glb'
        shutil.copyfile(source, glb)
        return {'exports': {'glb': {'sha256': sha256(glb), 'bytes': glb.stat().st_size,
                                    'unchanged_source_copy': True}}, 'appearance': None}
    mesh, colors, normals = load_surface(source)
    obj = destination/f'{stem}.obj'
    stl = destination/f'{stem}.stl'
    appearance = write_color_obj(mesh, colors, normals, obj) if 'obj' in formats else None
    validation = source.parent/'validation.json'
    if appearance and validation.exists():
        source_appearance = json.loads(validation.read_text()).get('appearance_check', {})
        if 'pattern_provenance' in source_appearance:
            appearance['pattern_provenance'] = source_appearance['pattern_provenance']
    if 'stl' in formats:
        mesh.export(stl)
    glb = destination/f'{stem}.glb'
    if 'glb' in formats:
        shutil.copyfile(source, glb)
    checks = {}
    tolerance = max(1e-8, float(np.abs(mesh.vertices).max())*2e-7)
    for path in (obj, stl):
        if path.suffix[1:] not in formats:
            continue
        loaded = trimesh.load(path, force='mesh', process=False)
        if len(loaded.faces) != len(mesh.faces):
            raise ValueError(f'Triangle count changed: {path}')
        # Comparing every triangle checks position, winding and scene transforms.
        error = float(np.max(np.abs(loaded.triangles-mesh.triangles)))
        if error > tolerance:
            raise ValueError(f'Geometry changed by {error}: {path}')
        checks[path.suffix[1:]] = {'max_coordinate_error': error, 'sha256': sha256(path),
                                  'bytes': path.stat().st_size}
    if 'glb' in formats:
        checks['glb'] = {'sha256': sha256(glb), 'bytes': glb.stat().st_size, 'unchanged_source_copy': True}
    return {'triangles': len(mesh.faces), 'bounds': mesh.bounds.tolist(),
            'extents': mesh.extents.tolist(), 'exports': checks, 'appearance': appearance}


def export_pair(source, destination, stem):
    """Compatibility helper for explicitly requested multi-format deliveries."""
    return export_visual(source, destination, stem, ('glb', 'obj', 'stl'))
