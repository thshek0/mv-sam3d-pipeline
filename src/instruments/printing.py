"""Explicit native CAD part export in millimetres; never treat a GLB as printable CAD."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .paths import ROOT
from .storage.diagnostics import session, workspace, command_log


def export(model, part, pose='closed', keep_validation=False):
    import trimesh
    from .design import parameters, module
    machine = ROOT / 'output/hkqai-instruments' / model
    current = machine
    params = json.loads((current / 'parameters.json').read_text())
    values, _ = parameters(params)
    names = {p.name for p in module(model).definition(values).parts}
    if part not in names:
        raise ValueError('Choose one semantic part: ' + ', '.join(sorted(names)))
    executable = os.environ.get('OPENSCAD') or shutil.which('openscad')
    if not executable:
        raise RuntimeError('Set OPENSCAD or install OpenSCAD to export a native STL.')
    with session(ROOT/'work/hkqai-instruments'/model/'codex', keep=keep_validation):
        with workspace('printing') as folder:
            work = Path(folder)
            # Regenerate from authoritative parameters, not potentially edited/stale SCAD.
            from .cad import export as export_cad
            shutil.copy2(current / 'parameters.json', work / 'parameters.json')
            source = export_cad(work) / 'assembly.scad'
            path = work / f'{part}_{pose}.stl'
            command = [executable, '-o', str(path), '-D', 'part=' + json.dumps(part),
                       '-D', 'pose=' + json.dumps(pose), str(source.resolve())]
            result = subprocess.run(command, capture_output=True, text=True, timeout=240)
            command_log(path, command, result)
            if result.returncode or not path.exists() or 'ERROR:' in result.stderr:
                raise RuntimeError(result.stderr)
            mesh = trimesh.load(path, force='mesh')
            if not mesh.is_watertight or not mesh.is_volume:
                raise ValueError('Native part is not a closed positive-volume solid; inspect retained diagnostics.')
            report = dict(model=model, part=part, pose=pose, units='millimetres',
                          bounds_mm=mesh.bounds.tolist(), watertight=True,
                          positive_volume=True, connected_components=len(mesh.split()),
                          parameters_sha256=hashlib.sha256((current / 'parameters.json').read_bytes()).hexdigest(),
                          limitation='Geometry checks only. Fit, wall thickness, material, supports and printer settings require review.')
            dest = current / 'printing'
            dest.mkdir(exist_ok=True)
            shutil.copy2(path, dest / path.name)
            (dest / path.with_suffix('.json').name).write_text(json.dumps(report, indent=2) + '\n')
            print(dest / path.name, flush=True)
            return dest / path.name
