"""Exercise OpenSCAD's dynamic $t scope, not just generated control text."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import trimesh

from src.instruments.shared_geometry import Assembly


@pytest.mark.parametrize('time,offset', [(0, 0), (.25, 10), (.5, 20), (1, 0)])
def test_animation_translates_moving_part_at_evaluation_time(tmp_path, time, offset):
    executable = os.environ.get('OPENSCAD') or shutil.which('openscad')
    if not executable:
        pytest.skip('OpenSCAD is required for native motion regression')
    assembly = Assembly({}, [], travel=20, axis=(1, 0, 0))
    assembly.add('lid', 'block', [[0, 0, 0], [2, 3, 4]], group='cover')
    assembly.export_scad(tmp_path/'assembly.scad')
    (tmp_path/'parameters_native.scad').write_text('// no parameters')
    shutil.copy2(Path(__file__).parents[2]/'src/instruments/scad/common.scad', tmp_path/'common.scad')
    result = subprocess.run([executable, '-o', str(tmp_path/'pose.stl'),
                             '-D', 'animate_motion=true', '-D', '$t='+json.dumps(time),
                             str(tmp_path/'assembly.scad')], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and 'WARNING:' not in result.stderr and 'ERROR:' not in result.stderr, result.stderr
    mesh = trimesh.load(tmp_path/'pose.stl', force='mesh')
    np.testing.assert_allclose(mesh.bounds, [[offset, 0, 0], [offset+2, 3, 4]])
