"""Output lifecycle: retain failures, preserve simulation dependencies, prune reviews."""
from pathlib import Path
import json
import pytest
from PIL import Image
from src.instruments.storage.diagnostics import session, workspace
from src.instruments.storage.policy import apply


def test_success_cleanup_and_requested_retention(tmp_path):
    for keep in (False, True):
        with session(tmp_path, keep=keep):
            with workspace('edited') as folder:
                path = Path(folder) / 'edited.scad'
                path.write_text('width=123;')
        assert not path.exists()
        assert (tmp_path / 'validation/latest').exists() == keep
    latest = tmp_path / 'validation/latest'
    assert next(latest.rglob('edited.scad')).read_text() == 'width=123;'
    assert json.loads((latest / 'session.json').read_text())['status'] == 'passed'


def test_failed_nested_work_is_kept_without_flag(tmp_path):
    with pytest.raises(ValueError, match='bad geometry'):
        with session(tmp_path):
            with session(tmp_path):
                with workspace('native') as folder:
                    (Path(folder) / 'broken.stl').write_text('invalid solid')
                    raise ValueError('bad geometry')
    failure = next((tmp_path / 'validation/failures').iterdir())
    assert next(failure.rglob('broken.stl')).read_text() == 'invalid solid'
    assert 'bad geometry' in (failure / 'failure.txt').read_text()
    assert not list(tmp_path.glob('.validation_*'))


@pytest.mark.parametrize('extras', [False, True])
def test_output_policy_preserves_loadable_assets(tmp_path, extras):
    stage = tmp_path / 'model'
    for name in ('parameters.json', 'reports/validation.json', 'reports/provenance.json'):
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    for name in ('meshes/visual_closed.glb', 'meshes/visual_open.glb',
                 'meshes/visual_upper_open.glb', 'meshes/collision_closed.glb',
                 'urdf/meshes/collision.obj', 'urdf/visual/image.png',
                 'openscad/common.scad', 'openscad/assembly.scad',
                 'previews/user_notes.txt'):
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'preserve exact bytes')
    for name in ('dimensions/overview.png', 'previews/assembly.png', 'previews/interior.png'):
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (10, 10)).save(path)
    with session(tmp_path, keep=True):
        apply(stage, extras=extras)
    for name in ('meshes/visual_closed.glb', 'urdf/meshes/collision.obj',
                 'urdf/visual/image.png', 'openscad/assembly.scad', 'previews/user_notes.txt'):
        assert (stage / name).read_bytes() == b'preserve exact bytes'
    for name in ('previews/interior.png', 'meshes/visual_upper_open.glb', 'meshes/collision_closed.glb'):
        assert (stage / name).exists() == extras
        assert (tmp_path / 'validation/latest/review' / name).exists()
    assert (stage / 'dimensions/overview.png').exists()
