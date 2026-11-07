"""Instrument grouping must reach every pipeline artifact without changing other runs."""
from pathlib import Path

import pytest

from helpers import (dataset_root, dataset_scene_dir, dataset_depth_npz,
                     dataset_meshes_dir, dataset_name_from_input)


@pytest.mark.parametrize('model', ['cary60', 'c13472', 'c16361', 'c13534'])
def test_instrument_paths_and_explicit_machine_root(tmp_path: Path, model: str):
    name = model + '_photo_closed'
    expected = tmp_path / 'objects/hkqai-instruments' / model / 'sam3d/intermediate' / name
    assert dataset_root(tmp_path, name) == expected
    assert dataset_root(tmp_path / 'objects/hkqai-instruments' / model, name) == expected
    assert dataset_root(tmp_path / 'objects/hkqai-instruments', name) == expected
    assert dataset_scene_dir(tmp_path, name) == expected / 'scene'
    assert dataset_depth_npz(tmp_path, name, 'rs') == expected / 'depth/rs.npz'
    assert dataset_meshes_dir(tmp_path, name) == tmp_path / 'objects/hkqai-instruments' / model / 'sam3d/products' / name
    assert dataset_name_from_input(expected / 'scene') == name


@pytest.mark.parametrize('name', ['gripper_fixture', 'c134720_photo_closed'])
def test_other_names_use_misc_output_family(tmp_path: Path, name: str):
    assert dataset_root(tmp_path, name) == tmp_path / "objects/misc" / name / "sam3d/intermediate"


def test_grouped_batch_identity_and_mixed_state_rejection(tmp_path: Path):
    import json
    from helpers import dataset_input_dir
    base = tmp_path / 'dataset'
    batch = base / 'hkqai-instruments/c13472/captures/2026-10-02_rgbd_open_01'
    (batch / 'raw').mkdir(parents=True)
    (batch / 'dataset.json').write_text(json.dumps({'run_id':'c13472_rgbd_open','eligible_for_rigid_reconstruction':True}))
    assert dataset_name_from_input(batch / 'raw') == 'c13472_rgbd_open'
    assert dataset_name_from_input(batch / 'scene') == 'c13472_rgbd_open'
    (base / 'hkqai-instruments/catalog.json').write_text(json.dumps({'datasets': {
        'c13472_rgbd_open': {'raw_dir':str((batch/'raw').relative_to(tmp_path))},
        'c13472_photo_mixed_20261102': {'raw_dir':'unused','eligible_for_rigid_reconstruction':False}}}))
    assert dataset_input_dir('c13472_rgbd_open',base) == batch/'raw'
    assert dataset_input_dir('c13472_rgbd_open',base/'hkqai-instruments') == batch/'raw'
    assert dataset_input_dir('other',base) == base/'other'
    with pytest.raises(ValueError,match='mixed'):
        dataset_input_dir('c13472_photo_mixed_20261102',base)
    (batch/'dataset.json').write_text(json.dumps({'run_id':'mixed','eligible_for_rigid_reconstruction':False}))
    with pytest.raises(ValueError,match='mixed'):
        dataset_name_from_input(batch/'raw')


def test_publication_cleanup_is_repeatable_and_preserves_assets(tmp_path):
    from src.instruments.layout import normalize
    for name, content in {
        'preview.png': b'actual preview', 'preview_textured.png': b'actual preview',
        'meshes/visual_closed.glb': b'visual', 'meshes/closed.glb': b'visual',
        'meshes/collision_closed.glb': b'collision',
        'urdf/c13534.urdf': b'robot', 'review.json': b'{}',
        'notes.txt': b'preserve unrecognized user file',
    }.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    normalize(tmp_path, 'c13534')
    first = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    normalize(tmp_path, 'c13534')
    assert first == {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert first['previews/assembly.png'] == b'actual preview'
    assert first['meshes/visual_closed.glb'] == b'visual'
    assert first['meshes/collision_closed.glb'] == b'collision'
    assert first['urdf/robot.urdf'] == b'robot'
    assert first['notes.txt'] == b'preserve unrecognized user file'
    assert 'meshes/closed.glb' not in first


def test_cleanup_refuses_to_delete_a_differing_alias(tmp_path):
    from src.instruments.layout import normalize
    (tmp_path / 'meshes').mkdir()
    (tmp_path / 'meshes/visual_closed.glb').write_bytes(b'canonical')
    alias = tmp_path / 'meshes/closed.glb'
    alias.write_bytes(b'possibly edited')
    with pytest.raises(ValueError, match='Refusing to discard'):
        normalize(tmp_path, 'cary60')
    assert alias.read_bytes() == b'possibly edited'


def test_split_share_keeps_models_portable_and_sources_image_only(tmp_path, monkeypatch):
    import hashlib
    import json
    import zipfile
    from PIL import Image
    from src.instruments import sharing

    monkeypatch.setattr(sharing, 'ROOT', tmp_path)
    monkeypatch.setattr(sharing, 'MODELS', ('cary60',))
    product = tmp_path/'output/objects/hkqai-instruments/cary60/codex/products'
    (product/'openscad').mkdir(parents=True)
    (product/'openscad/assembly.scad').write_text('include <parameters_native.scad>\nuse <common.scad>\ncube(width);')
    (product/'openscad/parameters_native.scad').write_text('width=10;')
    (product/'openscad/common.scad').write_text('// helpers')
    (product/'index.html').write_text('<a href="openscad/assembly.scad">CAD</a>')
    (product/'parameters.json').write_text('{}')
    for name in ('assembly.png','overview.png'):
        target=product/name
        target.parent.mkdir(exist_ok=True)
        Image.new('RGB',(80,60),'white').save(target)
    dataset=tmp_path/'dataset/hkqai-instruments/cary60'
    raw=dataset/'user_captures/2026-11-02/raw/IMG_9914.png'
    raw.parent.mkdir(parents=True)
    Image.new('RGB',(80,60),'red').save(raw)
    (raw.parent/'IMG_9855.MOV').write_bytes(b'omit motion source')
    (dataset/'evidence.json').write_text(json.dumps({'geometry_sources':[
        str(raw.relative_to(tmp_path))], 'motion_sources':[str((raw.parent/'IMG_9855.MOV').relative_to(tmp_path))]}))
    (dataset/'measurements.json').write_text('{}')
    files = {str(p.relative_to(product)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in product.rglob('*') if p.is_file()}
    (product/'manifest.json').write_text(json.dumps({'format':'openscad','files': files}))
    sharing.main()  # Also checks archived hashes and that entry-point links resolve.
    with zipfile.ZipFile(tmp_path/'output/packages/instruments_models.zip') as archive:
        pages = {name for name in archive.namelist() if name.endswith('.html')}
        assert pages == {'index.html'}
        assert b'cary60/openscad/assembly.scad' in archive.read('index.html')
        assert 'cary60/openscad/common.scad' in archive.namelist()
        assert not any('/evidence/' in name or name.endswith('.json') for name in archive.namelist())
    with zipfile.ZipFile(tmp_path/'output/packages/instruments_sources.zip') as archive:
        assert archive.namelist()==['cary60/user_captures_01.jpg']
    assert raw.is_file() and (raw.parent/'IMG_9855.MOV').is_file()
    (product/'openscad/common.scad').unlink()
    with pytest.raises((AssertionError, FileNotFoundError)):
        sharing.main()
