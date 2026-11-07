import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest
import trimesh

from src.assets import export_asset, save_manifest
from src.delivery import (PROFILE, build, build_one, check_scad, compact_mujoco,
                          local_file, selections, validate_relocated)
from src.export import sha256


def registered_boxes(root, *, units='m'):
    root.mkdir(parents=True)
    scene = trimesh.Scene()
    # Parts are deliberately unrelated to any production object's naming scheme.
    for name, position in [('left_wall', -.03), ('right_wall', .03), ('base', 0)]:
        mesh = trimesh.creation.box(extents=[.01, .02, .03])
        mesh.apply_translation([position, 0, 0])
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    visual, collision = root/'visual.glb', root/'hulls.glb'
    scene.export(visual); scene.export(collision)
    physics = root/'physics.json'
    physics.write_text(json.dumps({'mass_kg': .2, 'friction': [.6, .005, .0001]}))
    manifest = root/'asset.json'
    save_manifest(manifest, kind='mesh', sources=[visual],
                  variants={'arbitrary': {'visual': visual, 'collision': collision, 'physics': physics}},
                  units=units, frame='Z up', rebuild=[])
    return manifest


def test_mujoco_uses_registered_parts_and_loads_after_source_is_removed(tmp_path):
    pytest.importorskip('mujoco')
    manifest = registered_boxes(tmp_path/'source')
    raw = tmp_path/'raw'
    export_asset(manifest, raw, ['mujoco'])
    delivery = tmp_path/'delivery'; delivery.mkdir()
    compact_mujoco(raw, delivery)
    shutil.rmtree(raw); shutil.rmtree(manifest.parent)
    root = ET.parse(delivery/'model.xml').getroot()
    collision_geoms = [g for g in root.iter('geom') if g.get('group') == '3']
    assert {g.get('name') for g in collision_geoms} == {'left_wall', 'right_wall', 'base'}
    assert len(list((delivery/'meshes/visual').glob('*.obj'))) == 1
    assert len(list((delivery/'meshes/collision').glob('*.obj'))) == 3
    for asset in root.findall('asset/*'):
        if asset.get('file'):
            assert local_file(delivery, asset.get('file')).exists()
    for obj in delivery.rglob('*.obj'):
        assert 'mtllib ' not in obj.read_text()
    assert not list(delivery.rglob('*.mtl'))
    assert not list(delivery.rglob('*.stl'))
    result = validate_relocated(delivery, 'mujoco', 'model.xml', thumbnails=False)
    assert result['relocated_load'] and result['compiled'] and result['dynamic']
    assert result['mesh_assets'] == 4


def test_uncalibrated_stl_preserves_scale_and_labels_slicer_requirement(tmp_path, monkeypatch):
    manifest = registered_boxes(tmp_path/'source', units='source_units')
    def thumbnail(path, folder):
        folder.mkdir(); Image.new('RGB', (32, 32)).save(folder/'1.png')
    monkeypatch.setattr('src.delivery.mesh_thumbnails', thumbnail)
    result = build_one(manifest, 'arbitrary',
                       {'format': 'stl', 'print_scale': 'preserve_source_set_in_slicer'},
                       tmp_path/'delivery', 'bird')
    delivered = trimesh.load(tmp_path/'delivery/bird.stl', force='mesh')
    np.testing.assert_allclose(delivered.extents, [.07, .02, .03], atol=1e-8)
    assert result['units'] == 'source_units'
    assert result['checks']['watertight']
    assert 'NOT calibrated millimetres' in (tmp_path/'delivery/README.txt').read_text()
    assert not list((tmp_path/'delivery').rglob('*.glb'))


def test_scad_rejects_external_dependency(tmp_path):
    folder = tmp_path/'model'; folder.mkdir()
    (tmp_path/'external.scad').write_text('cube(1);')
    (folder/'assembly.scad').write_text('include <../external.scad>')
    with pytest.raises(ValueError, match='external'):
        check_scad(folder)


def test_profile_scope_and_variant_selection():
    entries = list(selections(json.loads(PROFILE.read_text())))
    assert len(entries) == 16
    assert all(family != 'hkqai-trial' for family, *_ in entries)
    assert sum(e[4]['format'] == 'scad' for e in entries) == 4
    assert sum(e[4]['format'] == 'mujoco' for e in entries) == 10
    assert sum(e[4]['format'] == 'stl' for e in entries) == 2
    assert all(e[2].endswith('_rl') for e in entries if e[1].startswith('plate_'))
    balances = [e for e in entries if e[1] == 'electronic_balance']
    assert {e[3].name for e in balances} == {'electronic_balance_round', 'electronic_balance_square'}


def test_missing_source_does_not_publish_partial_collection(tmp_path):
    destination = tmp_path/'delivery'
    with pytest.raises(FileNotFoundError):
        build(tmp_path/'empty_workspace', destination, only=['cary60'])
    assert not destination.exists()
    assert not list(tmp_path.glob('.delivery_*'))


def test_scad_delivery_checks_snapshot_and_preserves_source(tmp_path):
    workspace = tmp_path/'workspace'
    folder = workspace/'output/objects/hkqai-instruments/cary60/codex'
    product = folder/'products'; product.mkdir(parents=True)
    sources = {}
    for name, code in {'assembly.scad': 'include <parameters_native.scad>\ncube(size);',
                       'parameters_native.scad': 'size=10;'}.items():
        path = workspace/name; path.write_text(code); sources[name] = path
    manifest = folder/'asset.json'
    save_manifest(manifest, kind='openscad', sources=sources.values(), units='mm', frame='Z up',
                  variants={'cary60': {}}, native_files=sources, rebuild=[])
    published = {f'openscad/{name}': {'sha256': sha256(path)} for name, path in sources.items()}
    for name in ['assembly.png', 'overview.png']:
        Image.new('RGB', (32, 32)).save(product/name)
        published[name] = {'sha256': sha256(product/name)}
    (product/'manifest.json').write_text(json.dumps({'files': published}))
    destination = tmp_path/'delivery'
    build(workspace, destination, only=['cary60'])
    obj = destination/'hkqai-instruments/cary60'
    assert (obj/'openscad/assembly.scad').read_bytes() == sources['assembly.scad'].read_bytes()
    assert (obj/'thumbnails/1.png').is_file() and (obj/'thumbnails/2.png').is_file()
    assert (obj/'preview.jpg').is_file() and (destination/'preview.jpg').is_file()
    assert 'data:image/png;base64,' in (obj/'index.html').read_text()
    assert 'data:image/png;base64,' in (destination/'index.html').read_text()
    assert not list(obj.rglob('*.stl')) and not list(obj.rglob('*.glb'))
    metadata = json.loads((obj/'asset.json').read_text())
    for name, digest in metadata['files'].items():
        assert sha256(obj/name) == digest
    # A changed preview is rejected even when the authoring sources still match.
    Image.new('RGB', (32, 32), 'red').save(product/'assembly.png')
    with pytest.raises(ValueError, match='Stale preview'):
        build(workspace, tmp_path/'bad_delivery', only=['cary60'])
    assert not (tmp_path/'bad_delivery').exists()
