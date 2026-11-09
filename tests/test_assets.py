import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from src.assets import export_asset, save_manifest
from src.export import export_visual


def asset(tmp_path, *, units='m', physics=None):
    visual = tmp_path/'visual.glb'
    collision = tmp_path/'collision.glb'
    parts = trimesh.Scene()
    for i in range(2):
        box = trimesh.creation.box(extents=[.01, .02, .03])
        box.apply_translation([i*.04, 0, .015])
        box.visual.face_colors = [20, 120, 220, 255]
        parts.add_geometry(box, node_name=f'part_{i}', geom_name=f'part_{i}')
    parts.export(visual)
    parts.export(collision)
    variant = dict(visual=visual, collision=collision)
    if physics is not None:
        path = tmp_path/'physics.json'
        path.write_text(json.dumps(physics))
        variant['physics'] = path
    manifest = tmp_path/'asset.json'
    save_manifest(manifest, kind='mesh', sources=[visual], variants={'boxes': variant},
                  units=units, frame='Z up', rebuild=[])
    return manifest, visual


def test_glb_copy_is_exact_and_has_no_conversion_dependencies(tmp_path, monkeypatch):
    manifest, source = asset(tmp_path)
    import src.appearance
    monkeypatch.setattr(src.appearance, 'load_surface', lambda _: pytest.fail('GLB should not be decoded'))
    destination = tmp_path/'delivery'
    export_asset(manifest, destination, ['glb'])
    assert (destination/'boxes.glb').read_bytes() == source.read_bytes()
    assert {p.name for p in destination.iterdir()} == {'boxes.glb', 'manifest.json'}


@pytest.mark.parametrize('formats,expected', [(['stl'], {'.stl'}),
                                           (['obj'], {'.obj', '.mtl', '.png'})])
def test_only_requested_visual_formats_are_generated(tmp_path, formats, expected):
    manifest, _ = asset(tmp_path)
    destination = tmp_path/'delivery'
    report = export_asset(manifest, destination, formats)
    assert {p.suffix for p in destination.iterdir() if p.name != 'manifest.json'} == expected
    mesh = trimesh.load(destination/f'boxes.{formats[0]}', force='mesh')
    np.testing.assert_allclose(mesh.extents, [.05, .02, .03], atol=1e-8)
    assert report['units'] == 'm'
    assert report['exporter_sha256'] and report['versions']


def test_changed_source_and_existing_delivery_are_not_silently_used(tmp_path):
    manifest, source = asset(tmp_path)
    destination = tmp_path/'delivery'
    export_asset(manifest, destination, ['glb'])
    with pytest.raises(ValueError, match='new or empty'):
        export_asset(manifest, destination, ['obj'])
    before = source.read_bytes()
    with pytest.raises(ValueError, match='overwrite the source'):
        export_visual(source, source.parent, source.stem)
    assert source.read_bytes() == before
    source.write_bytes(before+b'changed')
    with pytest.raises(ValueError, match='Source changed'):
        export_asset(manifest, tmp_path/'other', ['obj'])
    assert not (tmp_path/'other').exists()


def test_unsupported_cad_conversion_does_not_create_delivery(tmp_path):
    manifest, _ = asset(tmp_path)
    for requested in ['step', 'scad']:
        with pytest.raises(ValueError):
            export_asset(manifest, tmp_path/requested, [requested])
        assert not (tmp_path/requested).exists()


def test_native_scad_dependencies_are_copied_exactly(tmp_path):
    files = {}
    for name, text in {'assembly.scad': 'include <parameters.scad>\ncube(size);',
                       'parameters.scad': 'size=10;'}.items():
        source = tmp_path/name
        source.write_text(text)
        files[name] = source
    manifest = tmp_path/'asset.json'
    save_manifest(manifest, kind='openscad', sources=files.values(), variants={'box': {}},
                  native_files=files, units='mm', frame='Z up', rebuild=[])
    destination = tmp_path/'delivery'
    export_asset(manifest, destination, ['scad'])
    for name, source in files.items():
        assert (destination/'openscad'/name).read_bytes() == source.read_bytes()
    assert not list(destination.rglob('*.stl'))


def test_mujoco_preserves_hulls_and_configured_inertia_without_extra_stls(tmp_path):
    physics = dict(mass_kg=2., friction=[.2, .003, .0002],
                   center_of_mass_m=[.02, 0, .015], inertia_kg_m2=np.diag([.1, .15, .2]).tolist())
    manifest, _ = asset(tmp_path, physics=physics)
    destination = tmp_path/'delivery'
    report = export_asset(manifest, destination, ['mujoco'])
    model_path = destination/'mujoco/collision/model.xml'
    model = ET.parse(model_path).getroot()
    body = model.find('worldbody/body')
    hulls = [g for g in body.findall('geom') if g.get('group') == '3']
    assert len(hulls) == 2 and all(g.get('friction') == '0.2 0.003 0.0002' for g in hulls)
    assert body.find('freejoint') is not None
    np.testing.assert_allclose(np.fromstring(body.find('inertial').get('fullinertia'), sep=' '), [.1, .15, .2, 0, 0, 0])
    assert body.find('inertial').get('mass') == '2.0'
    for entry in model.findall('asset/*'):
        if entry.get('file'):
            assert (model_path.parent/entry.get('file')).is_file()
    assert not list(destination.rglob('*.stl'))
    assert not list(destination.rglob('*.glb'))
    assert report['checks']['mujoco']['mujoco_runtime_tested'] is False


def test_mujoco_rejects_uncalibrated_units_and_rolls_back_bad_physics(tmp_path):
    manifest, _ = asset(tmp_path, units='source_units')
    with pytest.raises(ValueError, match='metre units'):
        export_asset(manifest, tmp_path/'delivery', ['mujoco'])
    manifest, _ = asset(tmp_path, physics={'mass_kg': -1, 'friction': [.6, .005, .0001]})
    with pytest.raises(ValueError, match='Mass'):
        export_asset(manifest, tmp_path/'delivery', ['mujoco'])
    assert not (tmp_path/'delivery').exists()
    assert not list(tmp_path.glob('.asset_export_*'))


@pytest.mark.parametrize('relative,kind', [('object/codex/asset.json','recipe'),
                                          ('object/asset.json','radial_profile_cad'),
                                          ('object/simulation/asset.json','radial_profile')])
def test_codex_registration_rejects_experimental_source_and_symlink(tmp_path, relative, kind):
    source=tmp_path/'object/sam3d/products/source.glb'
    source.parent.mkdir(parents=True);source.write_bytes(b'source')
    stable=tmp_path/'inputs/source.glb';stable.parent.mkdir();stable.symlink_to(source)
    manifest=tmp_path/relative
    for candidate in [source,stable]:
        with pytest.raises(ValueError,match='stable authoring input'):
            save_manifest(manifest,kind=kind,sources=[candidate],variants={},units='m',frame='Z up',rebuild=[])
    assert not manifest.exists()
    stable.unlink();stable.write_bytes(b'independent source')
    data=save_manifest(manifest,kind=kind,sources=[stable],variants={},units='m',frame='Z up',rebuild=[])
    assert data['dependency_policy']['experimental_method_inputs_allowed'] is False
