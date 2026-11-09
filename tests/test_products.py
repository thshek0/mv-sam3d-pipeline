import json
import numpy as np
import pytest

from src.products import INPUTS, bowl, container, scale, export_scene, validate, symmetry, probe_collision


@pytest.fixture(scope='module')
def designs():
    return json.loads((INPUTS/'designs.json').read_text())


@pytest.mark.parametrize('name,builder', [('bowl', bowl), ('container', container), ('scale', scale)])
def test_collision_export_geometry_and_symmetry(name, builder, designs, tmp_path):
    for variant, (visual, collision) in builder(designs[name]).items():
        path = tmp_path/f'{variant}.glb'
        export_scene(collision, path, 'collision')
        report = validate(collision, path, convex=True)
        symmetry(report['vertices'], [0] if name=='scale' else [0, 1])
        assert np.isclose(sum(m.volume for m in collision.values()),
                          sum(m.volume for m in visual.values() if m.is_volume), rtol=.01 if name=='bowl' else 1e-8)


def test_hollow_interiors_and_walls(designs):
    for parts, h, radius, floor in [(bowl(designs['bowl'])['bowl'][1], 60, 48.8, 3),
                                    (container(designs['container'])['container_open'][1], 200, 143.1, .5)]:
        assert not probe_collision(parts, [[0, 0, z] for z in np.linspace(5, h+5, 50)]).any()
        assert probe_collision(parts, [[0, 0, floor], [radius, 0, h/2]]).all()


def test_open_container_and_scale_pan(designs):
    assert list(container(designs['container']))==['container_open']
    for visual, collision in scale(designs['scale']).values():
        vertices = np.vstack([m.vertices for m in collision.values()])
        assert np.allclose(np.ptp(vertices, axis=0)*1000, [192.5, 280, 80])
        assert probe_collision(collision, [[0, 42, 79]])[0]
        assert 'display' in visual and 'display' not in collision


def test_bowl_raised_foot_and_rounded_heel(designs):
    visual, collision = bowl(designs['bowl'])['bowl']
    assert all(m.is_volume for m in visual.values())
    assert not probe_collision(collision, [[0, 0, 1], [40, 0, 1], [43, 0, 3]]).any()
    assert probe_collision(collision, [[37, 0, 1], [42, 0, 6], [45.5, 0, 11]]).all()
    # The outside radius grows quickly around the curved heel, then tapers gently.
    profile = np.array(designs['bowl']['profile_z_outer_inner_mm'])
    slope = np.diff(profile[:5, 1])/np.diff(profile[:5, 0])
    assert np.all(np.diff(slope)<0)


@pytest.mark.parametrize('name', [
    'plate_maehata_pop_blue_23cm', 'plate_strimmig_pale_grey_green_21cm',
    'plate_paradisisk_off_white_20cm',
])
def test_plate_rebuild_uses_tracked_inputs_and_exports_without_output_recipes(name, tmp_path, monkeypatch):
    from src.plates import rebuild
    from src.assets import export_asset
    from src.collision import plate_parts, validate_parts, source_digest
    from src.export import sha256
    import trimesh

    # No pre-existing product folder, parameter snapshots or profile CSVs.
    monkeypatch.setattr(rebuild, 'OUT', tmp_path/'output')
    report=rebuild.build(name)
    folder=tmp_path/'output'/name
    visual=folder/(name+'_m.glb')
    assert not (folder/'parameters.json').exists()
    assert not (folder/'dimensions.json').exists()
    assert not list(folder.glob('*.csv'))
    assert report['mesh_checks']['watertight']
    dimensions=report['dimensions']
    np.testing.assert_allclose(trimesh.load(visual,force='mesh').extents*1000,
                              [dimensions['outer_diameter_mm']]*2+[dimensions['overall_height_mm']],atol=.0001)
    parts,check=plate_parts(visual)
    validate_parts(parts)
    assert check['visual_collision_volume_relative_error'] < 1e-5
    assert check['central_cavity_probes_free'] > 0
    manifest=folder/'asset.json'
    export_asset(manifest,tmp_path/'delivery',['mujoco'],name)
    metadata=json.loads(manifest.read_text())
    assert name+'_rl' in metadata['variants']
    assert metadata['variants'][name]['collision_recipe']['sha256']==sha256(rebuild.INPUTS)
    # A changed tracked profile invalidates the collision cache.
    original=source_digest(visual)
    changed=tmp_path/'designs.json'
    designs=json.loads(rebuild.INPUTS.read_text())
    designs[name]['dimensions']['overall_height_mm'] += 1
    changed.write_text(json.dumps(designs))
    monkeypatch.setattr(rebuild,'INPUTS',changed)
    assert source_digest(visual) != original
