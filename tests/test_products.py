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
