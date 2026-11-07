import numpy as np
import trimesh

from src.products_mujoco import inertial_properties, load_parts


def test_compound_inertia_matches_a_single_uniform_box():
    # Adjacent halves must have the same inertia as one complete box,
    # including the parallel-axis contribution and a translated origin.
    parts = {}
    for side in (-1, 1):
        mesh = trimesh.creation.box(extents=[1, 3, 4])
        mesh.apply_translation([5 + side*.5, -2, 7])
        parts[str(side)] = mesh
    center, inertia = inertial_properties(parts, 12)
    np.testing.assert_allclose(center, [5, -2, 7], atol=1e-12)
    np.testing.assert_allclose(inertia, np.diag([25, 20, 13]), atol=1e-10)


def test_scene_instances_keep_their_separate_transforms(tmp_path):
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(), geom_name='shared', node_name='first')
    transform = trimesh.transformations.translation_matrix([2, 3, 4])
    scene.graph.update(frame_to='second', matrix=transform, geometry='shared')
    path = tmp_path/'instances.glb'
    scene.export(path)
    parts = load_parts(path)
    assert set(parts) == {'first', 'second'}
    np.testing.assert_allclose(parts['second'].center_mass-parts['first'].center_mass, [2, 3, 4])
