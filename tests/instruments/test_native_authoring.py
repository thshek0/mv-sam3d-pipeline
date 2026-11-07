"""Regress the direct-SCAD workflow and functional simplification contract."""
import shutil

import numpy as np
import pytest

from src.instruments import native_build as native


@pytest.fixture
def source(tmp_path):
    if not shutil.which('openscad'):
        pytest.skip('OpenSCAD engine required')
    folder = tmp_path / 'scad'
    native.stage_source('cary60', folder)
    return folder


def test_component_selection_does_not_include_unrelated_groups(source, tmp_path):
    mesh = native.export_mesh(source/'assembly.scad', tmp_path, 'socket', part='socket_shell')
    np.testing.assert_allclose(mesh.extents, [27, 27, 41], atol=.001)
    assert mesh.euler_number == 0
    with pytest.raises(RuntimeError):
        native.export_mesh(source/'assembly.scad', tmp_path, 'missing', part='not_a_part')


def test_native_topology_and_parameter_expressions_are_authoritative(source, tmp_path):
    parameters = source/'parameters_native.scad'
    parameters.write_text(parameters.read_text().replace('socket_height_mm = 41;', 'socket_height_mm = 20 + 21;'))
    assert native.read_parameters(source, tmp_path)['socket_height_mm'] == 41
    assembly = source/'assembly.scad'
    assembly.write_text(assembly.read_text()+'\ncomponent("new_local_component") cube([7,11,13]);\n')
    mesh = native.export_mesh(assembly, tmp_path, 'new_component', part='new_local_component')
    np.testing.assert_allclose(mesh.extents, [7, 11, 13])


def test_zero_and_diagonal_corner_profiles_have_exact_bounds(source, tmp_path):
    probe = tmp_path/'profile.scad'
    for radii in ('[0,0,0,0]', '[0,4,0,4]'):
        probe.write_text('include <'+str(source/'common.scad')+'>\n'
                         'fillets=false; simplified=true;\n'
                         f'linear_extrude(2) corner_rectangle([20,30],{radii});\n')
        mesh = native.export_mesh(probe,tmp_path,'profile')
        np.testing.assert_allclose(mesh.bounds,[[0,0,0],[20,30,2]],atol=.001)


def test_simplification_preserves_tau_nine_bores(tmp_path):
    if not shutil.which('openscad'):
        pytest.skip('OpenSCAD engine required')
    folder=tmp_path/'scad'; native.stage_source('c16361',folder)
    a=native.export_mesh(folder/'assembly.scad',tmp_path,'detailed',part='perforated_floor')
    b=native.export_mesh(folder/'assembly.scad',tmp_path,'simple',part='perforated_floor',simplified=True)
    assert a.euler_number == b.euler_number == -16
    np.testing.assert_allclose(a.volume,b.volume,rtol=1e-6)
