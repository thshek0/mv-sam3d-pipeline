"""Grouped input/output paths must agree and not duplicate explicit family roots."""
from pathlib import Path

import pytest

from helpers import dataset_family, dataset_input_dir, dataset_root, dataset_name_from_input


@pytest.mark.parametrize('name,family', [
    ('bird1_v1', 'birds'), ('bird1_v3', 'birds'), ('bird2', 'birds'),
    ('hestia_bowl', 'bowls_and_plates'), ('plate_green', 'bowls_and_plates'),
    ('plate_pink', 'bowls_and_plates'), ('plate_patterned', 'bowls_and_plates'),
    ('heater_v5', 'heater'), ('heater_v6', 'heater'),
    ('structure_before', 'structure'), ('structure_after_v2', 'structure'),
    ('gripper', 'gripper'), ('gripper_v2', 'gripper'),
])
def test_family_routing(tmp_path: Path, name, family):
    assert dataset_family(name) == family
    for base in (tmp_path, tmp_path / family):
        assert dataset_input_dir(name, base) == tmp_path / family / name
        assert dataset_root(base, name) == tmp_path / family / name
    assert dataset_name_from_input(tmp_path / family / name) == name
    assert dataset_name_from_input(tmp_path / family / name / 'scene') == name


@pytest.mark.parametrize('name', ['birdhouse', 'heater_controller', 'structure_notes', 'gripper_fixture'])
def test_unrelated_runs_stay_flat(tmp_path, name):
    assert dataset_family(name) is None
    assert dataset_input_dir(name, tmp_path) == tmp_path / name
    assert dataset_root(tmp_path, name) == tmp_path / name


@pytest.mark.parametrize('old,new', [
    ('bowl', 'hestia_bowl'), ('green_plate', 'plate_green'),
    ('pink_plate', 'plate_pink'), ('capture_test', 'plate_patterned'),
])
def test_legacy_names_resolve_to_current_folders(tmp_path, old, new):
    family = 'bowls_and_plates'
    for base in (tmp_path, tmp_path / family):
        assert dataset_input_dir(old, base) == tmp_path / family / new
        assert dataset_root(base, old) == tmp_path / family / new
    assert dataset_name_from_input(tmp_path / family / old / 'scene') == new
