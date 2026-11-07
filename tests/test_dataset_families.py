"""Source names and trial grouping stay consistent across input/output paths."""
from pathlib import Path

import pytest

from helpers import dataset_family, dataset_input_dir, dataset_root, dataset_name_from_input
from src.output_layout import ALIASES, canonical_name, identity


@pytest.mark.parametrize('name', [
    'bird_swallow', 'bird_mahjong', 'bird1_v3', 'hestia_bowl_v1',
    'plate_strimmig_pale_grey_green_21cm', 'plate_paradisisk_off_white_20cm', 'plate_maehata_pop_blue_23cm', 'plate_harmynta_white_22cm',
    'electronic_balance', 'sand_container', 'spoon_home_black', 'spoon_daiso_pink',
])
def test_hestia_family_routing(tmp_path: Path, name):
    assert dataset_family(name) == 'hestia-rl'
    for base in (tmp_path, tmp_path/'hestia-rl'):
        assert dataset_input_dir(name, base) == tmp_path/'hestia-rl'/name
    for base in (tmp_path, tmp_path/'objects', tmp_path/'objects/hestia-rl'):
        assert dataset_root(base, name) == tmp_path/'objects/hestia-rl'/name/'sam3d/intermediate'
    assert dataset_name_from_input(tmp_path/'hestia-rl'/name/'scene') == name


@pytest.mark.parametrize('name,group', [
    ('coffee','coffee'), ('pc','pc'), ('heater','heater'), ('heater_legacy','heater'),
    ('heater_v5','heater'), ('heater_v6','heater'),
    ('structure_before','structure'), ('structure_after_v2','structure'),
    ('gripper','gripper'), ('gripper_v2','gripper'),
])
def test_trial_captures_keep_groups_outputs_keep_run_identity(tmp_path, name, group):
    assert dataset_family(name) == 'hkqai-trial'
    folder = tmp_path/'hkqai-trial'/group
    expected = folder if name == group else folder/name
    for base in (tmp_path, tmp_path/'hkqai-trial', folder):
        assert dataset_input_dir(name, base) == expected
    for base in (tmp_path, tmp_path/'objects', tmp_path/'objects/hkqai-trial'):
        assert dataset_root(base, name) == tmp_path/'objects/hkqai-trial'/name/'sam3d/intermediate'
    assert dataset_name_from_input(expected/'scene') == name


@pytest.mark.parametrize('name', ['birdhouse', 'heater_controller', 'structure_notes', 'gripper_fixture'])
def test_unrelated_inputs_stay_flat_outputs_use_misc(tmp_path, name):
    assert dataset_family(name) is None
    assert dataset_input_dir(name, tmp_path) == tmp_path/name
    assert dataset_root(tmp_path, name) == tmp_path/'objects/misc'/name/'sam3d/intermediate'


@pytest.mark.parametrize('old,new', list(ALIASES.items()))
def test_legacy_names_resolve_to_current_folders(tmp_path, old, new):
    for base in (tmp_path, tmp_path/'hestia-rl'):
        assert dataset_input_dir(old, base) == tmp_path/'hestia-rl'/new
    for base in (tmp_path, tmp_path/'objects', tmp_path/'objects/hestia-rl'):
        assert dataset_root(base, old) == tmp_path/'objects/hestia-rl'/new/'sam3d/intermediate'
    assert dataset_name_from_input(tmp_path/'hestia-rl'/old/'scene') == new
    assert identity(old) == ('hestia-rl', new)


def test_removed_capture_is_not_silently_redirected(tmp_path):
    for call in (lambda: canonical_name('bird1_v1'),
                 lambda: dataset_input_dir('bird1_v1', tmp_path),
                 lambda: dataset_root(tmp_path, 'bird1_v1')):
        with pytest.raises(ValueError, match='removed'):
            call()
