from pathlib import Path
import pytest
from src.output_layout import products_dir, intermediate_dir, navigation
from helpers import dataset_name_from_input, mesh_glb_path
from pipeline import resolve_input_scene


@pytest.mark.parametrize('name,family', [('sand_container','hestia-rl'),('electronic_balance','hestia-rl'),
                                       ('hestia_bowl_v1','hestia-rl'),('plate_strimmig_pale_grey_green_21cm','hestia-rl'),('bird_swallow','hestia-rl'),('cary60','hkqai-instruments')])
def test_explicit_roots_never_duplicate_layers(tmp_path, name, family):
    output=tmp_path/'output';obj=output/family/name
    for root in [output, output/'objects', output/family, obj, obj/'codex', obj/'codex/products']:
        assert products_dir(root,name,'codex')==obj
    assert intermediate_dir(obj/'tsdf',name)==tmp_path/'work'/family/name/'tsdf'
    assert intermediate_dir(tmp_path/'work'/family/name/'tsdf',name)==tmp_path/'work'/family/name/'tsdf'


def test_reconstruction_products_separate_methods(tmp_path):
    a=mesh_glb_path(tmp_path/'output','bowl','da3_sam3d')
    b=mesh_glb_path(tmp_path/'output','bowl','da3_tsdf')
    assert a==tmp_path/'output/hestia-rl/hestia_bowl_v1/sam3d/da3_sam3d_mesh.glb'
    assert b==tmp_path/'output/hestia-rl/hestia_bowl_v1/tsdf/da3_tsdf_mesh.glb'


def test_navigation_links_directly_to_shallow_current_model(tmp_path):
    import json
    folder=tmp_path/'hestia-rl/plate';folder.mkdir(parents=True)
    (folder/'plate.step').write_text('CAD')
    (folder/'preview.png').write_bytes(b'preview')
    (folder/'asset.json').write_text(json.dumps({'variants':{'plate':{'step':{'path':'plate.step'}}}}))
    navigation(tmp_path)
    page=(tmp_path/'index.html').read_text()
    assert 'hestia-rl/plate/plate.step' in page and 'hestia-rl/plate/preview.png' in page
    assert not (folder/'index.html').exists()


def test_scene_identity_and_navigation(tmp_path):
    output=tmp_path/'output';obj=output/'hestia-rl/hestia_bowl_v1'
    scene=tmp_path/'work/hestia-rl/hestia_bowl_v1/sam3d/scene'
    (scene/'images').mkdir(parents=True);(scene/'masks').mkdir()
    obj.mkdir(parents=True);(obj/'index.html').write_text('Accepted product page')
    assert dataset_name_from_input(scene)=='hestia_bowl_v1'
    assert resolve_input_scene(obj,'masks')==(scene,'hestia_bowl_v1')
    navigation(output)
    assert (obj/'index.html').read_text()=='Accepted product page'
    overview=(output/'index.html').read_text()
    assert 'hestia-rl/hestia_bowl_v1/index.html' in overview
    assert not (obj/'codex').exists()
    assert not (obj/'products').exists()
    assert not (obj/'model').exists()
    assert not (obj.parent/'index.html').exists()
    assert not (output/'objects').exists()


def test_navigation_stays_compact_and_links_resolve(tmp_path):
    import re
    output=tmp_path/'output';product=output/'misc/sample/sam3d'
    product.mkdir(parents=True);(product/'mesh.glb').write_bytes(b'mesh')
    intermediate=tmp_path/'work/misc/sample/sam3d/experiments/run/depth'
    intermediate.mkdir(parents=True);(intermediate/'depth.npz').write_bytes(b'depth')
    for name in ('comparisons','maintenance','packages'):(output/name).mkdir()
    (output/'packages/models.zip').write_bytes(b'archive')
    (output/'comparisons/review.html').write_text('<h1>Custom review</h1>')
    navigation(output)
    first={p.relative_to(output):p.read_bytes() for p in output.rglob('*.html')}
    navigation(output/'objects')
    assert first=={p.relative_to(output):p.read_bytes() for p in output.rglob('*.html')}
    assert set(first)=={Path('index.html'),Path('comparisons/review.html')}
    overview=(output/'index.html').read_text()
    assert 'misc/sample/sam3d/mesh.glb' in overview and 'packages/models.zip' in overview
    assert not list((tmp_path/'work').rglob('*.html'))
    for page in output.rglob('*.html'):
        for link in re.findall(r'href="([^"]+)"',page.read_text()):
            assert (page.parent/link).exists(),(page,link)


def test_navigation_does_not_rewrite_or_link_legacy_generated_product_page(tmp_path):
    product=tmp_path/'misc/sample/sam3d';product.mkdir(parents=True)
    legacy=product/'index.html';legacy.write_text('<!-- data-output-navigation --><h1>Old listing</h1>')
    (product/'mesh.glb').write_bytes(b'mesh')
    navigation(tmp_path)
    assert legacy.read_text()=='<!-- data-output-navigation --><h1>Old listing</h1>'
    overview=(tmp_path/'index.html').read_text()
    assert 'misc/sample/sam3d/mesh.glb' in overview and 'sam3d/index.html' not in overview


def test_navigation_empty_root_creates_only_overview(tmp_path):
    navigation(tmp_path)
    assert list(tmp_path.iterdir())==[tmp_path/'index.html']


def test_instrument_states_keep_independent_work_and_outputs(tmp_path):
    name='cary60_open_01'
    assert products_dir(tmp_path/'output',name,'sam3d')==tmp_path/'output/hkqai-instruments/cary60/sam3d'/name
    scene=intermediate_dir(tmp_path/'output',name)/'scene'
    assert scene==tmp_path/'work/hkqai-instruments/cary60/sam3d'/name/'scene'
    assert dataset_name_from_input(scene)==name
