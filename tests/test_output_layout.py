from pathlib import Path
import pytest
from src.output_layout import products_dir, intermediate_dir, navigation
from helpers import dataset_name_from_input, mesh_glb_path
from pipeline import resolve_input_scene


@pytest.mark.parametrize('name,family', [('sand_container','hestia-rl'),('electronic_balance','hestia-rl'),
                                       ('hestia_bowl_v1','hestia-rl'),('plate_strimmig_pale_grey_green_21cm','hestia-rl'),('bird_swallow','hestia-rl'),('cary60','hkqai-instruments')])
def test_explicit_roots_never_duplicate_layers(tmp_path, name, family):
    obj=tmp_path/'objects'/family/name
    for root in [tmp_path, tmp_path/'objects', tmp_path/'objects'/family, obj, obj/'codex', obj/'codex/products']:
        assert products_dir(root,name,'codex')==obj/'codex/products'
    assert intermediate_dir(obj/'tsdf',name)==obj/'tsdf/intermediate'
    assert intermediate_dir(obj/'tsdf/intermediate',name)==obj/'tsdf/intermediate'


def test_reconstruction_products_separate_methods(tmp_path):
    a=mesh_glb_path(tmp_path,'bowl','da3_sam3d')
    b=mesh_glb_path(tmp_path,'bowl','da3_tsdf')
    assert a==tmp_path/'objects/hestia-rl/hestia_bowl_v1/sam3d/products/da3_sam3d_mesh.glb'
    assert b==tmp_path/'objects/hestia-rl/hestia_bowl_v1/tsdf/products/da3_tsdf_mesh.glb'


def test_scene_identity_and_navigation(tmp_path):
    obj=tmp_path/'objects/hestia-rl/hestia_bowl_v1'
    scene=obj/'sam3d/intermediate/scene'
    (scene/'images').mkdir(parents=True);(scene/'masks').mkdir()
    product=obj/'codex/products';product.mkdir(parents=True)
    (product/'index.html').write_text('Accepted product page')
    assert dataset_name_from_input(scene)=='hestia_bowl_v1'
    assert resolve_input_scene(obj,'masks')==(scene,'hestia_bowl_v1')
    navigation(tmp_path)
    assert (product/'index.html').read_text()=='Accepted product page'
    overview = (tmp_path/'index.html').read_text()
    assert 'objects/hestia-rl/hestia_bowl_v1/codex/products/index.html' in overview
    assert 'objects/hestia-rl/hestia_bowl_v1/sam3d/intermediate/' in overview
    assert not (obj/'index.html').exists()
    assert not (obj/'codex/index.html').exists()
    assert not (obj/'sam3d/index.html').exists()
    assert not (obj/'sam3d/intermediate/index.html').exists()
    assert not (obj/'model').exists()
    assert not (obj.parent/'index.html').exists()
    assert not (tmp_path/'objects/index.html').exists()


def test_navigation_stays_compact_and_links_resolve(tmp_path):
    import re
    product = tmp_path/'objects/misc/sample/sam3d/products'
    product.mkdir(parents=True)
    (product/'mesh.glb').write_bytes(b'mesh')
    intermediate = product.parent/'intermediate/experiments/run/depth'
    intermediate.mkdir(parents=True)
    (intermediate/'depth.npz').write_bytes(b'depth')
    for name in ('comparisons', 'maintenance', 'packages'):
        (tmp_path/name).mkdir()
    (tmp_path/'packages/models.zip').write_bytes(b'archive')
    review = tmp_path/'comparisons/review.html'
    review.write_text('<h1>Custom review</h1>')
    navigation(tmp_path)
    first = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*.html')}
    navigation(tmp_path/'objects')
    assert first == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*.html')}
    assert set(first) == {Path('index.html'), Path('comparisons/review.html')}
    overview = (tmp_path/'index.html').read_text()
    assert 'objects/misc/sample/sam3d/products/' in overview
    assert 'packages/models.zip' in overview
    assert not list((product.parent/'intermediate').rglob('*.html'))
    assert not list((tmp_path/'maintenance').rglob('*.html'))
    for page in tmp_path.rglob('*.html'):
        for link in re.findall(r'href="([^"]+)"', page.read_text()):
            assert (page.parent/link).exists(), (page, link)


def test_navigation_does_not_rewrite_or_link_legacy_generated_product_page(tmp_path):
    product = tmp_path/'objects/misc/sample/sam3d/products'
    product.mkdir(parents=True)
    legacy = product/'index.html'
    legacy.write_text('<!-- data-output-navigation --><h1>Old listing</h1>')
    navigation(tmp_path)
    assert legacy.read_text() == '<!-- data-output-navigation --><h1>Old listing</h1>'
    overview = (tmp_path/'index.html').read_text()
    assert 'href="objects/misc/sample/sam3d/products/"' in overview
    assert 'products/index.html' not in overview


def test_navigation_empty_root_creates_only_overview(tmp_path):
    navigation(tmp_path)
    assert list(tmp_path.iterdir()) == [tmp_path/'index.html']
