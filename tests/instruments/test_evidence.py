"""Published evidence must describe sources without generating history artifacts."""
import hashlib
import json

from src.instruments import inputs, publish


def test_evidence_records_sources_and_authoring_inputs_without_history(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, 'ROOT', tmp_path)
    config = {
        'model': 'cary60', 'schema_version': 1,
        'geometry_sources': ['dataset/hkqai-instruments/cary60/user_captures/front.jpg'],
        'materials': {},
        'geometry_policy': {'primary': 'Use the matching manufacturer specification drawing first.'},
        'specification_sources': [{'file': 'spec.pdf', 'pdf_pages': [8]}],
    }
    monkeypatch.setattr(inputs, 'evidence', lambda model: config)
    source = tmp_path / config['geometry_sources'][0]
    source.parent.mkdir(parents=True)
    source.write_bytes(b'original photograph')
    annotation = source.with_suffix('.json')
    annotation.write_text('{"shapes": []}')
    depth = source.parent / 'depth.npz'
    depth.write_bytes(b'measured depth')
    measurements = source.parent.parent / 'measurements.json'
    measurements.write_text('{"units": "mm"}')
    out = tmp_path / 'products'
    (out / 'evidence').mkdir(parents=True)

    publish.evidence('cary60', out)
    first = (out / 'evidence/source_inventory.json').read_bytes()
    publish.evidence('cary60', out)

    assert (out / 'evidence/source_inventory.json').read_bytes() == first
    assert {p.name for p in (out / 'evidence').iterdir()} == {'source_inventory.json'}
    inventory = json.loads(first)
    assert inventory['used_evidence'] == config
    assert inventory['priority'] == config['geometry_policy']['primary']
    assert inventory['specification_sources'] == config['specification_sources']
    records = {r['path']: r for r in inventory['sources']}
    assert set(records) == {str(p.relative_to(tmp_path)) for p in (source, annotation, depth, measurements)}
    for path, record in records.items():
        assert record['sha256'] == hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()


def test_publishing_evidence_does_not_write_into_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, 'ROOT', tmp_path)
    monkeypatch.setattr(inputs, 'evidence', lambda model: {'model': model})
    dataset = tmp_path / 'dataset/hkqai-instruments/cary60'
    dataset.mkdir(parents=True)
    (dataset / 'evidence.json').write_text('{"model": "cary60"}')
    before = {str(p): p.read_bytes() for p in dataset.rglob('*') if p.is_file()}
    product = tmp_path / 'products'
    product.mkdir()
    publish.evidence('cary60', product)
    first = (product / 'evidence/source_inventory.json').read_bytes()
    publish.evidence('cary60', product)
    assert (product / 'evidence/source_inventory.json').read_bytes() == first
    assert {str(p): p.read_bytes() for p in dataset.rglob('*') if p.is_file()} == before


def test_dataset_is_authoritative_for_evidence_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(inputs, 'ROOT', tmp_path)
    monkeypatch.setattr(inputs, 'DIRECTORY', tmp_path / 'src/inputs')
    pointer = tmp_path / 'src/inputs/test/evidence.json'
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({'model': 'test', 'dataset_file': 'dataset/hkqai-instruments/test/evidence.json'}))
    source = tmp_path / 'dataset/hkqai-instruments/test/evidence.json'
    source.parent.mkdir(parents=True)
    config = {'model': 'test', 'schema_version': 1, 'materials': {}, 'geometry_sources': ['front.jpg']}
    source.write_text(json.dumps(config))
    assert inputs.evidence('test') == config
    config['geometry_sources'].append('back.jpg')
    source.write_text(json.dumps(config))
    assert inputs.evidence('test') == config


def test_review_images_are_generated_from_canonical_photos_only_in_output(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(inputs, 'ROOT', tmp_path)
    source = tmp_path / 'dataset/hkqai-instruments/test/user_captures/2026-11-02/raw/front.png'
    source.parent.mkdir(parents=True)
    Image.new('RGB', (400, 600), (100, 120, 140)).save(source)
    original = source.read_bytes()
    config = {'model': 'test', 'schema_version': 1,
              'geometry_sources': [str(source.relative_to(tmp_path))],
              'materials': {'white': {'original': str(source.relative_to(tmp_path)),
                                      'review_image': 'front.jpg', 'review_size': [100, 150],
                                      'crop_pixels': [10, 20, 50, 60]}}}
    monkeypatch.setattr(inputs, 'evidence', lambda model: config)
    dest = inputs.stage_evidence('test', tmp_path / 'output')
    with Image.open(dest / 'front.jpg') as image:
        assert image.size == (100, 150)
        assert image.crop(tuple(config['materials']['white']['crop_pixels'])).size == (40, 40)
    assert source.read_bytes() == original
    assert not (tmp_path / 'dataset/hkqai-instruments/test/review').exists()
