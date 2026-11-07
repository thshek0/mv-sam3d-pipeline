"""Check native extraction without recreating redundant images or page renders."""
from pathlib import Path
import shutil

from PIL import Image
import pytest

from src.instruments.pdf_evidence import extract, refresh


@pytest.mark.skipif(any(shutil.which(tool) is None for tool in ('pdfimages', 'pdfinfo')),
                    reason='Poppler executables are required')
def test_extracts_all_pages_and_native_images_once_per_document(tmp_path):
    folder = tmp_path / 'dataset/hkqai-instruments/test/online_sources'
    docs = folder
    docs.mkdir(parents=True)
    source = docs / 'reference.pdf'
    Image.new('RGB', (30, 60), 'red').save(
        source, 'PDF', resolution=30, save_all=True,
        append_images=[Image.new('RGB', (60, 30), 'blue')])
    shutil.copy2(source, folder / 'reference_alias.pdf')
    result = extract('test', root=tmp_path)
    assert result['status'] == 'complete_for_local_pdfs'
    assert len(result['documents']) == 1
    doc = result['documents'][0]
    assert doc['page_count'] == 2
    assert [(r['width'], r['height']) for r in doc['embedded_images']] == [(30, 60), (60, 30)]
    assert 'pages' not in doc
    assert not (folder / 'pages').exists()
    for record in doc['embedded_images']:
        assert (tmp_path / record['path']).is_file()
    assert extract('test', root=tmp_path) == result
    assert len(list((folder / 'pdf_images').glob('*.png'))) == 2
    # A curated deletion must stay deleted when the index is refreshed.
    removed = tmp_path / doc['embedded_images'][0]['path']
    removed.unlink()
    curated = refresh('test', root=tmp_path)
    assert curated['status'] == 'curated_retained_images'
    assert len(curated['documents'][0]['embedded_images']) == 1
    assert curated['documents'][0]['omitted_embedded_occurrences'] == 1
    assert not removed.exists()
    assert refresh('test', root=tmp_path) == curated


def test_absent_pdf_is_reported_without_fabricated_extractions(tmp_path):
    result = extract('test', root=tmp_path)
    assert result['status'] == 'no_local_pdf_available'
    assert result['documents'] == []
