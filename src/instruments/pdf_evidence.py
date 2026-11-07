"""Maintain page mappings for curated PDF images; PDFs retain complete pages.

Requires Poppler's pdfinfo and pdfimages executables.
Refresh retained files: python -m src.instruments.pdf_evidence --model cary60
Extract all native images explicitly: add --extract (restores missing images).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

from PIL import Image

from .paths import ROOT, MODELS

PREFIXES = {
    'hamamatsu-c13534': 'qy',
    'hamamatsu-c16361': 'tau',
    'SHSS0025E_C16361': 'tau',
    'cary60-brochure': 'cary',
    'cary60-user-guide': 'cary_manual',
    'low_5990-7789JAJP': 'cary_ja',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixels(path):
    with Image.open(path) as image:
        return image.size, hashlib.sha256(image.convert('RGBA').tobytes()).hexdigest()


def command(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def image_records(listing):
    records = []
    for line in listing.splitlines():
        fields = line.split()
        if not fields or not fields[0].isdigit():
            continue
        records.append(dict(page=int(fields[0]), image_number=int(fields[1]),
                            type=fields[2], width=int(fields[3]), height=int(fields[4]),
                            object_id=int(fields[10]), object_generation=int(fields[11]),
                            source_x_ppi=float(fields[12]), source_y_ppi=float(fields[13])))
    return records


def extract(model, root=ROOT):
    """Keep one native image per decoded pixel set; full pages stay in the PDF."""
    root = Path(root)
    folder = root / 'dataset/hkqai-instruments' / model / 'online_sources'
    folder.mkdir(parents=True, exist_ok=True)
    groups = {}
    for path in sorted(folder.glob('*.pdf')):
        groups.setdefault(sha(path), []).append(path)
    canonical = {}
    file_hashes = {}
    anchors = {p.resolve() for p in folder.rglob('*') if p.is_symlink()}
    for path in sorted((folder / 'pdf_images').glob('*.png'), key=lambda p: (p.resolve() not in anchors, str(p))):
        canonical.setdefault(pixels(path), path)
        file_hashes[path] = sha(path)
    documents = []
    prefixes = set()
    for digest, paths in groups.items():
        source = paths[0]
        prefix = PREFIXES.get(source.stem, source.stem)
        if prefix in prefixes:
            prefix += '_' + digest[:12]
        prefixes.add(prefix)
        info = command('pdfinfo', str(source))
        count = int(re.search(r'^Pages:\s+(\d+)', info, re.MULTILINE)[1])
        embedded = image_records(command('pdfimages', '-list', str(source)))
        print(f'{model}: {source.name}: {count} pages, {len(embedded)} embedded images/masks', flush=True)
        with tempfile.TemporaryDirectory(prefix='instrument_pdf_') as temp:
            stage = Path(temp)
            command('pdfimages', '-png', str(source), str(stage / 'image'))
            extracted = sorted(stage.glob('image-*.png'))
            if len(extracted) != len(embedded):
                raise ValueError(f'Incomplete image extraction for {source}')
            for record, path in zip(embedded, extracted):
                number = int(path.stem.split('-')[-1])
                with Image.open(path) as image:
                    size = image.size
                if number != record['image_number'] or size != (record['width'], record['height']):
                    raise ValueError(f'Embedded image dimensions/order differ: {path}')
                pixel_key = pixels(path)
                target = canonical.get(pixel_key, folder / 'pdf_images' / f'{prefix}-{number:03d}.png')
                if target.exists() and pixels(target) != pixel_key:
                    raise ValueError(f'Existing extraction has different contents: {target}')
                canonical[pixel_key] = target
                file_hashes.setdefault(target, sha(path))
                record.update(path=str(target.relative_to(root)), sha256=file_hashes[target], pixel_sha256=pixel_key[1])
            # Publish only after verifying the complete extraction for this document.
            for path, record in zip(extracted, embedded):
                target = root / record['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as pending:
                        pending.write(path.read_bytes())
                    Path(pending.name).replace(target)
        documents.append(dict(source=str(source.relative_to(root)), sha256=digest,
                              page_count=count, embedded_images=embedded))
    index = dict(schema_version=1, model=model, documents=documents,
                 status='complete_for_local_pdfs' if documents else 'no_local_pdf_available',
                 resolution='Embedded images retain native pixels. Identical images share one file. Open the source PDF for complete pages and vector drawings.')
    (folder / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    return index


def refresh(model, root=ROOT):
    """Index retained images only; never recreate files removed during curation."""
    root = Path(root)
    folder = root / 'dataset/hkqai-instruments' / model / 'online_sources'
    index_path = folder / 'index.json'
    old = json.loads(index_path.read_text()) if index_path.exists() else {}
    if not list(folder.glob('*.pdf')):
        index_path.unlink(missing_ok=True)
        return {'model': model, 'status': 'no_local_pdf_available'}
    retained = {}
    for path in sorted((folder / 'pdf_images').glob('*.png')):
        if path.is_file():
            retained.setdefault(pixels(path), path)
    documents = []
    indexed = set()
    for source in sorted(folder.glob('*.pdf')):
        digest = sha(source)
        previous = next((d for d in old.get('documents', []) if d['sha256'] == digest), None)
        if previous is None:
            raise ValueError(f'New or changed PDF needs explicit extraction first: {source}')
        records = []
        for record in previous['embedded_images']:
            key = ((record['width'], record['height']), record['pixel_sha256'])
            path = retained.get(key)
            if path is None:
                continue
            records.append(dict(record, path=str(path.relative_to(root)), sha256=sha(path)))
            indexed.add(path)
        total = previous.get('embedded_occurrences_in_pdf', len(previous['embedded_images']))
        documents.append(dict(source=str(source.relative_to(root)), sha256=digest,
                              page_count=previous['page_count'], embedded_images=records,
                              embedded_occurrences_in_pdf=total,
                              omitted_embedded_occurrences=total-len(records)))
    supplemental = []
    for path in sorted(set(retained.values()) - indexed):
        size, digest = pixels(path)
        supplemental.append(dict(path=str(path.relative_to(root)), width=size[0], height=size[1],
                                 sha256=sha(path), pixel_sha256=digest,
                                 role='Retained crop/reference; specification page links are in evidence.json'))
    result = dict(schema_version=2, model=model, status='curated_retained_images',
                  documents=documents, supplemental_images=supplemental,
                  resolution='Only retained images are indexed. Omitted embedded images remain available in the original PDFs. Use original PDFs for vector drawings; refreshing does not re-extract deleted files.')
    index_path.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, action='append')
    parser.add_argument('--extract', action='store_true', help='Explicitly extract all native images, including previously removed images')
    args = parser.parse_args()
    for model in args.model or MODELS:
        (extract if args.extract else refresh)(model)


if __name__ == '__main__':
    main()
