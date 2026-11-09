"""Image-only source sheets for the current authored instrument models."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp'}


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def selected_sources(root, model):
    """Explicit model inputs and measurement photo references, not pending catalog views."""
    dataset = root/'dataset/hkqai-instruments'/model
    config = json.loads((dataset/'evidence.json').read_text())
    metadata = [config, json.loads((dataset/'measurements.json').read_text()),
                json.loads((root/'output/hkqai-instruments'/model/'parameters.json').read_text())]
    images = set()
    for value in strings(metadata):
        if Path(value).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if value.startswith('dataset/'):
            images.add(root/value)
        elif value.startswith(('user_captures/', 'online_sources/')):
            images.add(dataset/value)
    # Measurements also identify photographs by basename or in prose.
    photo_text = '\n'.join(value for value in strings(metadata)
                           if Path(value).suffix.lower() not in {'.mov', '.mp4', '.avi', '.npz', '.bag'})
    for stem in sorted(set(re.findall(r'IMG_\d+', photo_text))):
        matches = [p for p in dataset.rglob('*'+stem+'.*')
                   if p.suffix.lower() in IMAGE_SUFFIXES and 'raw' in p.parts]
        if not matches:
            raise FileNotFoundError(f'{model}: no original photograph for {stem}')
        images.update(matches)
    pdfs = defaultdict(set)
    replaced_crops = set()
    for source in config.get('specification_sources', []):
        path = root/source['file']
        if path.suffix.lower() != '.pdf':
            continue
        pdfs[path].update(source['pdf_pages'])
        if source.get('diagram_crop'):
            replaced_crops.add(root/source['diagram_crop'])
    # A bundled document without cited pages has no narrower recorded selection.
    # Include its complete pages instead of silently dropping that reference.
    for value in config.get('package_references', {}).values():
        path = root/value
        if path.suffix.lower() == '.pdf' and path not in pdfs:
            info = subprocess.run(['pdfinfo', str(path)], capture_output=True, text=True, check=True).stdout
            count = int(re.search(r'^Pages:\s+(\d+)', info, re.M)[1])
            pdfs[path].update(range(1, count+1))
    images -= replaced_crops
    unique = {}
    for path in sorted(images):
        relative = path.relative_to(dataset)
        if not path.is_file():
            raise FileNotFoundError(path)
        # Only original photos and direct PDF image extracts. No masks/processed views.
        if 'raw' not in relative.parts and relative.parts[:2] != ('online_sources', 'pdf_images'):
            raise ValueError(f'Cannot substitute a processed image for an original: {path}')
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            key = (relative.parts[0], image.size, hashlib.sha256(image.tobytes()).hexdigest())
        unique.setdefault(key, path)
    return sorted(unique.values()), dict(pdfs)


def sheet_layout(sizes):
    cell_width = max(800, max(width for width, _ in sizes))
    cell_height = max(height for _, height in sizes)
    font_size = max(18, min(42, cell_width//50))
    columns = min(2, len(sizes))
    rows = (len(sizes)+columns-1)//columns
    size = (columns*(cell_width+font_size)+font_size,
            font_size*4+rows*(cell_height+font_size*5))
    return cell_width, cell_height, font_size, columns, size


def batches(entries, limit=4):
    """Keep native pixels while limiting large sheets to 64 megapixels."""
    batch, sizes = [], []
    for entry in entries:
        with Image.open(entry[0]) as image:
            size = image.size
            if image.getexif().get(274) in (5, 6, 7, 8):
                size = size[::-1]
        width, height = sheet_layout(sizes+[size])[-1]
        if batch and (len(batch) == limit or width*height > 64_000_000):
            yield batch
            batch, sizes = [], []
        batch.append(entry)
        sizes.append(size)
    if batch:
        yield batch


def stack_sheet(entries, destination, title, photographic=True):
    """Hstack pairs, then vstack rows; keep original pixels and add margin captions."""
    images = []
    for path, label in entries:
        with Image.open(path) as source:
            images.append((ImageOps.exif_transpose(source).convert('RGB'), label))
    cell_width, cell_height, font_size, columns, size = sheet_layout([image.size for image, _ in images])
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font_size)
    pad = font_size
    caption_height = font_size*4
    board = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(board)
    draw.text((pad, pad), title, font=font, fill='black')
    for index, (image, label) in enumerate(images):
        x = pad+(index%columns)*(cell_width+pad)
        y = pad*3+font_size+(index//columns)*(cell_height+caption_height+pad)
        board.paste(image, (x+(cell_width-image.width)//2, y))
        wrapped = textwrap.wrap(label, width=max(30, int(cell_width/(font_size*.6))))
        if len(wrapped)>3:
            raise ValueError('Source caption does not fit: '+label)
        draw.multiline_text((x, y+cell_height+pad//2), '\n'.join(wrapped), font=font, fill='black')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if photographic:
        board.save(destination, quality=94, subsampling=0, optimize=True)
    else:
        board.save(destination, optimize=True)
    return {'size_pixels': list(board.size), 'sources': [label for _, label in entries]}


def generate(root, models, output, work):
    audit = {}
    selections = {model: selected_sources(root, model) for model in models}
    for model in models:
        images, documents = selections[model]
        groups = defaultdict(list)
        for path in images:
            relative = path.relative_to(root/'dataset/hkqai-instruments'/model)
            groups[relative.parts[0]].append((path, str(relative)))
        records = []
        for origin, entries in sorted(groups.items()):
            for index, batch in enumerate(batches(entries), 1):
                name = f'{model}/{origin}_{index:02}.jpg'
                record = stack_sheet(batch, output/name,
                                     f'{model.upper()} / {origin} / original images')
                records.append({'sheet': name, **record})
        pages = []
        for path, numbers in sorted(documents.items()):
            for number in sorted(numbers):
                target = work/f'{model}_{path.stem}_page_{number:02}'
                subprocess.run(['pdftoppm', '-f', str(number), '-l', str(number), '-singlefile',
                                '-r', '300', '-png', str(path), str(target)],
                               capture_output=True, text=True, check=True)
                pages.append((target.with_suffix('.png'), f'online_sources/{path.name} / PDF page {number} / 300 dpi'))
        for start in range(0, len(pages), 2):
            name = f'{model}/online_documents_{start//2+1:02}.png'
            record = stack_sheet(pages[start:start+2], output/name,
                                 f'{model.upper()} / online documents / original pages', photographic=False)
            records.append({'sheet': name, **record})
        audit[model] = {'original_images': len(images), 'pdf_pages': len(pages), 'sheets': records}
        print(model, len(images), 'original images,', len(pages), 'PDF pages,', len(records), 'sheets', flush=True)
    return audit
