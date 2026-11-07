"""Load editable authoring inputs without requiring previous generated outputs."""
import json
from pathlib import Path
from .paths import ROOT

DIRECTORY = Path(__file__).with_name('inputs')


def read(model, filename):
    data = json.loads((DIRECTORY / model / filename).read_text())
    if 'dataset_file' in data:
        path = Path(data['dataset_file'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('dataset_file must be relative to the workspace')
        data = json.loads((ROOT / path).read_text())
    if data.get('model') != model:
        raise ValueError(f'{filename}: model must be {model}')
    return data


def initial_parameters(model):
    data = read(model, 'design_defaults.json')
    return {**data['metadata'], 'model': model, 'detail_parameters': {},
            'design_parameters': data['values'], 'design_parameter_evidence': data['evidence']}


def evidence(model):
    data = read(model, 'evidence.json')
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported evidence schema')
    for patch in data['materials'].values():
        name = Path(patch['review_image'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('review_image must be relative to the generated evidence directory')
        x0, y0, x1, y1 = patch['crop_pixels']
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError('Invalid material crop rectangle')
    return data


def stage_evidence(model, out):
    """Generate package review images from canonical photos, never from dataset previews."""
    from PIL import Image, ImageOps

    config = evidence(model)
    dest = Path(out) / 'evidence'
    dest.mkdir(parents=True, exist_ok=True)
    images = {Path(source).stem + '.jpg': (source, None)
              for source in config.get('geometry_sources', [])}
    images.update({name: (source, None)
                   for name, source in config.get('reference_images', {}).items()})
    for patch in config['materials'].values():
        images[patch['review_image']] = (patch['original'], patch.get('review_size'))
    for name, (source, size) in images.items():
        if Path(name).name != name:
            raise ValueError('Generated review image names must be filenames')
        with Image.open(ROOT / source) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            if size:
                if abs(image.width / image.height - size[0] / size[1]) > .01:
                    raise ValueError(f'Review aspect ratio differs from source: {source}')
                image = image.resize(tuple(size), Image.Resampling.LANCZOS)
            else:
                image.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
            options = {'icc_profile': image.info['icc_profile']} if image.info.get('icc_profile') else {}
            image.save(dest / name, quality=95, subsampling=0, **options)
    return dest
