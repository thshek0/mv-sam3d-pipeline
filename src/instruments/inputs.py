"""Load editable authoring inputs without requiring previous generated outputs."""
import json
from pathlib import Path

DIRECTORY = Path(__file__).with_name('inputs')


def read(model, filename):
    data = json.loads((DIRECTORY / model / filename).read_text())
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
            raise ValueError('review_image must be relative to references/model_evidence')
        x0, y0, x1, y1 = patch['crop_pixels']
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError('Invalid material crop rectangle')
    return data
