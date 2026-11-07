"""Output contract: objects / family / object / method / products | intermediate.

Hestia RL inputs share the same family while retaining each object's identity.
"""
from pathlib import Path
import html
import re

METHODS = frozenset(('codex', 'sam3d', 'tsdf', 'dualbrep'))
INSTRUMENTS = frozenset(('cary60', 'c13472', 'c16361', 'c13534'))
INSTRUMENT_FAMILY = 'hkqai-instruments'
ALIASES = {
    'scale': 'electronic_balance',
    'bowl': 'hestia_bowl_v1', 'hestia_bowl': 'hestia_bowl_v1',
    'green_plate': 'plate_strimmig_pale_grey_green_21cm', 'plate_green': 'plate_strimmig_pale_grey_green_21cm',
    'pink_plate': 'plate_paradisisk_off_white_20cm', 'plate_pink': 'plate_paradisisk_off_white_20cm',
    'capture_test': 'plate_maehata_pop_blue_23cm', 'plate_patterned': 'plate_maehata_pop_blue_23cm',
    'plate_new': 'plate_harmynta_white_22cm', 'Plate_new': 'plate_harmynta_white_22cm',
    'plate_strimmig': 'plate_strimmig_pale_grey_green_21cm',
    'plate_paradisisk': 'plate_paradisisk_off_white_20cm',
    'plate_maehata_pop_blue': 'plate_maehata_pop_blue_23cm',
    'plate_harmynta': 'plate_harmynta_white_22cm',
    'bird1_v2': 'bird_swallow', 'bird2': 'bird_mahjong',
    'container': 'sand_container', 'sand_bucket': 'sand_container',
    'spoon1': 'spoon_home_black', 'spoon2': 'spoon_daiso_pink',
    'spoon_home_coordy_ladle': 'spoon_home_black',
    'spoon_daiso_scoop_magical': 'spoon_daiso_pink',
}


def canonical_name(name):
    if name == 'bird1_v1':
        raise ValueError('bird1_v1 was removed; it is not an alias for the distinct bird_swallow capture.')
    return ALIASES.get(name, name)


def trial_group(name):
    """Capture-group folder inside dataset/hkqai-trial, or None."""
    if name in ('coffee', 'pc'):
        return name
    if re.fullmatch(r'gripper(?:_v\d+)?', name):
        return 'gripper'
    if re.fullmatch(r'heater(?:_v\d+|_legacy)?', name):
        return 'heater'
    if re.fullmatch(r'structure(?:_(?:before|after)(?:_v\d+)?)?', name):
        return 'structure'
    return None


def identity(name):
    name = canonical_name(name)
    machine = name.split('_')[0]
    if machine in INSTRUMENTS:
        return INSTRUMENT_FAMILY, machine
    if name in ALIASES.values() or re.fullmatch(r'bird\d+(?:_v\d+)?', name):
        return 'hestia-rl', name
    if trial_group(name):
        return 'hkqai-trial', name
    return 'misc', name


def object_dir(root, name):
    root = Path(root); family, obj = identity(name)
    if root.name in ('products', 'intermediate'): root = root.parent
    if root.name in METHODS: root = root.parent
    if root.name == obj and root.parent.name == family: return root
    if root.name == family: return root/obj
    return (root if root.name == 'objects' else root/'objects')/family/obj


def output_root(root, name):
    """Find the output browser root from an explicit object or method path."""
    parent = object_dir(root, name).parent.parent
    return parent.parent if parent.name == 'objects' else parent


def method_dir(root, name, method):
    if method not in METHODS: raise ValueError(f'Unknown output method: {method}')
    return object_dir(root, name)/method


def products_dir(root, name, method):
    directory = method_dir(root, name, method)/'products'
    # Instrument reconstruction IDs identify independent capture/state runs.
    _, obj = identity(name)
    if obj in INSTRUMENTS and name != obj: directory /= name
    return directory


def intermediate_dir(root, name, method='sam3d'):
    root = Path(root)
    if root.name in ('products', 'intermediate'): root = root.parent
    if root.name in METHODS: method = root.name
    directory = method_dir(root, name, method)/'intermediate'
    _, obj = identity(name)
    if obj in INSTRUMENTS and name != obj: directory /= name
    return directory


STYLE = '<style>body{font:16px/1.6 system-ui;max-width:1150px;margin:32px auto;padding:0 24px;color:#243546}a{color:#165f99}img{max-width:100%}li{margin:8px 0}</style>'


def navigation(root):
    """Write only output/index.html, linking directly to reviews and assets.

    Product publishers own substantive reviews. Never create directory listings
    in families, products, packages, maintenance or intermediate directories.
    """
    root = Path(root)
    if root.name == 'objects': root = root.parent
    objects_root = root/'objects'
    root.mkdir(parents=True, exist_ok=True)
    sections = []
    def section(title, entries):
        items = ''.join(f'<li><a href="{html.escape(target)}">{html.escape(label)}</a></li>' for label, target in entries)
        if items:
            sections.append('<h2>'+html.escape(title)+'</h2><ul>'+items+'</ul>')
    def target(path):
        return path.relative_to(root).as_posix() + ('/' if path.is_dir() else '')
    for family in sorted(objects_root.iterdir()) if objects_root.is_dir() else []:
        if not family.is_dir(): continue
        objects = []
        for obj in sorted(family.iterdir()):
            if not obj.is_dir(): continue
            for method in sorted(obj.iterdir()):
                if not method.is_dir() or method.name not in METHODS: continue
                folder = method/'products'
                if folder.is_dir():
                    review = folder/'index.html'
                    current = review.read_text() if review.is_file() else ''
                    generated_title = '<title>'+html.escape(f'{obj.name} / {method.name} / products')+'</title>'
                    authored = current and 'data-output-navigation' not in current and generated_title not in current
                    objects.append((f'{obj.name} / {method.name}', target(review if authored else folder)))
                if (method/'intermediate').is_dir():
                    objects.append((f'{obj.name} / {method.name} supporting files',
                                    target(method/'intermediate')))
        section(family.name, objects)
    comparisons = root/'comparisons'
    if comparisons.is_dir():
        section('Comparisons', [(p.name, target(p/'index.html' if (p/'index.html').is_file() else p))
                                for p in sorted(comparisons.iterdir())
                                if not p.name.startswith('.') and p.name != 'index.html'])
    packages = root/'packages'
    galleries = [(p.parent.name+' — delivery previews', target(p))
                 for p in sorted(packages.glob('*/index.html'))
                 if (p.parent/'catalog.json').is_file()]
    section('Shareable assets and downloads', galleries +
            [(p.name, target(p)) for p in sorted(packages.glob('*.zip'))])
    if (root/'maintenance').is_dir():
        section('Maintenance', [('Migration and validation records', 'maintenance/')])
    (root/'index.html').write_text('<!doctype html><!-- data-output-navigation --><meta charset="utf-8"><title>Model outputs</title>'+STYLE+'<h1>Model outputs</h1><p>Current products, comparisons and downloads. Original captures remain in dataset/.</p>'+''.join(sections))


if __name__ == '__main__':
    navigation(Path(__file__).resolve().parents[1]/'output')
