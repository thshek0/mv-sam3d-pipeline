"""Instrument sharing: reusable implementation behind the instruments CLI."""
from pathlib import Path
import hashlib,json,re,stat,tempfile,zipfile
from .paths import ROOT, SOURCE_ROOT, MODELS

def main():
 """Two deliveries: lightweight native CAD, and image-only source sheets."""
 from html import escape
 from .source_sheets import generate
 packages=ROOT/'output/packages';packages.mkdir(parents=True,exist_ok=True)
 targets=[packages/'instruments_models.zip',packages/'instruments_sources.zip']
 pending=[p.with_name('.'+p.name+'.tmp') for p in targets]
 with tempfile.TemporaryDirectory(prefix='instrument_handoff_') as temp:
  temp=Path(temp);models=temp/'models';sheets=temp/'sources';work=temp/'rendered_pages'
  models.mkdir();sheets.mkdir();work.mkdir()
  page='<!doctype html><meta charset="utf-8"><title>Instrument OpenSCAD models</title><style>body{font:17px/1.5 system-ui;max-width:1100px;margin:30px auto}img{max-width:100%}</style><h1>Instrument OpenSCAD models</h1>'
  for model in MODELS:
   product=ROOT/'output/hkqai-instruments'/model
   manifest=json.loads((product/'manifest.json').read_text())
   selected=['openscad/assembly.scad','openscad/parameters_native.scad','openscad/common.scad',
             'assembly.png','overview.png']
   for name in selected:
    source=product/name;dest=models/model/name;dest.parent.mkdir(parents=True,exist_ok=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['files'][name]['sha256'],(model,name)
    dest.write_bytes(source.read_bytes())
   readme=f"""{model.upper()} / OpenSCAD viewing and editing
Open openscad/assembly.scad. Keep the three SCAD files together.
Edit dimensions in parameters_native.scad; units are millimetres and degrees.
Choose pose=closed/open. For manual motion choose pose=custom and change opening
(0 closed, 1 open). QY uses its named door/handle/selector sliders instead.
For animation enable animate_motion, then View > Animate, FPS=20, Steps=100.
Animation displays joint transforms; it does not simulate forces or collisions.
Unmeasured details and motion limits remain estimates. The native CAD is solid
coloured; previews are rendered directly from the same source.
Set simplified=true to remove cosmetic rounding and small hardware. Functional
bores, cavities and moving assemblies remain. Edit named modules in assembly.scad
for geometry; parameters_native.scad contains editable dimensions.
No external libraries, Python or imported meshes are required.
assembly.png: closed/open view. overview.png: dimension sheet.
Source images are in the separate instruments_sources.zip. The full authoritative
dataset and rebuilding metadata remain in the authoring workspace.
"""
   (models/model/'README.txt').write_text(readme)
   page+=f'<h2>{escape(model.upper())}</h2><p><a href="{model}/openscad/assembly.scad">OpenSCAD</a> · <a href="{model}/openscad/parameters_native.scad">Dimensions to edit</a> · <a href="{model}/overview.png">Dimension sheet</a> · <a href="{model}/README.txt">Instructions</a></p><img src="{model}/assembly.png">'
  (models/'index.html').write_text(page)
  (models/'README.txt').write_text('Open index.html for the four model previews, or open <model>/openscad/assembly.scad in OpenSCAD. Read the model README for motion controls. Sources are supplied separately in instruments_sources.zip.\n')
  audit=generate(ROOT,MODELS,sheets,work)
  for folder,target in zip((models,sheets),pending):
   with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(folder.rglob('*')):
     if path.is_file():archive.write(path,path.relative_to(folder))
   with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert not any(stat.S_ISLNK(info.external_attr>>16) for info in archive.infolist())
    assert set(archive.namelist())=={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    for name in archive.namelist():
     assert archive.read(name)==(folder/name).read_bytes(),name
  for path in models.rglob('*.scad'):
   for ref in re.findall(r'(?:include|use)\s*<([^>]+)>',path.read_text()):assert (path.parent/ref).is_file(),ref
  for link in re.findall(r'(?:href|src)="([^"]+)"',page):assert (models/link).is_file(),link
  from PIL import Image
  for path in sheets.rglob('*'):
   if path.is_file():
    assert path.suffix in ('.jpg','.png')
    with Image.open(path) as image:image.verify()
  audit_path=ROOT/'output/maintenance/instrument_source_sheets.json'
  audit_path.parent.mkdir(parents=True,exist_ok=True)
  audit_path.write_text(json.dumps({'scope':'Declared current evidence plus measurement/parameter photo references. Unselected catalog images, annotation layers, depth and video omitted. Diagram crops replaced by 300-dpi source pages. Photos retain native dimensions, encoded JPEG quality 94; document sheets are lossless PNG.','models':audit},indent=2)+'\n')
 for source,target in zip(pending,targets):
  source.replace(target)
  print(target,f'{target.stat().st_size/1024**2:.1f} MiB',flush=True)
 # Retire the superseded full-data handoff only after both replacements verify.
 (packages/'instruments_share.zip').unlink(missing_ok=True)
 from ..output_layout import navigation
 navigation(ROOT/'output')
 return targets

def project():
 """Share the authored pipeline and its data; no environments or generated meshes."""
 folder=ROOT/'output/packages';folder.mkdir(parents=True,exist_ok=True)
 target=folder/'instruments_project.zip';pending=folder/'.instruments_project.zip.tmp'
 files={}
 for relative in ('src/instruments','docs/instruments','tests/instruments'):
  for path in sorted((SOURCE_ROOT/relative).rglob('*')):
   if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
    files[str(path.relative_to(SOURCE_ROOT))]=path
 for relative in ('src/helpers.py','src/output_layout.py','pytest.ini','LICENSE','env/instruments-requirements.txt'):
  files[relative]=SOURCE_ROOT/relative
 for path in sorted((ROOT/'dataset/hkqai-instruments').rglob('*')):
  if path.is_file():files[str(path.relative_to(ROOT))]=path
 for model in MODELS:
  parameters=ROOT/'output/hkqai-instruments'/model/'parameters.json'
  if parameters.exists():files[str(parameters.relative_to(ROOT))]=parameters
 if not any(name.startswith('dataset/hkqai-instruments/') for name in files):
  raise FileNotFoundError('Instrument datasets are required for the project archive')
 hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
 readme='''# Instrument modeling project

This archive contains the Python build tools, all four native OpenSCAD models, editable
inputs, current parameter snapshots, documentation, tests, and instrument datasets
(photos/videos/references). It includes no installed environment, pretrained weights
or generated model meshes.
For editable OpenSCAD models use instruments_models.zip; source image sheets are in instruments_sources.zip.

Start with docs/instruments/README.md for the flowchart, setup and limitations.
Run commands from this extracted directory. Linux/Python 3.11 is the tested setup:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r env/instruments-requirements.txt
.venv/bin/python -m pytest tests/instruments -q
.venv/bin/python -m src.instruments build
.venv/bin/python -m src.instruments validate
```

Install OpenSCAD or set OPENSCAD to its executable; every build calls the engine.
Add --check-cad for extended functional-opening and motion checks.
The data root defaults to this directory; unset INSTRUMENT_WORKSPACE unless you
deliberately want an external dataset/output directory.

Edit dataset/hkqai-instruments/<machine>/evidence.json to use new photographs; review
measurements.json in that same dataset before
editing src/instruments/native/<machine>/assembly.scad and parameters_native.scad.
Additional photos are not automatically fitted to CAD. Native SCAD is authoritative;
the JSON parameter snapshot is generated output. No prior meshes are required.
The MIT license covers the source code; bundled datasets retain their provenance.
PROJECT_MANIFEST.json records SHA-256 hashes of all included source/data files.
'''
 with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as archive:
  for name,path in files.items():archive.write(path,'instruments_project/'+name)
  archive.writestr('instruments_project/README.md',readme)
  archive.writestr('instruments_project/PROJECT_MANIFEST.json',json.dumps({'schema_version':1,'files':hashes},indent=2)+'\n')
 with zipfile.ZipFile(pending) as archive:
  assert archive.testzip() is None
  for name,digest in hashes.items():
   assert hashlib.sha256(archive.read('instruments_project/'+name)).hexdigest()==digest,name
 pending.replace(target)
 print(str(target),f'{target.stat().st_size/1024**2:.1f} MiB; {len(files)} source/data files verified',flush=True)
