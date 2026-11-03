"""Instrument sharing: reusable implementation behind the instruments CLI."""
from pathlib import Path
import hashlib,json,os,re,stat,tempfile,zipfile,xml.etree.ElementTree as ET
from .paths import ROOT, SOURCE_ROOT, MODELS

def main():
 folder=ROOT/'output/instruments';target=folder/'instruments_share.zip';pending=folder/'.instruments_share.zip.tmp'
 with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
  for directory,subdirs,names in os.walk(folder,followlinks=True):
   subdirs[:]=[name for name in subdirs if not name.startswith('.') and name not in ('validation','experiments','model_staging','__pycache__')]
   for name in sorted(names):
    path=Path(directory)/name
    if path in (target,pending,folder/'instruments_project.zip',folder/'.instruments_project.zip.tmp'):continue
    if any(part in ('.build_work','model_staging','__pycache__') for part in path.relative_to(folder).parts):continue
    z.write(path,Path('instruments')/path.relative_to(folder))
  z.writestr('instruments/SHARING_README.txt','Open index.html for the four-machine overview. Each machine/model contains GLBs, URDF, previews and dimensions. ZIPs and validation scratch are excluded from this snapshot. Evidence links have been replaced by regular files. No original dataset folder is needed to view or load the models.\nUse metres and a fixed base. For QY use the supplied urdf/load_pybullet.py to apply self-contact filters. Read each package reports/limitations.json before simulation.\nThis is a viewing/simulation package. Regenerating geometry requires the source repository, authoring inputs and original captures; provenance paths refer to that workspace.\n')
 with tempfile.TemporaryDirectory(prefix='instruments_share_',dir='/tmp') as tmp:
  with zipfile.ZipFile(pending) as z:
   assert z.testzip() is None
   assert all(not stat.S_ISLNK(info.external_attr>>16) for info in z.infolist())
   z.extractall(tmp)
  extracted=Path(tmp)/'instruments'
  for model in MODELS:
   package=extracted/model/'model';manifest=json.loads((package/'manifest.json').read_text())
   for name,record in manifest['files'].items():
    assert hashlib.sha256((package/name).read_bytes()).hexdigest()==record['sha256'],(model,name)
   robot=ET.parse(package/'urdf/robot.urdf').getroot()
   for mesh in robot.findall('.//mesh'):assert (package/'urdf'/mesh.get('filename')).is_file()
   for path in (package/'urdf').rglob('*.obj'):
    for line in path.read_text().splitlines():
     if line.startswith('mtllib '):assert (path.parent/line.split(' ',1)[1]).is_file()
   for path in (package/'urdf').rglob('*.mtl'):
    for line in path.read_text().splitlines():
     if line.startswith('map_Kd '):assert (path.parent/line.split(' ',1)[1]).is_file()
   for page in [extracted/model/'index.html',package/'index.html']:
    for link in re.findall(r'(?:href|src)="([^"]+)"',page.read_text()):assert (page.parent/link).exists(),link
   print(model,'extracted archive: hashes, URDF assets, materials and links verified',flush=True)
  for link in re.findall(r'(?:href|src)="([^"]+)"',(extracted/'index.html').read_text()):assert (extracted/link).exists(),link
 pending.replace(target)
 print(str(target),f'{target.stat().st_size/1024**2:.1f} MiB',flush=True)


def project():
 """Share the authored pipeline and its data; no environments or generated meshes."""
 folder=ROOT/'output/instruments';folder.mkdir(parents=True,exist_ok=True)
 target=folder/'instruments_project.zip';pending=folder/'.instruments_project.zip.tmp'
 files={}
 for relative in ('src/instruments','docs/instruments','tests/instruments'):
  for path in sorted((SOURCE_ROOT/relative).rglob('*')):
   if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
    files[str(path.relative_to(SOURCE_ROOT))]=path
 for relative in ('src/helpers.py','pytest.ini','LICENSE','env/instruments-requirements.txt'):
  files[relative]=SOURCE_ROOT/relative
 for path in sorted((ROOT/'dataset/instruments').rglob('*')):
  if path.is_file():files[str(path.relative_to(ROOT))]=path
 for model in MODELS:
  parameters=ROOT/'output/instruments'/model/'model/parameters.json'
  if parameters.exists():files[str(parameters.relative_to(ROOT))]=parameters
 if not any(name.startswith('dataset/instruments/') for name in files):
  raise FileNotFoundError('Instrument datasets are required for the project archive')
 hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
 readme='''# Instrument modeling project

This archive contains the Python pipeline, all four machine recipes, editable
inputs, current parameter snapshots, documentation, tests, and instrument datasets
(photos/videos/references). It includes no installed environment, pretrained weights
or generated model meshes.
For ready-to-view GLB/URDF/OpenSCAD models use the separate instruments_share.zip.

Start with docs/instruments/README.md for the flowchart, setup and limitations.
Run commands from this extracted directory. Linux/Python 3.11 is the tested setup:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r env/instruments-requirements.txt
.venv/bin/python -m pytest tests/instruments -q
.venv/bin/python -m src.instruments build
.venv/bin/python -m src.instruments validate
```

For native CAD engine checks, install OpenSCAD or set OPENSCAD to its executable,
then add --check-cad to build. The ordinary build still exports native SCAD source.
The data root defaults to this directory; unset INSTRUMENT_WORKSPACE unless you
deliberately want an external dataset/output directory.

Edit evidence.json to use new photographs; review shapes/measurements before
editing a machine recipe. Additional photos are not automatically fitted to CAD.
Bundled parameters take precedence over defaults; no prior meshes are required.
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
