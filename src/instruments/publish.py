"""Instrument publish: reusable implementation behind the instruments CLI."""
from pathlib import Path
import hashlib,json,shutil
from PIL import Image,ImageDraw,ImageFont
from .paths import ROOT, MODELS
from ..output_layout import METHODS
VERSION='2026-11-02_v04'
DOCS={
 'c16361':{'url':'https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/sys/SHSS0025E_C16361.pdf','finding':'Brochure page 5: body W450 x D500 mm, body H230.8 mm; overall H253.3 for A16362-02 or H298.3 for A16362-01. User confirms overall height 253.3 mm, matching the low box; detector suffix alone does not identify accessories.'},
 'cary60':{'url':'https://www.chem-agilent.com/pdf/low_5990-7789JAJP.pdf','product_page':'https://www.chem-agilent.com/contents.php?id=1001878','finding':'Product brochure: exterior and accessory references, not an engineering service manual. Does not establish the measured 230 mm cover travel or this installed holder dimensions.'}}

def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def evidence(model,out):
 from .inputs import evidence as authoring_evidence
 config=authoring_evidence(model)
 (out/'evidence').mkdir(parents=True,exist_ok=True)
 for name,source in config.get('package_references',{}).items():
  if Path(name).name!=name:raise ValueError('Package reference names must be filenames')
  p=ROOT/source;target=out/'evidence'/name
  if not target.exists() or not p.samefile(target):shutil.copy2(p,target)
 records=[]
 dataset=ROOT/'dataset/hkqai-instruments'/model
 for p in sorted(dataset.rglob('*')):
  if not p.is_file() or p.name=='source_inventory.json':continue
  parts=p.relative_to(dataset).parts
  role={'user_captures':'user-captured photograph, video, annotation or measured depth','online_sources':'verified manufacturer/web source or derived annotation','unverified_sources':'unverified origin; do not attribute to user captures or online sources'}.get(parts[0],'source metadata')
  records.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'role':role})
 from .paths import SOURCE_ROOT
 authoring_inputs=[]
 for p in sorted((SOURCE_ROOT/'src/instruments/inputs'/model).glob('*.json')):
  authoring_inputs.append({'path':str(p.relative_to(SOURCE_ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 pdf_index=dataset/'online_sources/index.json'
 save(out/'evidence/source_inventory.json',{'scope':'All files under the authoritative instrument dataset. This output evidence folder is a generated review snapshot; sources lists the complete dataset. Inclusion does not imply geometric registration or use by the model.','sources':records,'authoring_inputs':authoring_inputs,'used_evidence':config,'pdf_extraction_index':str(pdf_index.relative_to(ROOT)) if pdf_index.exists() else None,'manufacturer_document_review':DOCS.get(model,{}),'priority':config.get('geometry_policy',{}).get('primary','Manufacturer specification drawings and dimension tables for the matching variant first; record and resolve conflicting observations explicitly.'),'geometry_policy':config.get('geometry_policy',{}),'specification_sources':config.get('specification_sources',[])})

def publish(model,source=None,extras=False):
 """Publish standalone native CAD and the complete current evidence dataset."""
 from html import escape
 from .layout import normalize, flatten_reviews
 from .storage.policy import apply
 source=Path(source) if source else ROOT/'output/hkqai-instruments'/model
 dest=ROOT/'output/hkqai-instruments'/model
 stage=ROOT/'work/hkqai-instruments'/model/'codex/model_staging'
 stage.parent.mkdir(parents=True,exist_ok=True)
 if stage.exists():raise RuntimeError(f'Unfinished staging directory: {stage}')
 # Normalize in staging, so a failed publication leaves current products intact.
 shutil.copytree(source,stage,ignore=shutil.ignore_patterns('*.zip','manifest.json','__pycache__','*.pyc','evidence',*METHODS),symlinks=False)
 normalize(stage,model)
 (stage/'reports').mkdir(exist_ok=True)
 apply(stage,extras=extras)
 flatten_reviews(stage)
 keep={'parameters.json','openscad/assembly.scad','openscad/common.scad',
       'openscad/parameters_native.scad','assembly.png','overview.png',
       'reports/cad_summary.json','reports/subtractive_solids.json','reports/motion.json'}
 keep.update(str(p.relative_to(stage)) for p in (stage/'printing').rglob('*') if p.is_file())
 if extras:
  keep.update(str(p.relative_to(stage)) for folder in ('previews','dimensions')
              for p in (stage/folder).glob('*.png'))
 for p in sorted(stage.rglob('*'),reverse=True):
  if p.is_file() and str(p.relative_to(stage)) not in keep:p.unlink()
  elif p.is_dir() and not any(p.iterdir()):p.rmdir()
 dataset=ROOT/'dataset/hkqai-instruments'/model
 shutil.copytree(dataset,stage/'evidence',symlinks=False)
 params=json.loads((stage/'parameters.json').read_text())
 limitations=params.get('limitations',[])
 notes='\n'.join('- '+str(x) for x in limitations)
 motion=('Set pose="custom" and change each named joint slider (0 closed, 1 open). '
         'QY doors, handles and selector are independent; opening is unused. '
         if model=='c13534' else 'Set pose="custom" and change opening (0 closed, 1 open). ')
 readme=f"""{model.upper()} — OpenSCAD model and complete evidence

Open openscad/assembly.scad in OpenSCAD. Keep all three SCAD files together.
Edit dimensions in openscad/parameters_native.scad (millimetres/degrees).
Use the Customizer or edit the motion assignments at the top of assembly.scad.
Choose pose=closed/open. {motion}
For animation: enable animate_motion, select View > Animate, set FPS=20 and
Steps=100. $t runs 0 to 1; the model opens and closes once per cycle.
QY also provides handles_released, upper_open and lower_open poses.
Animation demonstrates transforms only: no forces, contact solving or interlocks.
Motion limits and unmeasured clearances remain estimates.
CAD and review images use the same native geometry and solid colours.
Use simplified=true to omit cosmetic rounding and small hardware. Functional
openings remain. Edit assembly.scad modules for topology and parameters_native.scad
for dimensions; source in src/instruments/native is authoritative for rebuilding.
No external libraries, Python, URDF or imported meshes are needed to open the CAD.

Evidence is a full snapshot of the current dataset, including original photos,
annotations, videos, PDFs and retained PDF-extracted images, with no resizing.
user_captures contains your photographs; online_sources contains downloaded sources.
Manufacturer specification diagrams take priority for known dimensions. See
measurements.json and evidence.json for the source selection and uncertainty.
Source metadata paths beginning dataset/hkqai-instruments/{model}/ resolve under evidence/.
Other repository paths describe the authoring pipeline; they are not CAD dependencies.
parameters.json is an evaluated export snapshot, not an authoring input.
The manifest hashes every bundled file and records its original dataset path.
reports/ holds numerical validation summaries; intermediate test meshes are omitted.
Open index.html for the preview, dimensions and evidence entry points.

Known limitations:
{notes}
"""
 (stage/'README.txt').write_text(readme)
 page='<!doctype html><meta charset="utf-8"><title>'+model+' OpenSCAD</title><style>body{font:17px/1.5 system-ui;max-width:1100px;margin:30px auto}img{max-width:100%}</style>'
 page+=f'<h1>{model.upper()} — OpenSCAD model</h1><p>Native editable CAD with movable parts. Units: mm and degrees. See README for animation and limitations.</p>'
 links=['README.txt','openscad/assembly.scad','openscad/parameters_native.scad','parameters.json','manifest.json','overview.png']
 links += [str(p.relative_to(stage)) for p in sorted((stage/'reports').glob('*.json'))]
 page+='<p>'+' · '.join(f'<a href="{x}">{escape(x)}</a>' for x in links)+'</p><img src="assembly.png">'
 page+='<h2>Complete evidence</h2><p>Original resolution; source diagrams take priority. Your captures and online downloads remain separate.</p><ul>'
 for name in ('user_captures','online_sources','measurements.json','states.json','evidence.json'):
  if (stage/'evidence'/name).exists():page+=f'<li><a href="evidence/{name}">{name}</a></li>'
 page+='</ul>'
 (stage/'index.html').write_text(page)
 files={}
 for p in sorted(stage.rglob('*')):
  if not p.is_file():continue
  name=str(p.relative_to(stage));record={'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  if name.startswith('evidence/'):record['dataset_source']=f'dataset/hkqai-instruments/{model}/'+name[len('evidence/'):]
  files[name]=record
 save(stage/'manifest.json',{'schema_version':4,'format':'openscad','instrument':model,
      'units':{'length':'millimetre','angle':'degree'},'physical_contact_accuracy_validated':False,
      'evidence_scope':'Complete current instrument dataset, copied without resizing; symlinks dereferenced.',
      'files':files})
 from .validation import check_cad_package
 check_cad_package(stage)
 # The registry now sits beside the package; preserve it during republishing.
 if (dest/'asset.json').is_file():shutil.copy2(dest/'asset.json',stage/'asset.json')
 for method in METHODS-{'codex'}:
  if (dest/method).is_dir():(dest/method).rename(stage/method)
 if dest.exists():shutil.rmtree(dest)
 stage.rename(dest)
 save(dest/'current_model.json',{'package':'.','format':'openscad','publisher':'src/instruments/publish.py'})
 print(model,'published OpenSCAD and full evidence',len(files),'files',flush=True)
 return dest

def overview():
 board=Image.new('RGB',(1600,1300),'white');d=ImageDraw.Draw(board);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25)
 for i,m in enumerate(MODELS):
  p=ROOT/'output/hkqai-instruments'/m/'assembly.png'
  if not p.exists():continue
  im=Image.open(p).convert('RGB');im.thumbnail((780,580));x=(i%2)*800;y=(i//2)*650;d.text((x+20,y+10),m.upper()+' / current authored model',font=font,fill='black');board.paste(im,(x+10,y+50))
 (ROOT/'output/comparisons/hkqai-instruments').mkdir(parents=True,exist_ok=True)
 board.save(ROOT/'output/comparisons/hkqai-instruments/current_review.png')
 from ..output_layout import navigation
 navigation(ROOT/'output')
