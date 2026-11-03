"""Instrument publish: reusable implementation behind the instruments CLI."""
from pathlib import Path
import hashlib,json,shutil,os,xml.etree.ElementTree as ET
from PIL import Image,ImageDraw,ImageFont
from .paths import ROOT, MODELS
VERSION='2026-11-02_v04'
DOCS={
 'c16361':{'url':'https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/sys/SHSS0025E_C16361.pdf','finding':'Brochure page 5: body W450 x D500 mm, body H230.8 mm; overall H253.3 for A16362-02 or H298.3 for A16362-01. User confirms overall height 253.3 mm, matching the low box; detector suffix alone does not identify accessories.'},
 'cary60':{'url':'https://www.chem-agilent.com/pdf/low_5990-7789JAJP.pdf','product_page':'https://www.chem-agilent.com/contents.php?id=1001878','finding':'Product brochure: exterior and accessory references, not an engineering service manual. Does not establish the measured 230 mm cover travel or this installed holder dimensions.'}}

def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def evidence(model,out):
 records=[]
 for p in sorted((ROOT/'dataset/instruments'/model).rglob('*')):
  if not p.is_file() or p.suffix.lower() not in ('.jpg','.jpeg','.png','.heic','.mov','.pdf'):continue
  records.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'role':'Available evidence; inclusion in this inventory alone does not imply geometric registration'})
 selected={'previous_exterior':'Earlier supplied exterior image: checked appearance and supplies body material crop.','new_photos':'November 2 interior geometry and hardware material crops; see parameters.json and textures/provenance.json.', 'prior_rgbd':'Historical views retained as evidence; no new DA3 alignment or metric fusion claimed.'}
 if model=='c13534':selected.update(previous_IMG_9838='Retained measured tower profile and tapered flush strip; checked against earlier top view.',previous_IMG_9855_MOV='Previously reviewed control motion retained in the seven-DOF authored model; not a new motion measurement.')
 save(out/'evidence/source_inventory.json',{'sources':records,'used_evidence':selected,'manufacturer_document_review':DOCS.get(model,{}),'priority':'User measurements and observed installed configuration; manufacturer dimensions for matching variant; photo estimates explicitly provisional.'})
 for p in (ROOT/'dataset/instruments'/model/'references/pdf/manufacturer_documents').glob('*.pdf'):
  shutil.copy2(p,out/'evidence'/p.name)
 board=Image.new('RGB',(1200,720),'white');draw=ImageDraw.Draw(board);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
 draw.text((20,15),model.upper()+' / previous and current evidence',font=font,fill='black')
 paths=[out/'evidence/previous_exterior.jpg']+sorted((out/'evidence').glob('IMG_*.jpg'))[:3]
 if model=='c13534':paths.insert(1,out/'evidence/previous_IMG_9838.jpg')
 for i,p in enumerate(paths[:6]):
  if not p.exists():continue
  im=Image.open(p).convert('RGB');im.thumbnail((380,280));x=(i%3)*400;y=(i//3)*320+60;board.paste(im,(x+10,y+30));draw.text((x+10,y),p.stem,font=font,fill='black')
 board.save(out/'evidence/history.png')

def publish(model,source=None,extras=False):
 source=Path(source) if source else ROOT/'output/instruments'/model/'model'
 version=json.loads((source/'parameters.json').read_text()).get('iteration',VERSION)
 assert (source/'meshes/visual_closed.glb').is_file(),source
 evidence(model,source)
 if model=='c13534' and (source/'dimensions/index.html').exists():
  page=source/'dimensions/index.html';text=page.read_text().replace('../urdf/index.html','../urdf/robot.urdf')
  for rel in ['sources/hamamatsu-c13534.pdf','references/qy_dimension_drawing.png']:
   original=ROOT/'dataset/instruments/c13534/references/pdf'/Path(rel).name
   shutil.copy2(original,source/'evidence'/original.name)
  page.write_text(text)
 dest=ROOT/'output/instruments'/model/'model';stage=dest.parent/'model_staging'
 if stage.exists():raise RuntimeError(f'Unfinished staging directory: {stage}')
 shutil.copytree(source,stage,ignore=shutil.ignore_patterns('*.zip','manifest.json','__pycache__','*.pyc'),symlinks=False)
 # Manifest exclusions above are only stale summaries; retain per-URDF provenance explicitly.
 if (source/'urdf/manifest.json').exists():shutil.copy2(source/'urdf/manifest.json',stage/'urdf/manifest.json')
 from .layout import normalize
 normalize(stage,model)
 from .storage.policy import apply
 apply(stage,extras=extras)
 loader=stage/'urdf/load_pybullet.py'
 if loader.exists():loader.write_text(loader.read_text().replace("PACKAGE/'c13534.urdf'","PACKAGE/'robot.urdf'"))
 params=json.loads((stage/'parameters.json').read_text())
 limitations=params.get('limitations',[])+['Photographed logos/text are placed as visual-only planar decals; coverage and placement remain incomplete and uncalibrated.','Internal joints, inertias and insertion/contact tolerances are not physically validated.','Motion GIF and dimension views may use untextured collision materials; GLBs and assembly/interior PNGs use photo materials.']
 if model=='cary60':limitations.append('Cover receiver clearance 0.5 mm is a numerical modeling assumption, not a measured gap.')
 save(stage/'reports/limitations.json',limitations)
 robot=ET.parse(stage/'urdf/robot.urdf').getroot()
 for mesh in robot.findall('.//mesh'):
  assert (stage/'urdf'/mesh.get('filename')).is_file(),mesh.attrib
 joints=[{'name':j.get('name'),'type':j.get('type'),'limit':j.find('limit').attrib if j.find('limit') is not None else {}} for j in robot.findall('joint') if j.get('type')!='fixed']
 page='<!doctype html><meta charset="utf-8"><title>'+model+' current model</title><style>body{font:17px/1.5 system-ui;max-width:1200px;margin:30px auto}img{max-width:100%}</style>'
 page+=f'<h1>{model.upper()} — current authored model</h1><p>Best available consolidated version, {version}. Metric prototype with measured and estimated details; physical contact accuracy remains unverified.</p>'
 page+='<p><a href="manifest.json">Manifest</a> · <a href="parameters.json">Dimensions/parameters</a> · <a href="reports/validation.json">Checks</a> · <a href="reports/limitations.json">Uncertainties</a> · <a href="urdf/robot.urdf">URDF</a></p>'
 page+='<p>'+ ' · '.join(f'<a href="meshes/visual_{s}.glb">{s.replace("_"," ")} textured GLB</a>' for s in ['closed','open','upper_open','lower_open'] if (stage/'meshes'/f'visual_{s}.glb').exists())+'</p>'
 if (stage/'openscad/parameters_native.scad').exists():
  page+='<p>Native parametric CSG: <a href="openscad/assembly.scad">OpenSCAD source</a> · <a href="openscad/parameters_native.scad">Editable dimensions</a> · <a href="openscad/README.txt">Scope and limitations</a></p>'
  if (stage/'openscad/csg_review.png').exists():page+='<p><a href="openscad/csg_review.png">CSG parameter-change review</a> · <a href="openscad/csg_validation.json">OpenSCAD checks</a></p>'
  if (stage/'openscad/csg_shared_parameters.png').exists():page+='<p>CAD, GLB and URDF share their part/joint definition. <a href="openscad/csg_shared_parameters.png">Synchronized parameter-change example</a>. See openscad/README.txt for importing dimension edits.</p>'
 page+='<img src="previews/assembly.png">'
 if (stage/'previews/interior.png').exists():page+='<h2>Interior</h2><img src="previews/interior.png">'
 page+='<p>Collision geometry uses separate convex pieces with real openings. Textures use real photo crops; they are not a calibrated photogrammetric skin.</p><h2>Dimension views</h2><p>'
 page+=' · '.join(f'<a href="dimensions/{p.name}">{p.stem}</a>' for p in sorted((stage/'dimensions').glob('*.png')))
 page+='</p><h2>Evidence across captures</h2><img src="evidence/history.png"><p><a href="evidence/source_inventory.json">Source inventory and manufacturer review</a> · <a href="textures/provenance.json">Texture provenance</a> · <a href="textures/decals/provenance.json">Photographed markings</a></p>'
 (stage/'index.html').write_text(page)
 (stage/'README.txt').write_text('CURRENT AUTHORITATIVE AUTHORED MODEL\nOpen index.html. Geometry and texture files are local; evidence/ links into dataset. Use the share command to create a self-contained ZIP. No SAM3D outputs are needed to load this package.\nUnits: metres, radians. X right, Y rear, Z up. Fixed base required.\nmeshes/visual_{state}.glb: textured closed/open visuals. Optional reviews: build --extras. Collision geometry always lives in urdf/meshes.\nURDF: urdf/robot.urdf; preserve separate convex collision pieces. QY: apply supplied self-contact filters using urdf/load_pybullet.py.\nRead reports/limitations.json before contact/insertion tasks. Masses, internal motions and contact tolerances remain provisional.\nEditable dimensions: parameters.json/design_parameters (mm/degrees). Shared part/joint recipes: src/instruments/<machine>.py. Rebuild through src.instruments; see openscad/README.txt for numeric SCAD parameter import. Reconstruction is not required.\n')
 files={str(p.relative_to(stage)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(stage.rglob('*')) if p.is_file() and p!=stage/'manifest.json'}
 from .layout import RELOCATIONS
 manifest={'package_file_relocations':RELOCATIONS,'schema_version':3,'instrument':model,'status':'current best available authored prototype','iteration':version,'units':{'length':'metre','angle':'radian'},'fixed_base':True,'frame':'X right, Y rear, Z up; support plane Z=0','physical_contact_accuracy_validated':False,'source_build':str((ROOT/'output/instruments'/model/'model').relative_to(ROOT)),'runtime_reconstruction_dependency':False,'joints':joints,'regenerate':f'.venv/bin/python -m src.instruments build --model {model}','files':files}
 save(stage/'manifest.json',manifest)
 assert not any(p.is_symlink() for p in stage.rglob('*'))
 if dest.exists():
  shutil.rmtree(dest)  # User policy: retain only the current package, after staging validation.
 stage.rename(dest)
 # Store source/review images in dataset; Sharing embeds portable copies.
 evidence_dir=ROOT/'dataset/instruments'/model/'references/model_evidence'
 shutil.copytree(dest/'evidence',evidence_dir,dirs_exist_ok=True)
 shutil.rmtree(dest/'evidence')
 (dest/'evidence').symlink_to(os.path.relpath(evidence_dir,dest),target_is_directory=True)
 save(dest.parent/'current_model.json',{'source_build':str((ROOT/'output/instruments'/model/'model').relative_to(ROOT)),'package':'model','publisher':'src/instruments/publish.py'})
 (dest.parent/'index.html').write_text(f'<!doctype html><meta http-equiv="refresh" content="0;url=model/index.html"><a href="model/index.html">{model} current model</a>')
 print(model,'published current model',len(files),'files',flush=True)
 return dest

def overview():
 board=Image.new('RGB',(1600,1300),'white');d=ImageDraw.Draw(board);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25)
 for i,m in enumerate(MODELS):
  p=ROOT/'output/instruments'/m/'model/previews/assembly.png'
  if not p.exists():continue
  im=Image.open(p).convert('RGB');im.thumbnail((780,580));x=(i%2)*800;y=(i//2)*650;d.text((x+20,y+10),m.upper()+' / current authored model',font=font,fill='black');board.paste(im,(x+10,y+50))
 board.save(ROOT/'output/instruments/instruments_review.png')
 (ROOT/'output/instruments/index.html').write_text('<!doctype html><meta charset="utf-8"><title>Current instrument models</title><style>body{font:18px system-ui;max-width:1300px;margin:30px auto}img{max-width:100%}</style><h1>Current instrument models</h1><p>One authoritative model/ package per machine. Only the current model package is retained; source evidence lives in dataset/. Best available metric prototypes; unmeasured interiors remain estimates.</p><p>'+' · '.join(f'<a href="{m}/model/index.html">{m.upper()}</a>' for m in MODELS)+'</p><img src="instruments_review.png">')
