"""Instrument decals: reusable implementation behind the instruments CLI."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
import numpy as np,cv2,trimesh
from PIL import Image,ImageOps
from .paths import ROOT
def apply(model,out):
 from .inputs import evidence
 config=evidence(model)['markings'];size=config['review_size'];source=ROOT/config['source'];specs=config['items']
 im=ImageOps.exif_transpose(Image.open(source)).convert('RGB');im=im.resize(size);folder=out/'textures/decals';folder.mkdir(exist_ok=True)
 decals={};records=[]
 for spec in specs:
  label=spec['name'];corners=spec['corners_tl_tr_br_bl'];bounds=spec['target_xxz_z_m'];y=spec['target_y_m']
  from .design import parameters,decal_target
  values,_=parameters(json.loads((out/'parameters.json').read_text()))
  bounds,y=decal_target(model,label,bounds,y,values)
  width,height=600,240;matrix=cv2.getPerspectiveTransform(np.float32(corners),np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]]));patch=Image.fromarray(cv2.warpPerspective(np.array(im),matrix,(width,height)));patch.save(folder/(label+'.png'))
  x0,x1,z0,z1=bounds;y-=.00015
  q=trimesh.Trimesh(vertices=[[x0,y,z0],[x1,y,z0],[x1,y,z1],[x0,y,z1]],faces=[[0,1,2],[0,2,3]],process=False)
  q.visual=trimesh.visual.TextureVisuals(uv=[[0,0],[1,0],[1,1],[0,1]],material=trimesh.visual.material.PBRMaterial(name=label,baseColorTexture=patch,metallicFactor=0,roughnessFactor=.9,doubleSided=True))
  decals['photo_'+label]=q;records.append({'name':label,'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'review_size':size,'corners_tl_tr_br_bl':corners,'target_xxz_z_m':bounds,'target_y_m':y,'placement_status':'Photo-estimated planar placement, not calibrated camera registration'})
 for path in (out/'meshes').glob('visual_*.glb'):
  scene=trimesh.load(path,force='scene')
  for name,q in decals.items():
   if name in scene.geometry:scene.delete_geometry(name)
   scene.add_geometry(q,geom_name=name,node_name=name)
  scene.export(path)
 path=out/'urdf/robot.urdf';tree=ET.parse(path);robot=tree.getroot();children={j.find('child').get('link') for j in robot.findall('joint')};root=next(link for link in robot.findall('link') if link.get('name') not in children)
 for name,q in decals.items():
  for old in list(root.findall('visual')):
   if old.get('name')==name:root.remove(old)
  target=out/'urdf/visual'/name;target.mkdir(exist_ok=True);q=q.copy();q.visual.material=q.visual.material.to_simple();obj,assets=trimesh.exchange.obj.export_obj(q,return_texture=True,mtl_name='surface.mtl')
  (target/'mesh.obj').write_text(obj)
  for filename,data in assets.items():(target/filename).write_bytes(data if isinstance(data,bytes) else data.encode())
  visual=ET.SubElement(root,'visual',name=name);geometry=ET.SubElement(visual,'geometry');ET.SubElement(geometry,'mesh',filename=f'visual/{name}/mesh.obj')
 ET.indent(tree);tree.write(path,encoding='utf-8',xml_declaration=True)
 (folder/'provenance.json').write_text(json.dumps({'method':'Perspective-rectified real photo markings placed on corresponding panels. Visual-only quads; collision geometry unchanged.','decals':records},indent=2)+'\n')
