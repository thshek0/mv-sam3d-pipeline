"""Instrument textures: reusable implementation behind the instruments CLI."""
from pathlib import Path
import json,hashlib
import numpy as np
import trimesh
from PIL import Image,ImageDraw,ImageFont
import open3d as o3d

def prepare(model,out):
 dest=out/'textures';dest.mkdir(exist_ok=True);images={};records={}
 from .inputs import evidence
 config=evidence(model)['materials']
 board=Image.new('RGB',(900,220*len(config)+70),'white');draw=ImageDraw.Draw(board)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
 draw.text((15,15),model.upper()+' / real photo patches (material sampling)',font=font,fill='black')
 for i,(kind,spec) in enumerate(config.items()):
  source=out/'evidence'/spec['review_image'];roi=spec['crop_pixels']
  image=Image.open(source).convert('RGB')
  if roi[2]>image.width or roi[3]>image.height:raise ValueError(f'{source}: crop exceeds image size')
  patch=image.crop(roi);patch.save(dest/(kind+'.png'));images[kind]=patch
  records[kind]={'source':str(source),'original_photo':spec['original'],'crop_pixels_in_exif_oriented_review_jpeg':roi,'jpeg_size':list(image.size),'patch_sha256':hashlib.sha256((dest/(kind+'.png')).read_bytes()).hexdigest()}
  thumb=image.copy();thumb.thumbnail((280,195));board.paste(thumb,(10,65+i*220))
  example=patch.copy();example.thumbnail((280,195));board.paste(example,(310,65+i*220));draw.text((610,80+i*220),kind+'\n'+spec['review_image']+'\n'+str(roi),font=font,fill='black')
 board.save(dest/'sources.png')
 return images,records

def choose(name,mesh,available):
 if name.startswith(('upper_dish_','upper_post_')) and 'silver' in available:return 'silver'
 color=np.asarray(mesh.visual.to_color().vertex_colors if mesh.visual.kind=='texture' else mesh.visual.vertex_colors)[:,:3].mean(0)
 if name=='lower_guide_0' and 'thread' in available:return 'thread'
 if any(k in name for k in ('stationary','fixed_strip','tower_rear_wall','tower_middle_band','control_top','control_bottom','control_left','control_right')) and 'white' in available:return 'white'
 if color.min()>205:return 'white' if 'white' in available else None
 if color.mean()>125 and not any(k in name for k in ('housing','wall','cover','lid','panel','strip','band','base_support')):return 'metal'
 if color.mean()<40 and 'black' in available:return 'black'
 return 'dark'

def textured(mesh,patch,name):
 # Duplicate vertices at seams; collision topology and all triangle positions stay intact.
 points=np.asarray(mesh.vertices)[mesh.faces];vertices=points.reshape(-1,3);faces=np.arange(len(vertices)).reshape(-1,3)
 uv=np.empty((len(mesh.faces),3,2),float)
 # Planar per-face material mapping; not camera registration.
 for i,normal in enumerate(mesh.face_normals):
  axis=int(np.argmax(abs(normal)));axes=[a for a in range(3) if a!=axis]
  uv[i]=points[i][:,axes]/1.5+.30
 material=trimesh.visual.material.PBRMaterial(name='photo_'+name,baseColorTexture=patch,baseColorFactor=[255,255,255,255],metallicFactor=.8 if name=='silver' else 0.0,roughnessFactor=.25 if name=='silver' else .85)
 q=trimesh.Trimesh(vertices=vertices,faces=faces,process=False);q.visual=trimesh.visual.TextureVisuals(uv=uv.reshape(-1,2),material=material)
 return q

def apply(model,out):
 patches,records=prepare(model,out);coverage={};results={}
 for source in sorted((out/'meshes').glob('collision_*.glb')):
  scene=trimesh.load(source,force='scene');before=scene.bounds.copy()
  for name,mesh in list(scene.geometry.items()):
   kind=choose(name,mesh,patches);coverage[name]=kind or 'untextured_no_corresponding_photo_patch'
   if kind:
    q=textured(mesh,patches[kind],kind)
    np.testing.assert_allclose(q.triangles,mesh.triangles,atol=1e-12)
    scene.geometry[name]=q
  scene.metadata.update(texture_method='Real photo material patches; not calibrated reprojection',texture_uv_scale='Model-scale planar projection; not calibrated texel density',collision_geometry='Use collision_*.glb / URDF collision meshes')
  target=source.with_name(source.name.replace('collision_','visual_',1));scene.export(target)
  restored=trimesh.load(target,force='scene');np.testing.assert_allclose(restored.bounds,before,atol=1e-6)
  textured_count=sum(q.visual.kind=='texture' and getattr(q.visual.material,'baseColorTexture',None) is not None for q in restored.geometry.values())
  assert textured_count>0,target
  results[target.name]={'embedded_texture_meshes':textured_count,'mesh_count':len(restored.geometry),'bounds_unchanged':True}
 report={'method':'Actual photo crops mapped as repeatable material patches, not whole-machine calibrated reprojection.',
  'limitations':['Texture patches retain photographic lighting/reflections; no de-lighting performed.','UV scale is approximate and avoids material tiling across the body; additional logo/text placement is recorded separately in textures/decals/provenance.json; fine surface wear is not reconstructed.','Unknown white surfaces without corresponding photos remain flat material.','Collision files intentionally keep geometry-only materials.'],
  'patches':records,'coverage':coverage,'exports':results}
 (out/'textures/provenance.json').write_text(json.dumps(report,indent=2)+'\n')
 return report

class TextureRenderer:
 def __init__(self,parts):
  vertices=[];faces=[];uv=[];material_ids=[];colors=[];self.images=[];offset=0
  for name,mesh in parts.items():
   vertices.append(mesh.vertices);faces.append(mesh.faces+offset);offset+=len(mesh.vertices)
   image=getattr(getattr(mesh.visual,'material',None),'baseColorTexture',None)
   if image is not None and getattr(mesh.visual,'uv',None) is not None:
    matid=len(self.images);self.images.append(np.array(image.convert('RGB')));uv.extend(mesh.visual.uv[mesh.faces]);material_ids.extend([matid]*len(mesh.faces));colors.extend([[255,255,255]]*len(mesh.faces))
   else:
    uv.extend(np.zeros((len(mesh.faces),3,2)));material_ids.extend([-1]*len(mesh.faces));colors.extend(mesh.visual.to_color().face_colors[:,:3] if mesh.visual.kind=='texture' else mesh.visual.face_colors[:,:3])
  self.uv=np.asarray(uv);self.mats=np.asarray(material_ids);self.colors=np.asarray(colors)
  self.scene=o3d.t.geometry.RaycastingScene(nthreads=4)
  self.scene.add_triangles(o3d.core.Tensor(np.vstack(vertices).astype(np.float32)),o3d.core.Tensor(np.vstack(faces).astype(np.uint32)))
 def image(self,u,v,camera,extent,width=1000):
  u,v,camera=map(np.asarray,(u,v,camera));xmin,xmax,ymin,ymax=extent;height=round(width*(ymax-ymin)/(xmax-xmin))
  x,y=np.meshgrid(np.linspace(xmin,xmax,width,endpoint=False)+(xmax-xmin)/(2*width),np.linspace(ymax,ymin,height,endpoint=False)-(ymax-ymin)/(2*height));starts=x[...,None]/1000*u+y[...,None]/1000*v+camera*3
  rays=np.concatenate([starts,np.broadcast_to(-camera,starts.shape)],-1).astype(np.float32);hit=self.scene.cast_rays(o3d.core.Tensor(rays),nthreads=4)
  valid=np.isfinite(hit['t_hit'].numpy());tri=hit['primitive_ids'].numpy()[valid].astype(int);bary=hit['primitive_uvs'].numpy()[valid]
  uv=self.uv[tri,0]*(1-bary[:,0,None]-bary[:,1,None])+self.uv[tri,1]*bary[:,0,None]+self.uv[tri,2]*bary[:,1,None]
  rgb=self.colors[tri].copy()
  for i,tex in enumerate(self.images):
   mask=self.mats[tri]==i;coord=uv[mask]%1;xx=np.floor(coord[:,0]*tex.shape[1]).astype(int)%tex.shape[1];yy=np.floor((1-coord[:,1])*tex.shape[0]).astype(int)%tex.shape[0];rgb[mask]=tex[yy,xx]
  normals=hit['primitive_normals'].numpy()[valid];light=camera+u*.35+v*.65;light/=np.linalg.norm(light);shade=.65+.35*np.clip(normals@light,0,1)
  rgba=np.zeros((height,width,4),np.uint8);rgba[valid,:3]=np.clip(rgb*shade[:,None],0,255);rgba[valid,3]=255
  return rgba

def urdf_visuals(model,out,coverage):
 import xml.etree.ElementTree as ET
 path=out/'urdf/robot.urdf';tree=ET.parse(path);count=0
 for link in tree.getroot().findall('link'):
  for visual in link.findall('visual'):
   name=visual.get('name');kind=coverage.get(name)
   if not kind or kind.startswith('untextured'):continue
   # Always rebuild visuals from the authoritative collision OBJ, avoiding repeat-UV duplication.
   source=out/'urdf/meshes'/f'{name}.obj'
   if not source.exists():continue
   mesh=trimesh.load(source,force='mesh',process=False);patch=Image.open(out/'textures'/f'{kind}.png')
   q=textured(mesh,patch,kind);q.visual.material=q.visual.material.to_simple()
   text,assets=trimesh.exchange.obj.export_obj(q,include_texture=True,include_color=False,return_texture=True,mtl_name='surface.mtl')
   folder=out/'urdf/visual'/name;folder.mkdir(parents=True,exist_ok=True)
   (folder/'mesh.obj').write_text(text)
   for filename,data in assets.items():
    target=folder/filename;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data if isinstance(data,bytes) else data.encode())
   visual.find('geometry/mesh').set('filename',f'visual/{name}/mesh.obj')
   material=visual.find('material')
   if material is not None:visual.remove(material)
   count+=1
 ET.indent(tree);tree.write(path,encoding='utf-8',xml_declaration=True)
 manifest=out/'urdf/manifest.json'
 if manifest.exists():
  data=json.loads(manifest.read_text())
  for name,info in data.get('parts',{}).items():
   if (out/'urdf/visual'/name/'mesh.obj').exists():info['visual_mesh']=f'visual/{name}/mesh.obj'
  data['visual_texture_method']='Real photo patches, OBJ+MTL+PNG; collision OBJ files remain unchanged';manifest.write_text(json.dumps(data,indent=2)+'\n')
 return count

def render_review(model,out):
 import matplotlib.pyplot as plt
 def world(scene):
  parts={}
  for name in scene.graph.nodes_geometry:
   t,g=scene.graph[name];q=scene.geometry[g].copy();q.apply_transform(t);parts[name]=q
  return parts
 states=['closed','open'] if model!='c13534' else ['closed','upper_open','lower_open','open']
 parts={s:world(trimesh.load(out/'meshes'/f'visual_{s}.glb',force='scene')) for s in states}
 cam=np.array([-.7,-1.6,1.15] if model=='c13472' else [.7,-1.6,1.15]);cam/=np.linalg.norm(cam);u=np.cross([0,0,1],cam);u/=np.linalg.norm(u);v=np.cross(cam,u)
 verts=np.vstack([q.vertices for group in parts.values() for q in group.values()])*1000;xx,yy=verts@u,verts@v;extent=[xx.min()-20,xx.max()+20,yy.min()-20,yy.max()+20]
 fig,axes=plt.subplots(1 if len(states)==2 else 2,2,figsize=(14,7 if len(states)==2 else 11),squeeze=False)
 for ax,state in zip(axes.flat,states):ax.imshow(TextureRenderer(parts[state]).image(u,v,cam,extent,width=900));ax.axis('off');ax.set_title(state.replace('_',' '))
 fig.suptitle(model.upper()+' / real photo material textures',fontsize=18);fig.text(.03,.025,'Actual exported visual meshes. Photo-derived materials; placement/lighting are not a calibrated texture reconstruction.',fontsize=10)
 fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(out/'preview_textured.png',dpi=140);plt.close(fig)
 original=out/'preview.png';backup=out/'preview_collision.png'
 if original.exists() and not backup.exists():backup.write_bytes(original.read_bytes())
 original.write_bytes((out/'preview_textured.png').read_bytes())
 selected={}
 info=json.loads((out/'parts.json').read_text())
 for n,q in parts['open'].items():
  keep={'cary60':n.startswith(('holder_','socket_','inner_')),'c16361':n.startswith(('perforated_floor_','floor_fastener_','front_latch_','rear_adjuster_')),'c13472':n.startswith(('holder_','stage_','side_','micrometer_')),'c13534':info.get(n,{}).get('group')=='lower_interior' and not any(s in n for s in ('lining','infill'))}[model]
  if keep:selected[n]=q
 cam=np.array([.35,-1.6,.45] if model=='c13534' else ([-.6,-1.1,1.5] if model=='c13472' else [.3,-1.1,1.5]));cam/=np.linalg.norm(cam);u=np.cross([0,0,1],cam);u/=np.linalg.norm(u);v=np.cross(cam,u)
 verts=np.vstack([q.vertices for q in selected.values()])*1000;xx,yy=verts@u,verts@v;extent=[xx.min()-10,xx.max()+10,yy.min()-10,yy.max()+10]
 fig,ax=plt.subplots(figsize=(10,8));ax.imshow(TextureRenderer(selected).image(u,v,cam,extent,width=1200));ax.axis('off');ax.set_title(model.upper()+' / detailed interior with photo material patches')
 fig.text(.04,.025,'Unmeasured positions / sizes remain estimates; collision openings are real geometry.',fontsize=10);fig.savefig(out/'interior_textured.png',dpi=150);plt.close(fig)
 original=out/'interior.png';backup=out/'interior_collision.png'
 if original.exists() and not backup.exists():backup.write_bytes(original.read_bytes())
 original.write_bytes((out/'interior_textured.png').read_bytes())
 if model=='c16361':render_front_reference(model,out,parts['closed'])
 return len(selected)


def render_front_reference(model,out,parts):
 """Render a front view beside the supplied PDF instrument crop."""
 from .paths import ROOT as root
 image=Image.fromarray(TextureRenderer(parts).image([1,0,0],[0,0,1],[0,-1,0],[-10,460,-10,275],1000))
 background=Image.new('RGBA',image.size,'white');background.alpha_composite(image)
 current=background.convert('RGB');current.save(out/'front_textured.png')
 from .inputs import evidence
 source=root/evidence(model)['front_reference']
 reference=Image.open(source).convert('RGB');w,h=reference.size
 reference=reference.crop((int(w*.06),int(h*.39),int(w*.45),int(h*.60)))
 reference.thumbnail((850,650));current.thumbnail((850,650))
 board=Image.new('RGB',(1800,800),'white');draw=ImageDraw.Draw(board)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',26)
 params=json.loads((out/'parameters.json').read_text());height=params.get('overall_height_m',.2533)*1000
 draw.text((20,20),'Manufacturer PDF reference (instrument crop)',font=font,fill='black')
 draw.text((920,20),f'Current front / overall height {height:g} mm',font=font,fill='black')
 board.paste(reference,(20,90));board.paste(current,(920,90));board.save(out/'front_reference_comparison.png')
