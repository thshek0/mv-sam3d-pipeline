from pathlib import Path
import sys,os,json,numpy as np,trimesh
from PIL import Image,ImageDraw
os.environ['OMP_NUM_THREADS']='4'
sys.path.insert(0,str(Path(__file__).parent));from inspection_helpers import ROOT,ray_scene
import open3d as o3d
for name in sys.argv[1:]:
 out=ROOT/'output/objects/hestia-rl'/name;path=out/'codex/products'/f'{name}_m.glb'
 source=trimesh.load(path,force='scene');assert len(source.graph.nodes_geometry)==1
 transform,key=source.graph[next(iter(source.graph.nodes_geometry))];m=source.geometry[key].copy();m.apply_transform(transform)
 loaded_normals=m.vertex_normals.copy();m.apply_scale(1000);m.vertex_normals=loaded_normals;scene=ray_scene(m);tint=np.array(json.loads((out/'codex/products/validation.json').read_text())['appearance_check']['rgba'][:3])
 board=Image.new('RGB',(1600,450),'white');draw=ImageDraw.Draw(board)
 for j,(title,direction) in enumerate([('Top',[0,0,1]),('Oblique',[.8,-1,1]),('Profile',[0,-1,.12]),('Underside',[.4,-.6,-1])]):
  view=np.array(direction,dtype=float);view/=np.linalg.norm(view);right=np.cross([0,1,0] if abs(view[2])>.95 else [0,0,1],view);right/=np.linalg.norm(right);up=np.cross(view,right)
  size=400;span=max(m.extents)*1.15;u,v=np.meshgrid((np.arange(size)+.5-size/2)/size*span,(size/2-np.arange(size)-.5)/size*span)
  center=m.bounds.mean(0);orig=center+view*span*2+u[:,:,None]*right+v[:,:,None]*up;rays=np.concatenate([orig,np.broadcast_to(-view,orig.shape)],axis=2).astype('float32');cast=scene.cast_rays(o3d.core.Tensor(rays));dist=cast['t_hit'].numpy();hit=np.isfinite(dist);ids=cast['primitive_ids'].numpy();uv=cast['primitive_uvs'].numpy();normals=np.zeros((size,size,3));fid=ids[hit];weights=np.c_[1-uv[hit].sum(1),uv[hit]];normals[hit]=(m.vertex_normals[m.faces[fid]]*weights[:,:,None]).sum(1);normals/=np.maximum(np.linalg.norm(normals,axis=2,keepdims=True),1e-9)
  light=view+up*.5-right*.4;light/=np.linalg.norm(light);shade=.25+.75*np.maximum(normals@light,0);pixels=np.full((size,size,3),248,dtype=np.uint8);colours=np.broadcast_to(tint,(int(hit.sum()),3)).copy()
  material=getattr(m.visual,'material',None);texture=getattr(material,'baseColorTexture',None)
  if texture is not None and getattr(m.visual,'uv',None) is not None:
   tex=np.asarray(texture.convert('RGB'));coords=(m.visual.uv[m.faces[fid]]*weights[:,:,None]).sum(1);tx=np.clip(coords[:,0]*(tex.shape[1]-1),0,tex.shape[1]-1);ty=np.clip((1-coords[:,1])*(tex.shape[0]-1),0,tex.shape[0]-1);x0=np.floor(tx).astype(int);y0=np.floor(ty).astype(int);x1=np.minimum(x0+1,tex.shape[1]-1);y1=np.minimum(y0+1,tex.shape[0]-1);a=(tx-x0)[:,None];b=(ty-y0)[:,None]
   colours=(1-a)*(1-b)*tex[y0,x0]+a*(1-b)*tex[y0,x1]+(1-a)*b*tex[y1,x0]+a*b*tex[y1,x1]
  half_vector=light+view;half_vector/=np.linalg.norm(half_vector)
  roughness=getattr(material,'roughnessFactor',.4) or .4
  exponent=max(16,2/(roughness*roughness))
  highlight=45*np.maximum(normals[hit]@half_vector,0)**exponent
  pixels[hit]=np.clip(shade[hit,None]*colours+highlight[:,None],0,255).astype('uint8')
  board.paste(Image.fromarray(pixels),(j*400,40));draw.text((j*400+15,15),title,fill='black')
 board.save(out/'codex/products'/'review.png')
 print(name,'rendered',flush=True)

# Keep the dimension drawing synchronized with the exported profile.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for name in sys.argv[1:]:
 d=ROOT/'output/objects/hestia-rl'/name/'codex/products';a=json.loads((d/'validation.json').read_text());p=a['profile_mm'];r=np.array(p['radius']);t=np.array(p['top']);b=np.array(p['bottom']);x=np.r_[-r[:0:-1],r];top=np.r_[t[:0:-1],t];bottom=np.r_[b[:0:-1],b];diam=a['dimensions']['outer_diameter_mm'];height=a['dimensions']['overall_height_mm']
 fig,ax=plt.subplots(figsize=(10,4));ax.fill_between(x,bottom,top,color=np.array(a['appearance_check']['rgba'][:3])/255,edgecolor='#46505d',linewidth=.8)
 ax.annotate('',xy=(-diam/2,-7),xytext=(diam/2,-7),arrowprops={'arrowstyle':'<->'});ax.text(0,-8,f'Outer diameter {diam:.1f} mm',ha='center',va='top')
 ax.annotate('',xy=(diam/2+8,0),xytext=(diam/2+8,height),arrowprops={'arrowstyle':'<->'});ax.text(diam/2+10,height/2,f'{height:.1f} mm',rotation=90,va='center');ax.axhline(0,color='gray',linestyle=':',linewidth=.6)
 ax.set_aspect('equal');ax.set_xlim(-diam/2-15,diam/2+25);ax.set_ylim(-20,height+15);ax.set_title(name+' — '+a['dimensions'].get('status_label','Nominal model dimensions'));ax.set_xlabel('mm');ax.set_ylabel('mm');ax.spines[['right','top']].set_visible(False)
 fig.savefig(d/'dimensions.png',dpi=150,bbox_inches='tight');plt.close(fig)
