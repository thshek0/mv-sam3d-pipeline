"""Instrument construction: reusable implementation behind the instruments CLI."""
from pathlib import Path
import hashlib, json, shutil
import numpy as np
import trimesh
from PIL import Image
from . import geometry as base
from .render import Renderer

from .paths import ROOT

def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def outdir(model):return ROOT/'output/objects/hkqai-instruments'/model/'codex/intermediate/.build_work'


def export_scad(out):
 """Export the current native CAD source; obsolete mesh dumps are not retained."""
 from .cad import export
 return export(out)


def extras(model,out,p,review,parts):
 import matplotlib.pyplot as plt
 paths=sorted((out/'evidence').glob('*.jpg'));selected=paths[-4:]
 fig,axes=plt.subplots(1,len(selected),figsize=(14,6),squeeze=False)
 for ax,path in zip(axes.flat,selected):ax.imshow(Image.open(path));ax.axis('off');ax.set_title(path.stem)
 fig.suptitle(model.upper()+' / new photo evidence (not registered)',fontsize=18);fig.tight_layout();fig.savefig(out/'sources.png',dpi=140);plt.close(fig)
 save(out/'parts.json',parts)
 from .inputs import evidence
 sources=dict.fromkeys(evidence(model)['geometry_sources'])
 for name in sources:sources[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
 save(out/'provenance.json',dict(status='Current authored model',iteration=p.get('iteration','current'),sources_sha256=sources,
  method='Shared parametric solids with explicit Boolean material removal; convex simulation pieces derived from the same geometry. Specification drawings and measurements guide geometry; photos are interpreted manually.',
  regenerate=f'.venv/bin/python -m src.instruments build --model {model}',changes=p['changes'],limitations=p['limitations'],
  photo_registration=False,physical_contact_accuracy_validated=False,openscad='Native CSG; see openscad/README.txt for parameter ownership and csg_validation.json for actual engine checks'))
 save(out/'review.json',review)


def interior_view(out,parts,model):
 import matplotlib.pyplot as plt
 cam=np.array([.35,-1.6,.45] if model=='c13534' else [.2,-.6,1.3]);cam/=np.linalg.norm(cam);u=np.cross([0,0,1],cam);u/=np.linalg.norm(u);v=np.cross(cam,u)
 verts=np.vstack([q.vertices for q in parts.values()])*1000;xx,yy=verts@u,verts@v
 extent=[xx.min()-10,xx.max()+10,yy.min()-10,yy.max()+10]
 fig,ax=plt.subplots(figsize=(9,7));ax.imshow(Renderer(parts).image(u,v,cam,extent,width=1000));ax.axis('off')
 ax.set_title(model.upper()+' / isolated interior pieces (housing hidden)');fig.text(.06,.03,'Local shapes/positions estimated unless explicitly listed as user measurements.',fontsize=10)
 fig.savefig(out/'interior.png',dpi=140);plt.close(fig)




def qy_world(m,state):
 transforms=m.transforms(state);parts={}
 for name,q in m.parts.items():
  parts[name]=q.copy();parts[name].apply_transform(transforms[m.info[name]['group']])
 return parts


def export_qy_model(out,params):
 """Shared QY geometry/URDF export and engine checks, also used by parameter tests."""
 import importlib.util
 from .qy import urdf as exporter
 from .design import build as shared_build
 from .c13534 import configurations
 model='c13534';(out/'meshes').mkdir(parents=True,exist_ok=True)
 m=shared_build(params)
 save(out/'parameters.json',params)
 original_config,door_config=configurations(params,m)
 p=original_config['parameters'];states=m.demo_states
 for name,state in states.items():
  m.scene(state).export(out/'meshes'/f'collision_{name}.glb')
 cfg=dict(original_config,parameters=p,links=m.links,parts=m.info,pose_snapshots=states,
  review_status='Current authored interior and exterior geometry',
  regenerate='.venv/bin/python -m src.instruments build --model c13534')
 save(out/'collision_controls.json',cfg)
 save(out/'collision_doors.json',door_config)
 from .paths import ROOT
 readings=json.loads((ROOT/'dataset/hkqai-instruments/c13534/measurements.json').read_text())['confirmed']
 save(out/'confirmed_measurements.json',readings)
 exporter.OUT=out;exporter.PACKAGE=out/'urdf';exporter.CONFIG=out/'collision_controls.json';exporter.SOURCE=out/'meshes/collision_closed.glb'
 manifest=exporter.generate();review=exporter.validate(manifest)
 # Independent simulator loader from this new package, including its self filters.
 from .physics import pybullet
 bullet=pybullet()
 spec=importlib.util.spec_from_file_location('candidate_loader',out/'urdf/load_pybullet.py');loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
 c=bullet.connect(bullet.DIRECT);body,links,joints=loader.load_model(bullet,c)
 poses=[{}]
 for joint_name,index in joints.items():
  info=bullet.getJointInfo(body,index,physicsClientId=c)
  prerequisites={}
  side='upper' if joint_name.startswith('upper') else 'lower'
  if 'handle_turn' in joint_name or 'door_joint' in joint_name:
   prerequisites[side+'_handle_pop_joint']=float(np.deg2rad(p['handle_pop_preview_degrees']))
  if 'door_joint' in joint_name:
   prerequisites[side+'_handle_turn_joint']=float(np.deg2rad(p['handle_turn_preview_degrees']))
  for value in np.linspace(info[8],info[9],11):poses.append({**prerequisites,joint_name:float(value)})
 poses+=list(json.loads((out/'urdf/joint_states.json').read_text())['poses'].values())
 worst=0.;failures=[]
 for pose in poses:
  for name,index in joints.items():bullet.resetJointState(body,index,pose.get(name,0),physicsClientId=c)
  bullet.performCollisionDetection(physicsClientId=c)
  for hit in bullet.getContactPoints(body,body,physicsClientId=c):
   worst=max(worst,-hit[8])
   if hit[8]<-.0005:failures.append(dict(pose=pose,penetration_mm=-hit[8]*1000))
 bullet.disconnect(c);assert not failures,failures[:3]
 # Check the exact measured geometry and unchanged approved exterior.
 nominal={}
 for name,target in [('lower_guide_0',params['lead_screw_diameter_m']),('lower_guide_1',params['right_guide_diameter_m']),('lower_small_pin',params['small_pin_diameter_m']),('lower_central_piece',params['central_piece_diameter_m'])]:
  spans=np.ptp(m.parts[name].bounds,axis=0);np.testing.assert_allclose(spans[:2],target,atol=1e-10);nominal[name+'_diameter_mm']=target*1000
 clear=m.parts['lower_right_infill'].bounds[0,0]-m.parts['lower_left_infill'].bounds[1,0];np.testing.assert_allclose(clear,params['lower_clear_width_m'],atol=1e-10);nominal['lower_clear_width_mm']=clear*1000
 review.update(parts=len(m.parts),triangles=sum(len(q.faces) for q in m.parts.values()),physical_contact_accuracy_validated=False,
  engine_tested=True,engine='PyBullet',sampled_states=len(poses),maximum_self_penetration_mm=worst*1000,penetrations_over_0_5_mm=failures,
  authored_constraints_numeric_check_mm=nominal,shared_definition=True)
 review['motion_scope']='Lift then turn then open; arbitrary folded-handle turn states can collide. URDF does not implement latch interlocks.'
 save(out/'urdf/review.json',review)
 return m,p,states,review


def run_qy():
 import matplotlib.pyplot as plt
 from .qy import dimensions
 model='c13534';out=outdir(model)
 params=json.loads((out/'parameters.json').read_text())
 m,p,states,review=export_qy_model(out,params)
 if not (out/'index.html').exists():(out/'index.html').write_text('<html><body>Generating review candidate.</body></html>')
 dimensions.OUT=out;dimensions.PACKAGE=out/'urdf';dimensions.DEST=out/'dimensions';dimensions.main()
 # Actual model renders; common camera across the states, no synthetic image generation.
 cam=np.array([-.6,-1.7,1.05]);cam/=np.linalg.norm(cam);u=np.cross([0,0,1],cam);u/=np.linalg.norm(u);v=np.cross(cam,u)
 world=[qy_world(m,state) for state in states.values()];vertices=np.vstack([q.vertices for group in world for q in group.values()])*1000;xx,yy=vertices@u,vertices@v;extent=[xx.min()-20,xx.max()+20,yy.min()-20,yy.max()+20]
 fig,axes=plt.subplots(2,2,figsize=(13,10))
 for ax,name in zip(axes.flat,['closed','upper_open','lower_open','open']):
  ax.imshow(Renderer(qy_world(m,states[name])).image(u,v,cam,extent,width=700));ax.axis('off');ax.set_title(name.replace('_',' '))
 fig.suptitle('C13534 / current geometry review');fig.tight_layout();fig.savefig(out/'preview.png',dpi=140);plt.close(fig)
 selected={n:q for n,q in qy_world(m,{}).items() if m.info[n]['group']=='lower_interior' and not ('lining' in n or 'infill' in n)}
 interior_view(out,selected,model)
 export_scad(out)
 extras(model,out,params,review,m.info)
 print(model,'generated',len(m.parts),'parts; engine tested:',review['engine_tested'],flush=True)
