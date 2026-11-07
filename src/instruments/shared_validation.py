"""Parameter-change integration checks across native CAD, GLBs, textures and URDF."""
import copy,json
from .paths import ROOT
from .storage.diagnostics import workspace
from pathlib import Path
import numpy as np
import trimesh
from . import design,geometry,textures,decals
from .shared_geometry import read_parameter_include

def changed_values(name,v):
 if name=='c16361':return dict(clear_width_mm=v['clear_width_mm']+10,hole_diameter_mm=12,chamber_depth_mm=v['chamber_depth_mm']+5,overall_height_mm=v['overall_height_mm']+10,open_angle_deg=80)
 if name=='c13472':return dict(body_width_mm=v['body_width_mm']+20,body_height_mm=v['body_height_mm']+10,holder_hole_diameter_mm=48,cover_travel_fraction=.30)
 return dict(body_width_mm=v['body_width_mm']+10,door_width_mm=v['door_width_mm']+5,lower_clear_width_mm=135,lower_pin_spacing_mm=44,lower_recess_diameter_mm=50,door_open_angle_deg=85,selector_travel_mm=35)

def world(m,state,name):
 if name=='c13534':
  from .construction import qy_world
  return qy_world(m,state)
 return m.world_parts(state)

def check(dest,render,values):
 p=json.loads((dest.parent/'parameters.json').read_text());name=p['model'];changes=changed_values(name,values);edited=dict(values,**changes)
 # QY's portable exporter records source hashes relative to the workspace.
 with workspace('edited_parameters') as tmp:
  from .inputs import stage_evidence
  work=Path(tmp);(work/'meshes').mkdir();stage_evidence(p['model'],work)
  include=work/'edited_parameters.scad';include.write_text('\n'.join(f'{k} = {v};' for k,v in edited.items()))
  p['design_parameters']=read_parameter_include(include,values)
  if name=='c13534':
   from .construction import export_qy_model
   m,_,states,physics=export_qy_model(work,p)
  else:
   m=design.build(p);states={'closed':0,'open':geometry.open_value(p)}
   for state,value in states.items():m.scene(value).export(work/'meshes'/f'collision_{state}.glb')
   path=geometry.urdf(m,work);physics=geometry.validate(m,work,path)
   (work/'parameters.json').write_text(json.dumps(p,indent=2))
  parts=world(m,states['closed'],name);errors={}
  definition=design.module(name).definition(edited)
  for part in definition.parts:
   native=render('shared_'+part.name,part=part.name,**changes)
   if part.kind=='subtract':assert len(native.split())==1,(part.name,'Disconnected Boolean solid')
   group=[q for n,q in parts.items() if n==part.name or n.startswith(part.name+'__')]
   combined=trimesh.util.concatenate(group)
   error=float(np.max(abs(native.bounds-combined.bounds*1000)))
   # Smooth spheres and 12-sided circular profiles differ slightly from convex proxies.
   tolerance=.2 if part.kind=='ellipsoid' else .03
   np.testing.assert_allclose(native.bounds,combined.bounds*1000,atol=tolerance,err_msg=part.name)
   volume_error=abs(native.volume-combined.volume*1e9)/(combined.volume*1e9)
   volume_limit=.17 if part.kind=='ellipsoid' else .055 if part.kind=='radial_profile' else .03 if part.kind in ('rod','rounded_x','capsule_y','quarter_disc') else .005 if part.kind=='plate_hole' else 1e-4
   assert volume_error<volume_limit,(part.name,volume_error,volume_limit)
   errors[part.name]=dict(bounds_max_error_mm=error,volume_relative_error=float(volume_error),bounds_tolerance_mm=tolerance,volume_tolerance=volume_limit,convex_pieces=len(group))
  pose_errors={}
  for state,value in states.items():
   native=render('shared_edited_'+state,pose=state,**changes)
   # Rotating the smoother CAD lid/handle changes extreme vertices relative to coarse collision arcs.
   tolerance=1.0 if name=='c13534' else .2 if name=='c16361' else .03
   pose_errors[state]=float(np.max(abs(native.bounds-m.scene(value).bounds*1000)))
   np.testing.assert_allclose(native.bounds,m.scene(value).bounds*1000,atol=tolerance)
  report=textures.apply(name,work);textures.urdf_visuals(name,work,report['coverage']);decals.apply(name,work)
  for state,value in states.items():
   visual=trimesh.load(work/'meshes'/f'visual_{state}.glb',force='scene')
   for key,q in world(m,value,name).items():
    transform,geom=visual.graph[key];actual=visual.geometry[geom].copy();actual.apply_transform(transform)
    np.testing.assert_allclose(actual.bounds,q.bounds,atol=1e-7,err_msg=key)
    assert actual.visual.kind=='texture',key
  nominal=design.build(copy.deepcopy(json.loads((dest.parent/'parameters.json').read_text())))
  oldstate=nominal.demo_states['open'] if name=='c13534' else geometry.open_value(nominal.p)
  preview(dest,name,[world(nominal,oldstate,name),world(m,states['open'],name)],changes)
 return {'edited_values':changes,'parts_checked':errors,'convex_pieces_checked':len(m.parts),
  'numeric_scad_parameter_import':True,'textured_glb_bounds_match_collision':True,
  'urdf_geometry_and_joint_transforms_verified':True,'pybullet_sweep_states':physics['sampled_states'],
  'maximum_self_penetration_mm':physics['maximum_self_penetration_mm'],
  'pose_envelope_errors_mm':pose_errors,'pose_envelope_tolerance_mm':tolerance,'test_changes_published_as_model':False}

def preview(dest,name,worlds,changes):
 from .render import Renderer
 import matplotlib.pyplot as plt
 cam=np.array([.6,-1.6,1.2]);cam/=np.linalg.norm(cam);u=np.cross([0,0,1],cam);u/=np.linalg.norm(u);v=np.cross(cam,u)
 points=np.vstack([q.vertices for parts in worlds for q in parts.values()])*1000
 xx,yy=points@u,points@v;extent=[xx.min()-15,xx.max()+15,yy.min()-15,yy.max()+15]
 fig,axes=plt.subplots(1,2,figsize=(13,7))
 for ax,parts,label in zip(axes,worlds,['Current dimensions','Temporary parameter-change test']):
  ax.imshow(Renderer(parts).image(u,v,cam,extent,width=850));ax.axis('off');ax.set_title(label)
 fig.suptitle(name.upper()+': one shared definition drives CAD, GLB and URDF')
 fig.text(.04,.04,'Test values: '+', '.join(f'{k.replace("_mm","")}={v:g}' for k,v in changes.items()),fontsize=8,wrap=True)
 fig.tight_layout(rect=(0,.07,1,.94));fig.savefig(dest/'csg_shared_parameters.png',dpi=140);plt.close(fig)
