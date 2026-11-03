"""Authoritative parameter ownership and shared model dispatch for all instruments."""
import importlib,json
from pathlib import Path

def module(name):return importlib.import_module('.'+name,__package__)

def parameters(p):
 name=p['model']
 from .inputs import read
 defaults=read(name,'design_defaults.json')
 v=dict(defaults['values']);evidence=dict(defaults['evidence'])
 if 'design_parameters' in p:
  if set(p['design_parameters'])!=set(v):raise ValueError(name+' design parameter schema mismatch')
  v.update(p['design_parameters'])
 evidence.update(p.get('design_parameter_evidence',{}))
 module(name).definition(v)
 return v,evidence

def build(p):
 v,_=parameters(p);p['design_parameters']=v;p['authoring']='Shared part/joint definition; design_parameters is authoritative (mm/degrees)'
 if p['model']=='cary60':return module('cary60').model(p)
 return module(p['model']).model(p,v)

def decal_target(name,label,bounds,y,v):
 """Photo markings stay attached to their corresponding panel during parameter edits."""
 x0,x1,z0,z1=[x*1000 for x in bounds]
 if name=='cary60':return [x/1000 for x in module(name).decal_bounds(v)],y
 if name=='c16361':
  H=v['overall_height_mm']-v['lid_height_mm']
  if label=='front_identity':
   edge=v['chamber_center_x_mm']+v['clear_width_mm']/2+v['wall_mm']
   x0,x1=[edge+(x-237)/213*(v['body_width_mm']-edge) for x in (x0,x1)]
   z0,z1=[z*H/230.8 for z in (z0,z1)]
  else:
   x0,x1=[x*v['body_width_mm']/450 for x in (x0,x1)]
   z0,z1=[z*v['plinth_height_mm']/40 for z in (z0,z1)]
 elif name=='c13472':
  x0,x1=[x*v['body_width_mm']/645 for x in (x0,x1)]
  z0,z1=[z*v['body_height_mm']*v['base_height_fraction']/92.4 for z in (z0,z1)]
 else:
  tower=v['door_width_mm']+v['strip_width_mm']
  x0,x1=[tower+(x-340)/330*(v['body_width_mm']-tower) for x in (x0,x1)]
  z0,z1=[v['base_height_mm']+(z-33)/332*(v['rear_housing_height_mm']-v['base_height_mm']) for z in (z0,z1)]
  y+=(v['control_front_y_mm']-15.83864225091737)/1000
 return [x/1000 for x in (x0,x1,z0,z1)],y
