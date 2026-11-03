"""Instrument builder: reusable implementation behind the instruments CLI."""
from pathlib import Path
import json,shutil
import numpy as np
from . import construction as prior
from . import geometry as base
from .paths import ROOT

def save(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def current_model_dir(model):return ROOT/'output/instruments'/model/'model'
def setup(model):
 out=ROOT/'output/instruments'/model/'.build_work'
 if out.exists():
  import tempfile
  failures=out.parent/'validation/failures';failures.mkdir(parents=True,exist_ok=True)
  retained=Path(tempfile.mkdtemp(prefix='previous_build_',dir=failures))
  out.rename(retained/'build_work')
 out.mkdir(parents=True)
 shutil.copytree(ROOT/'dataset/instruments'/model/'references/model_evidence',out/'evidence',dirs_exist_ok=True)
 current=current_model_dir(model)/'parameters.json'
 if current.exists():
  shutil.copy2(current,out/'parameters.json')
 else:
  from .inputs import initial_parameters
  save(out/'parameters.json',initial_parameters(model))
 # CAD export invalidates these cached checks if any generated SCAD input changes.
 if (current_model_dir(model)/'openscad').exists():
  shutil.copytree(current_model_dir(model)/'openscad',out/'openscad',dirs_exist_ok=True)
 return out,json.loads((out/'parameters.json').read_text())

def simple(model,p):
 from .design import build as shared_build
 return shared_build(p)

def run_simple(model,out,p):
 m=simple(model,p);save(out/'parameters.json',p);(out/'meshes').mkdir(exist_ok=True)
 for label,value in [('closed',0),('open',base.open_value(p))]:m.scene(value).export(out/'meshes'/f'collision_{label}.glb')
 path=base.urdf(m,out);review=base.validate(m,out,path)
 if model=='cary60':
  bounds=np.array([m.scene(value).bounds for value in np.linspace(0,base.open_value(p),25)])
  assert bounds[:,0,1].min()>-1e-8 and bounds[:,1,1].max()<=p['body_wdh_m'][1]+1e-8
  review['cover_swept_y_bounds_mm']=[float(bounds[:,0,1].min()*1000),float(bounds[:,1,1].max()*1000)];review['cover_travel_mm']=base.open_value(p)*1000
 if model=='c16361':
  np.testing.assert_allclose(np.mean(p['chamber_x_m']),p['design_parameters']['chamber_center_x_mm']/1000,atol=1e-10)
  review['lid_centered_over_left_chamber']=True;review['round_hole_nominal_mm']=p['detail_parameters']['hole_diameter_m']*1000
 base.render_views(m,out)
 exclude=('housing','wall','cover','rail','lid','front_plinth','compartment','removable_front','back_')
 interior={n:q for n,q in m.parts.items() if not any(w in n for w in exclude) or n.startswith(('stage_','holder_','socket_','inner_','perforated_','floor_fastener','front_latch','micrometer','side_'))}
 prior.interior_view(out,interior,model)
 info={n:dict(group=m.groups[n],bounds_closed_m=q.bounds.tolist(),evidence=m.evidence[n]) for n,q in m.parts.items()}
 prior.extras(model,out,p,review,info)
 # Keep the same moving geometry in each pose and preserve editable local frames.
 return m


def build(models,from_cad_parameters=False,check_cad=False,keep_validation=False,extras=False):
 from .storage.diagnostics import session
 for model in models:
  with session(current_model_dir(model).parent,keep=keep_validation):
   _build_one(model,from_cad_parameters,check_cad,extras)


def _build_one(model,from_cad_parameters,check_cad,extras):
  out,p=setup(model)
  from .storage.diagnostics import capture
  capture(out/'parameters.json','inputs_and_reports/parameters.json')
  if from_cad_parameters:
   from .shared_geometry import read_parameter_include
   from .design import parameters,module
   values,_=parameters(p)
   imported=read_parameter_include(current_model_dir(model)/'openscad/parameters_native.scad',values)
   module(model).definition(imported)
   for key,value in imported.items():
    if value!=values[key]:p.setdefault('design_parameter_evidence',{})[key]='Imported CAD parameter edit; measurement/source not independently verified'
   p['design_parameters']=imported
  if model=='c13534':
   save(out/'parameters.json',p)
   prior.run_qy()
  else:
   m=run_simple(model,out,p)
   prior.export_scad(out)
  from . import textures as tex
  report=tex.apply(model,out)
  report['urdf_textured_visuals']=tex.urdf_visuals(model,out,report['coverage'])
  from .decals import apply as apply_decals
  apply_decals(model,out)
  report['interior_parts_shown']=tex.render_review(model,out)
  save(out/'textures/provenance.json',report)
  if check_cad:
   from .cad import check
   check(out/'openscad')
  from .publish import publish,overview
  publish(model,source=out,extras=extras)
  shutil.rmtree(out)
  overview()
  print(model,'detail geometry generated',flush=True)
