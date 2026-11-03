"""Native CSG export and OpenSCAD checks from shared CAD/simulation definitions."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess
import numpy as np
import trimesh
from .paths import ROOT

def export(out):
 p=json.loads((out/'parameters.json').read_text())
 model=p['model']
 dest=out/'openscad';dest.mkdir(exist_ok=True);old=dest/'assembly.scad'
 def signature():
  files=[old,dest/'parameters_native.scad']
  if old.exists() and 'common.scad' in old.read_text():files.append(dest/'common.scad')
  return hashlib.sha256(b''.join(path.read_bytes() if path.exists() else b'' for path in files)).hexdigest()
 before=signature()
 (dest/'faceted_reference.scad').unlink(missing_ok=True)
 from .design import parameters,module
 values,evidence=parameters(p)
 module(model).definition(values).export_scad(old)
 shutil.copy2(Path(__file__).with_name('scad')/'common.scad',dest/'common.scad')
 (dest/'parameters_native.scad').write_text('// Edit here or override with -D. Units: mm / degrees; *_fraction values are dimensionless.\n// Unnamed construction offsets in assembly.scad remain photo/model estimates.\n'+''.join(f'// {evidence.get(k,"Photo/model estimate; not a precision measurement")}\n{k} = {json.dumps(v)};\n' for k,v in values.items()))
 (dest/'parameters_native.json').write_text(json.dumps({'model':model,'units':'mm and degrees; fractions dimensionless','values':values,'evidence':{k:evidence.get(k,'estimate') for k in values}},indent=2)+'\n')
 if before!=signature():
  for path in dest.glob('csg_*'):
   if path.suffix in ('.json','.stl','.png'):path.unlink()
 (dest/'README.txt').write_text('FILE ROLES\nassembly.scad: generated entry point; OPEN this file.\nparameters_native.scad: editable numeric dimensions.\ncommon.scad: required reusable geometry helpers.\n\nAll four machines share their CAD, simulation geometry and joint definitions.\nCanonical parameters: ../parameters.json -> design_parameters (mm/degrees).\nImport numeric SCAD parameter edits and regenerate CAD, textured GLBs, collision meshes and URDF with:\n.venv/bin/python -m src.instruments build --model '+model+' --from-cad-parameters --check-cad\nChecks require OpenSCAD on PATH or OPENSCAD=/path/to/openscad.\nTopology edits belong in src/instruments/'+model+'.py. Arbitrary generated SCAD code is not imported.\nCurves can have finer CAD tessellation than convex simulation pieces. CAD STL snapshots contain no textures or joints.\nSee ../reports/cad_summary.json for check coverage. Use --keep-validation to retain detailed checks and comparisons. Local contact dimensions and motion limits remain provisional.\n')
 return dest

def check(dest,executable=None,keep_validation=False):
 from .storage.diagnostics import session, capture
 with session(dest.parent.parent, keep=keep_validation):
  for name in ('parameters.json', 'openscad/assembly.scad', 'openscad/parameters_native.scad', 'openscad/common.scad'):
   capture(dest.parent/name, 'inputs_and_reports/'+name)
  report=_check(dest,executable)
  for path in dest.glob('csg_*'):capture(path,'review/openscad/'+path.name)
  return report


def _check(dest,executable=None):
 executable=executable or os.environ.get('OPENSCAD') or shutil.which('openscad')
 if not executable:raise RuntimeError('Set OPENSCAD to an executable, or install OpenSCAD, to validate native CSG.')
 model=json.loads((dest.parent/'parameters.json').read_text())['model']
 if model!='c16361':
  from .cad_checks import check as check_other
  return check_other(dest,executable,model)
 values=json.loads((dest/'parameters_native.json').read_text())['values'];results={};meshes={}
 from .storage.diagnostics import workspace, command_log
 with workspace('native_csg') as tmp:
  def render(name,definitions):
   path=Path(tmp)/(name+'.stl');cmd=[executable,'-o',str(path)]
   for key,value in definitions.items():cmd+=['-D',f'{key}={json.dumps(value)}']
   cmd.append(str((dest/'assembly.scad').resolve()))
   run=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
   command_log(path,cmd,run)
   if run.returncode or not path.exists() or 'ERROR:' in run.stderr:raise RuntimeError(run.stderr)
   mesh=trimesh.load(path,force='mesh');assert mesh.is_watertight,(name,'non-watertight CSG')
   meshes[name]=mesh.copy()
   results[name]={'bounds_mm':mesh.bounds.tolist(),'watertight':True,'triangles':len(mesh.faces)}
   return mesh
  for pose in ['closed','open']:
   mesh=render(pose,{'pose':pose})
   current=trimesh.load(dest.parent/'meshes'/f'collision_{pose}.glb',force='scene')
   np.testing.assert_allclose(mesh.bounds,current.bounds*1000,atol=.02)
   mesh.export(dest/f'csg_{pose}.stl')
  for name,overrides in [('floor',{}),('wider_floor',{'clear_width_mm':210,'hole_diameter_mm':12}),('deeper_floor',{'chamber_depth_mm':55})]:
   mesh=render(name,{'part':'floor',**overrides});v=dict(values,**overrides)
   width=v['clear_width_mm'];depth=v['chamber_rear_y_mm']+v['front_projection_mm']-v['wall_mm'];thick=v['floor_thickness_mm']
   np.testing.assert_allclose(mesh.extents,[width,depth,thick],atol=.02)
   np.testing.assert_allclose(mesh.bounds[1,2],v['overall_height_mm']-v['lid_height_mm']-v['chamber_depth_mm'],atol=.02)
   r=v['hole_diameter_mm']/2;hole_area=64/2*r*r*np.sin(2*np.pi/64)
   np.testing.assert_allclose(mesh.volume,(width*depth-hole_area)*thick,rtol=2e-5)
   expected=[v['chamber_center_x_mm'],(-v['front_projection_mm']+v['wall_mm']+v['chamber_rear_y_mm'])/2]
   distances=np.linalg.norm(mesh.vertices[:,:2]-expected,axis=1);hole_vertices=mesh.vertices[distances<r+.02]
   assert len(hole_vertices)>0
   np.testing.assert_allclose(hole_vertices[:,:2].mean(0),expected,atol=.02)
   results[name]['hole_center_verified']=True
  mesh=render('taller_wider',{'overall_height_mm':283.3,'clear_width_mm':210})
  np.testing.assert_allclose(mesh.bounds[1,2],283.3,atol=.02)
  from .shared_validation import check as shared_check
  shared_report=shared_check(dest,lambda name,**defs:render(name,defs),values)
 report={'engine':subprocess.run([executable,'--version'],capture_output=True,text=True).stderr.strip(),
  'status':'Shared CAD/simulation part and joint definition','shared_backend_checks':shared_report,'internal_topology_repairs':['Walls extend through floor thickness to join with area instead of edge-only contacts.','Grip backing overlaps side/front frame to form a connected solid.'],'checks':results,'default_envelope_matches_collision_meshes':True}
 preview(dest,meshes)
 (dest/'csg_validation.json').write_text(json.dumps(report,indent=2)+'\n')
 return report


def preview(dest,meshes):
 import matplotlib.pyplot as plt
 from .render import Renderer
 figure,axes=plt.subplots(2,2,figsize=(13,10))
 camera=np.array([.6,-1.6,1.2]);camera/=np.linalg.norm(camera)
 u=np.cross([0,0,1],camera);u/=np.linalg.norm(u);v=np.cross(camera,u)
 cases=[('closed','Native CSG / closed / H 253.3 mm'),('open','Native CSG / open'),('floor','Floor: width 190 mm / hole 9.5 mm'),('wider_floor','Parameter edit: width 210 mm / hole 12 mm')]
 for ax,(name,label) in zip(axes.flat,cases):
  mesh=meshes[name].copy();vertices=mesh.vertices.copy();mesh.apply_scale(.001);mesh.visual.face_colors=[150,158,170,255]
  x,y=vertices@u,vertices@v;extent=[x.min()-10,x.max()+10,y.min()-10,y.max()+10]
  ax.imshow(Renderer({'csg':mesh}).image(u,v,camera,extent,width=900));ax.axis('off');ax.set_title(label)
 figure.suptitle('Tau: actual OpenSCAD CSG renders and parameter-change checks',fontsize=17)
 figure.text(.03,.02,'Shared part/joint definition; parameter import rebuilds CAD, textured GLB and URDF. Local dimensions remain estimates.',fontsize=10)
 figure.tight_layout(rect=(0,.05,1,.95));figure.savefig(dest/'csg_review.png',dpi=140);plt.close(figure)
