"""Rebuild the current simplified plates. No simulator or image generator required.
Edit the plate entry in src/product_inputs/designs.json.
"""
from pathlib import Path
import argparse,json,os,sys
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import trimesh
import open3d as o3d
from trimesh.visual.material import PBRMaterial
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.plates.appearance import apply_appearance
from src.output_layout import canonical_name
OUT=ROOT/'output/hestia-rl'
INPUTS=ROOT/'src/product_inputs/designs.json'
NAMES=('plate_maehata_pop_blue_23cm','plate_strimmig_pale_grey_green_21cm','plate_paradisisk_off_white_20cm')

def read_profile(name, source=None):
    data=json.loads((source or INPUTS).read_text())[canonical_name(name)]
    return data['parameters'],data['dimensions'],np.array(data['profile_radius_top_bottom_mm'],dtype=float)

def keep_profile(r,z,tolerance):
    selected={0,len(r)-1,int(np.argmin(z)),int(np.argmax(z))}
    while True:
        indices=sorted(selected);errors=np.abs(z-np.interp(r,r[indices],z[indices]));index=int(np.argmax(errors))
        if errors[index]<=tolerance:return indices,float(errors[index])
        selected.add(index)

def audit(mesh):
    om=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces))
    counts=np.bincount(mesh.edges_unique_inverse)
    checks={'vertices':len(mesh.vertices),'triangles':len(mesh.faces),'watertight':bool(mesh.is_watertight),
        'winding_consistent':bool(mesh.is_winding_consistent),'positive_volume':bool(mesh.is_volume),
        'connected_components':len(mesh.split(only_watertight=False)),'boundary_edges':int((counts==1).sum()),
        'nonmanifold_edges':int((counts>2).sum()),'self_intersecting_triangle_pairs':len(om.get_self_intersecting_triangles()),
        'extents_mm':mesh.extents.tolist()}
    assert checks['watertight'] and checks['winding_consistent'] and checks['positive_volume'],checks
    assert checks['connected_components']==1 and checks['boundary_edges']==checks['nonmanifold_edges']==checks['self_intersecting_triangle_pairs']==0,checks
    return checks

def build(name,destination=None):
    name=canonical_name(name)
    dest=Path(destination) if destination else OUT/name;dest.mkdir(parents=True,exist_ok=True)
    params,dims,profile=read_profile(name)
    r,top,bottom=profile.T
    r=r*(dims['outer_diameter_mm']/(2*r[-1]));scale=dims['overall_height_mm']/top.max();top=top*scale;bottom=bottom*scale
    sections=params['angular_sections'];assert sections>=12 and sections%4==0
    ti,te=keep_profile(r,top,params['profile_tolerance_mm']);bi,be=keep_profile(r,bottom,params['profile_tolerance_mm'])
    polygon=np.vstack([np.c_[r[bi],bottom[bi]],np.c_[r[ti[::-1]],top[ti[::-1]]]])
    mesh=trimesh.creation.revolve(polygon,sections=sections)
    if mesh.volume<0:mesh.invert()
    # Preserve the previous float32 STL roundtrip without keeping a second mesh format.
    import io
    mesh=trimesh.load(io.BytesIO(mesh.export(file_type='stl')),file_type='stl',force='mesh');checks=audit(mesh)
    assert np.allclose(mesh.extents,[dims['outer_diameter_mm']]*2+[dims['overall_height_mm']],atol=.0001)
    dense=np.linspace(0,r[-1],10001);gap=np.interp(dense,r[ti],top[ti])-np.interp(dense,r[bi],bottom[bi]);assert gap.min()>2.0
    base=trimesh.creation.revolve(np.vstack([np.c_[r,bottom],np.c_[r[::-1],top[::-1]]]),sections=256)
    np.random.seed(42);errors=[]
    for a,b in ((mesh,base),(base,mesh)):
        scene=o3d.t.geometry.RaycastingScene();om=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(b.vertices),o3d.utility.Vector3iVector(b.faces));scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(om))
        points=np.vstack([a.vertices,trimesh.sample.sample_surface(a,12000)[0]])
        errors.append(scene.compute_distance(o3d.core.Tensor(points.astype('float32'))).numpy())
    distances=np.concatenate(errors);assert distances.max()<1.0
    metres=mesh.copy();metres.apply_scale(.001)
    metres,appearance=apply_appearance(metres,name,params,dest)
    metres.export(dest/f'{name}_m.glb',include_normals=True)
    for suffix in ('m.glb',):
        loaded=trimesh.load(dest/f'{name}_{suffix}',force='mesh')
        loaded.merge_vertices(merge_tex=True,merge_norm=True)  # Weld UV seams for topology checks.
        assert loaded.is_volume and loaded.is_watertight and len(loaded.faces)==len(mesh.faces)
        assert np.allclose(loaded.bounds*1000,mesh.bounds,atol=.00003)
    report={'model':name,'representation':'Current simplified mesh (canonical filename)',
        'purpose':'Lightweight single closed mesh for RL geometry workflows','angular_sections':sections,
        'source_triangles':params['source_detailed_triangles'],
        'triangle_reduction_percent':100*(1-len(mesh.faces)/params['source_detailed_triangles']),
        'prior_detailed_stl_sha256':params['source_detailed_stl_sha256'],
        'top_profile_knots':len(ti),'bottom_profile_knots':len(bi),'maximum_profile_height_error_mm':max(te,be),
        'maximum_outer_circle_chord_error_mm':float(r[-1]*(1-np.cos(np.pi/sections))),
        'sampled_bidirectional_smooth_body_distance_mm':{'maximum':float(distances.max()),'p95':float(np.percentile(distances,95)),'samples':len(distances)},
        'minimum_sampled_vertical_thickness_mm':float(gap.min()),'mesh_checks':checks,
        'profile_mm':{'radius':r.tolist(),'top':top.tolist(),'bottom':bottom.tolist()},
        'dimensions':dims,'dimension_sources':dims['dimension_sources'],
        'appearance_check':appearance,
        'units':{'glb':'m','source_profile':'mm'},
        'origin':'Centre in XY, support plane Z=0, Z up',
        'limitations':['Single concave mesh; MuJoCo uses a convex hull for one mesh collider, filling the cavity. Decompose for inside-dish contact.',
            'Geometry checked; no simulation or performance benchmark run.',
            'Physical decorative relief omitted. Surface error is relative to the smooth body; print is visual only.',
            'Original dimensional uncertainty and inferred underside/thickness remain.'],
        'mujoco_reference':'https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh'}
    (dest/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    from src.assets import save_manifest
    inputs=[Path(__file__),Path(__file__).with_name('appearance.py'),
            ROOT/'src/rl_collision.py',INPUTS]
    if params.get('appearance_recipe'):inputs.append(ROOT/params['appearance_recipe'])
    # Refresh the configured RL variant with the same profile as the visual.
    from src.rl_collision import coarse_plate, scene_write
    from src.collision import validate_parts
    coarse,coarse_report=coarse_plate(dest/f'{name}_m.glb');validate_parts(coarse)
    coarse_out=dest/'rl_collision';coarse_out.mkdir(exist_ok=True)
    coarse_path=coarse_out/(name+'_rl.glb');scene_write(coarse,coarse_path)
    (coarse_out/(name+'_validation.json')).write_text(json.dumps(coarse_report,indent=2)+'\n')
    save_manifest(dest/'asset.json',
                  kind='radial_profile',sources=inputs,
                  variants={name:{'visual':dest/f'{name}_m.glb',
                                  'collision_recipe':INPUTS},
                            name+'_rl':{'visual':dest/f'{name}_m.glb','collision':coarse_path,
                                        'contact_settings':ROOT/'src/product_inputs/rl_contacts.json'}},units='m',
                  frame='XY centre, Z up; support Z=0',
                  rebuild=['python','-m','src.plates.rebuild',name])
    print(name,len(mesh.faces),'triangles, watertight, dimensions verified',flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('names',nargs='*',metavar='PLATE');parser.add_argument('--output-root',type=Path,help='Optional scratch output for verification; source parameters remain in the workspace')
    args=parser.parse_args()
    args.names=[canonical_name(name) for name in args.names]
    if any(name not in NAMES for name in args.names):parser.error('Valid plate names: '+', '.join(NAMES))
    for name in args.names or NAMES:build(name,args.output_root/name if args.output_root else None)
