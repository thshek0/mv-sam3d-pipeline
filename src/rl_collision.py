"""Generate compact RL collision alternatives, retaining original visuals/compounds.

python -m src.rl_collision --root output/hestia-rl
Only expensive plate/bird compounds are replaced by an explicitly named RL variant.
"""
from pathlib import Path
import argparse,json
import numpy as np
import trimesh
import open3d as o3d
from src.collision import validate_parts
from src.export import sha256
from src.assets import save_manifest
from src.products_mujoco import load_parts
from src.products import probe_collision


def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def scene_write(parts,path):
    s=trimesh.Scene()
    for k,m in parts.items():s.add_geometry(m,node_name=k,geom_name=k)
    s.export(path)


def coarse_plate(source):
    from src.plates.rebuild import keep_profile, read_profile, NAMES
    name=source.stem.removesuffix('_m')
    if name in NAMES:
        p,d,profile=read_profile(name)
    else:
        folder=source.parent;p=json.loads((folder/'parameters.json').read_text());d=json.loads((folder/'dimensions.json').read_text())
        profile=np.loadtxt(folder/p['profile_csv'],delimiter=',',skiprows=1)
    r,t,b=profile.T
    r*=d['outer_diameter_mm']/(2*r[-1]);factor=d['overall_height_mm']/t.max();t*=factor;b*=factor
    ti,_=keep_profile(r,t,.7);bi,_=keep_profile(r,b,.7);knots=np.unique(np.r_[r[ti],r[bi]])
    top=np.interp(knots,r,t);bottom=np.interp(knots,r,b)
    parts={};na=24
    for j in range(na):
        angles=np.array([j,j+1])*2*np.pi/na
        for k in range(len(knots)-1):
            points=[[knots[i]*np.cos(a),knots[i]*np.sin(a),z] for i in (k,k+1) for a in angles for z in (bottom[i],top[i])]
            mesh=trimesh.convex.convex_hull(np.array(points)/1000)
            if mesh.is_volume:parts[f'sector_{j:02}_band_{k:02}']=mesh
    rr,aa=np.meshgrid(np.linspace(0,r[-1]*.95,35),np.linspace(0,2*np.pi,64,endpoint=False));rr=rr.ravel();aa=aa.ravel()
    probes=np.c_[rr*np.cos(aa),rr*np.sin(aa),np.interp(rr,r,t)+1.5]/1000
    assert not probe_collision(parts,probes*1000).any(),'Plate cavity blocked'
    return parts,{'construction':'24 angular sectors and profile simplification at 0.7 mm tolerance; separate radial cells preserve dish opening.','radial_knots':knots.tolist(),'maximum_angular_chord_error_mm':float(r[-1]*(1-np.cos(np.pi/na))),'open_cavity_probes':len(probes),'original_angular_sections':p['angular_sections']}


def coarse_bird(source):
    import coacd
    m=trimesh.load(source,force='mesh');o=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices),o3d.utility.Vector3iVector(m.faces));o=o.simplify_quadric_decimation(4000)
    coacd.set_log_level('warn')
    settings=dict(threshold=.08,max_convex_hull=32,preprocess_mode='on',preprocess_resolution=40,resolution=1000,mcts_nodes=10,mcts_iterations=40,mcts_max_depth=3,merge=True,decimate=True,max_ch_vertex=32,seed=0)
    raw=coacd.run_coacd(coacd.Mesh(np.asarray(o.vertices),np.asarray(o.triangles)),**settings)
    parts={f'part_{i:02}':trimesh.Trimesh(v,f).convex_hull for i,(v,f) in enumerate(raw)}
    return parts,{'construction':'CoACD convex approximation capped at 32 pieces / 32 vertices each.','settings':settings,'units':'Uncalibrated source coordinates; actual bird dimensions are not recorded.'}


def surface_error(source,parts):
    visual=trimesh.load(source,force='mesh');compound=trimesh.util.concatenate(list(parts.values()));s=o3d.t.geometry.RaycastingScene(nthreads=4)
    s.add_triangles(o3d.core.Tensor(compound.vertices.astype('float32')),o3d.core.Tensor(compound.faces.astype('uint32')))
    points=visual.vertices[np.linspace(0,len(visual.vertices)-1,min(10000,len(visual.vertices)),dtype=int)]
    distance=s.compute_distance(o3d.core.Tensor(points.astype('float32')),nthreads=4).numpy()
    return {'sampled_surface_distance_p95':float(np.percentile(distance,95)),'sampled_surface_distance_max':float(distance.max()),'source_max_extent':float(visual.extents.max()),'note':'Nearest hull surface distance; not a global Hausdorff bound.'}


def build(root):
    inventory=[]
    for manifest in sorted([*root.glob('*/asset.json'), *root.glob('*/simulation/asset.json')]):
        d=json.loads(manifest.read_text());name=manifest.parent.parent.name if manifest.parent.name=='simulation' else manifest.parent.name
        for variant,entry in list(d['variants'].items()):
            if variant.endswith('_rl'):continue
            if 'visual' not in entry:continue
            source=(manifest.parent/entry['visual']['path']).resolve()
            if name.startswith('plate_'):
                parts,report=coarse_plate(source)
                old_count=len(load_parts(manifest.parent/entry['collision']['path'])) if 'collision' in entry else None
                out=manifest.parent/'rl_collision';out.mkdir(exist_ok=True)
                collision=out/(variant+'_rl.glb');scene_write(parts,collision);validate_parts(parts)
                report.update(parts=len(parts),triangles=sum(len(m.faces) for m in parts.values()),previous_parts=old_count,surface_check=surface_error(source,parts),source_sha256=sha256(source))
                save(out/(variant+'_validation.json'),report)
                new=dict(entry);new.pop('collision_recipe',None)
                new['collision']={'path':str(collision.relative_to(manifest.parent)),'sha256':sha256(collision)}
                previous=d['variants'].get(variant+'_rl',{})
                if 'contact_settings' in previous:new['contact_settings']=previous['contact_settings']
                d['variants'][variant+'_rl']=new
                inventory.append({'object':name,'variant':variant+'_rl','action':'Created simplified RL alternative',**report})
            elif 'collision' in entry:
                parts=load_parts(manifest.parent/entry['collision']['path']);validate_parts(parts)
                inventory.append({'object':name,'variant':variant,'action':'Reused existing convex assembly','parts':len(parts),'triangles':sum(len(m.faces) for m in parts.values())})
        manifest.write_text(json.dumps(d,indent=2)+'\n')
    for folder in sorted(root.glob('bird*/sam3d')):
        source=folder/'hybrid_sam3d_mesh.glb'
        if not source.exists():continue
        out=folder/'rl_collision';out.mkdir(exist_ok=True);collision=out/'bird_rl.glb'
        report_path=out/'validation.json'
        if report_path.exists() and json.loads(report_path.read_text()).get('source_sha256')==sha256(source):
            parts=load_parts(collision);report=json.loads(report_path.read_text())
        else:
            parts,report=coarse_bird(source);validate_parts(parts);scene_write(parts,collision)
            report.update(parts=len(parts),triangles=sum(len(m.faces) for m in parts.values()),surface_check=surface_error(source,parts),source_sha256=sha256(source))
            save(report_path,report)
        source_manifest=folder/'asset.json'
        save_manifest(source_manifest,kind='reconstruction_with_rl_colliders',sources=[source,Path(__file__),report_path],variants={folder.parent.name+'_rl':{'visual':source,'collision':collision}},units='source_units',frame='Original reconstruction coordinates; physical scale uncalibrated',rebuild=['python','-m','src.rl_collision','--root',str(root)])
        inventory.append({'object':folder.parent.name,'action':'Created simplified RL alternative',**report})
        print(folder.parent.name,report['parts'],'pieces',flush=True)
    out=root.parent/'comparisons/hestia_rl_collision';out.mkdir(parents=True,exist_ok=True);save(out/'inventory.json',inventory)
    print('Inventory:',out/'inventory.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);build(p.parse_args().root.resolve())
