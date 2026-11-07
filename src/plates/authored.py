"""Build a photo-textured dish from an editable radial profile, without CAD.

Run: .venv/bin/python -m src.plates.authored --parameters src/plates/inputs/plate_harmynta_white_22cm.json
Decoded source PNGs live in codex/intermediate/reference; original captures are untouched.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.material import PBRMaterial

from src.collision import plate_parts
from src.assets import save_manifest
from src.plates.rebuild import audit, keep_profile
from src.products import probe_collision
from src.products_mujoco import inertial_properties

ROOT = Path(__file__).resolve().parents[2]


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def photo_texture(recipe, references):
    """Sample two photo ellipses into a planar atlas and suppress broad lighting.

    The underside's central label is retained; specular microstructure may remain.
    No invented decorative pattern or baked broad reflection is added.
    """
    n = recipe['tile_size']
    tiles = []
    for side in ('top', 'bottom'):
        spec = recipe[side]
        with Image.open(references/spec['image']) as source:
            rgb = np.asarray(source.convert('RGB'), dtype=np.float32)
        sx, sy = rgb.shape[1]/spec['review_size'][0], rgb.shape[0]/spec['review_size'][1]
        x0, y0, x1, y1 = np.array(spec['ellipse_bbox'])*[sx,sy,sx,sy]
        # Keep a 2% margin matching the mesh UV mapping.
        v, u = np.mgrid[:n,:n].astype(np.float32)
        xx = (x0+x1)/2+(u/n-.5)/.96*(x1-x0)
        yy = (y0+y1)/2+(v/n-.5)/.96*(y1-y0)
        patch = cv2.remap(rgb,xx,yy,cv2.INTER_LINEAR)
        low = cv2.GaussianBlur(patch,(0,0),12)
        residual = np.clip(patch-low,-65,5)
        tile = np.array(recipe['background_rgb'])+residual
        if side == 'bottom':
            # Preserve the actual photographed maker print and removable sticker.
            central = ((u/n-.5)**2+(v/n-.5)**2) < .205**2
            smooth = cv2.GaussianBlur(patch,(0,0),70)
            corrected = patch*np.array(recipe['background_rgb'])/np.maximum(smooth,1)
            tile[central] = corrected[central]
        outside = ((u/n-.5)**2+(v/n-.5)**2) > .479**2
        tile[outside] = recipe['background_rgb']
        tiles.append(np.uint8(np.clip(tile,0,255)))
    return Image.fromarray(np.concatenate(tiles,axis=1))


def texture_mesh(mesh, texture, roughness):
    upper = mesh.face_normals[:,2] > 0
    keys, inverse = np.unique(np.c_[mesh.faces.ravel(),np.repeat(upper,3)],axis=0,return_inverse=True)
    indices = keys[:,0]
    result = trimesh.Trimesh(mesh.vertices[indices],inverse.reshape(-1,3),
                             vertex_normals=mesh.vertex_normals[indices],process=False)
    xy = result.vertices[:,:2]/np.linalg.norm(mesh.vertices[:,:2],axis=1).max()
    uv = np.c_[np.where(keys[:,1],.25,.75)+xy[:,0]*.24,.5+xy[:,1]*.48]
    result.visual = trimesh.visual.TextureVisuals(uv=uv, material=PBRMaterial(
        name='glazed_porcelain',baseColorFactor=[255]*4,baseColorTexture=texture,
        metallicFactor=0,roughnessFactor=roughness))
    np.testing.assert_array_equal(result.triangles,mesh.triangles)
    return result


def build(parameters):
    p = json.loads(parameters.read_text()); name = p['name']; stem = name+'_m'
    root = ROOT/'output/objects/hestia-rl'/name/'codex'
    out = root/'products'; out.mkdir(parents=True,exist_ok=True)
    references = root/'intermediate/reference'
    profile = np.array(p['profile_radius_top_bottom_mm'],dtype=float)
    r,t,b = profile.T
    assert np.all(np.diff(r)>0) and r[0]==0 and np.all(t>b) and b.min()==0
    assert r[-1]*2==p['outer_diameter_mm'] and t.max()==p['overall_height_mm']
    params = dict(p,profile_csv=name+'_profile_mm.csv',rgba=p['texture']['background_rgb']+[255],
                  editable_source=str(parameters.relative_to(ROOT)))
    dims = {k:p[k] for k in ('outer_diameter_mm','overall_height_mm')}
    dims.update(dimension_sources=p['evidence'],status_label=p.get('status_label','Nominal IKEA envelope; photo-estimated profile'))
    if 'foot_ring_measurements_mm' in p:
        dims['foot_ring_measurements_mm']=p['foot_ring_measurements_mm']
    save_json(out/'parameters.json',params);save_json(out/'dimensions.json',dims)
    np.savetxt(out/params['profile_csv'],profile,delimiter=',',header='radius_mm,top_mm,bottom_mm',comments='')
    ti,_ = keep_profile(r,t,p['profile_tolerance_mm']);bi,_ = keep_profile(r,b,p['profile_tolerance_mm'])
    polygon = np.vstack([np.c_[r[bi],b[bi]],np.c_[r[ti[::-1]],t[ti[::-1]]]])
    mesh = trimesh.creation.revolve(polygon,sections=p['angular_sections'])
    if mesh.volume<0: mesh.invert()
    geometry = audit(mesh)
    np.testing.assert_allclose(mesh.extents,[p['outer_diameter_mm']]*2+[p['overall_height_mm']],atol=1e-6)
    from scipy.spatial import cKDTree
    angle=2*np.pi/p['angular_sections']
    rotated=mesh.vertices @ np.array([[np.cos(angle),np.sin(angle),0],[-np.sin(angle),np.cos(angle),0],[0,0,1]])
    symmetry_error=float(cKDTree(mesh.vertices).query(rotated)[0].max())
    assert symmetry_error<1e-6,'Angular sections are not rotationally symmetric'
    geometry.update(angular_sections=p['angular_sections'],rotation_step_degrees=360/p['angular_sections'],
                    maximum_rotational_vertex_error_mm=symmetry_error)
    if 'foot_ring_measurements_mm' in p:
        foot=p['foot_ring_measurements_mm'];inner=foot['inner_diameter']/2;outer=foot['outer_diameter']/2
        # Validate the simplified profile, not just the source parameters.
        np.testing.assert_allclose(np.interp([inner,outer],r[bi],b[bi]),
                                   [foot['central_recess_depth_estimate'],foot['height']],atol=1e-5)
        contact=r[bi][b[bi]==0]
        assert contact.min()>inner and contact.max()<outer
        assert len(mesh.faces)<=5000,'Measured plate should remain a compact mesh'
        geometry['foot_ring_constraints_verified']=True
        geometry['contact_land_inner_outer_diameter_mm']=[float(contact.min()*2),float(contact.max()*2)]
    mesh.apply_scale(.001)
    texture = photo_texture(p['texture'],references)
    visual = texture_mesh(mesh,texture,p['texture']['roughness'])
    visual.export(out/(stem+'.glb'),include_normals=True)
    parts, construction = plate_parts(out/(stem+'.glb'))
    collision_scene = trimesh.Scene()
    for part, hull in parts.items(): collision_scene.add_geometry(hull,geom_name=part,node_name=part)
    collision_scene.export(out/(stem+'_collision.glb'))
    collision = dict(construction, convex_parts=len(parts), mujoco_runtime_tested=False)
    # Verify opening over the entire inner profile, including sector boundaries.
    probes = []; solids = []
    for rr in np.linspace(0,r[-1]-2,35):
        for angle in np.linspace(0,2*np.pi,129):
            probes.append([rr*np.cos(angle),rr*np.sin(angle),np.interp(rr,r,t)+.6])
    for rr,tt,bb in zip((r[1:]+r[:-1])/2,(t[1:]+t[:-1])/2,(b[1:]+b[:-1])/2):
        a = np.pi/p['angular_sections']
        solids.append([rr*np.cos(a),rr*np.sin(a),(tt+bb)/2])
    assert not probe_collision(parts,probes).any()
    assert probe_collision(parts,solids).all()
    assert not probe_collision(parts,[[0,0,1],[30,0,1]]).any()
    collision.update(open_cavity_probe_count=len(probes),material_probe_count=len(solids),
                     underside_recess_free=True,offline_only=True)
    center,inertia = inertial_properties(parts,p['estimated_mass_kg'])
    save_json(out/'physics.json',dict(mass_kg=p['estimated_mass_kg'],center_of_mass_m=center.tolist(),
        inertia_kg_m2=inertia.tolist(),friction=p['friction'],provenance=p['evidence']['mass'],runtime_tested=False))
    # Validate the canonical visual; delivery formats are exported separately.
    format_checks = {}
    for suffix in ('glb',):
        loaded = trimesh.load(out/(stem+'.'+suffix),force='mesh');loaded.merge_vertices(merge_tex=True,merge_norm=True)
        assert loaded.is_volume and loaded.is_watertight
        np.testing.assert_allclose(loaded.bounds,mesh.bounds,atol=2e-8)
        format_checks[suffix] = {'watertight':True,'triangles':len(loaded.faces)}
    restored = trimesh.load(out/(stem+'.glb'),force='mesh',process=False)
    np.testing.assert_allclose(restored.visual.uv[restored.faces],visual.visual.uv[visual.faces],atol=1e-7)
    np.testing.assert_array_equal(np.array(restored.visual.material.baseColorTexture),np.array(texture))
    appearance = {'uv_mapping_preserved': True, 'images': 1, 'textures': 1}
    report = dict(mesh_checks=geometry,format_checks=format_checks,units='metres',
        origin='XY centre, Z up, lowest foot contact at Z=0',mujoco_runtime_tested=False,
        dimensions=dims,profile_mm={'radius':r.tolist(),'top':t.tolist(),'bottom':b.tolist()},
        appearance_check=dict(appearance,rgba=params['rgba'],pattern_provenance=p['texture']['provenance']),
        collision=collision,maximum_circle_chord_error_mm=float(r[-1]*(1-np.cos(np.pi/p['angular_sections']))))
    save_json(out/'validation.json',report)
    instructions = f"""{p['product']}
Canonical visual and separate convex collision GLBs use metres, Z up, support Z=0.
Editable source: {parameters.relative_to(ROOT)}
Mass, inertia and friction in physics.json are uncalibrated estimates.
Geometry, UVs, texture and cavity checks are offline only; no simulation was run.
Use ../asset.json with python -m src.assets export to request OBJ, STL or MuJoCo.
Old delivery files, when present, are historical snapshots and may be stale.
"""
    (out/'README.txt').write_text(instructions)
    variants={name:{'visual':out/(stem+'.glb'),
                    'collision':out/(stem+'_collision.glb'),'physics':out/'physics.json'}}
    if p.get('generate_rl_variant'):
        from src.rl_collision import coarse_plate, scene_write, surface_error
        from src.collision import validate_parts
        from src.export import sha256
        coarse,coarse_report=coarse_plate(out/(stem+'.glb'));validate_parts(coarse)
        coarse_out=out/'rl_collision';coarse_out.mkdir(exist_ok=True)
        coarse_path=coarse_out/(name+'_rl.glb');scene_write(coarse,coarse_path)
        coarse_report.update(parts=len(coarse),triangles=sum(len(m.faces) for m in coarse.values()),
            surface_check=surface_error(out/(stem+'.glb'),coarse),source_sha256=sha256(out/(stem+'.glb')),
            mujoco_runtime_tested=False)
        save_json(coarse_out/(name+'_validation.json'),coarse_report)
        variants[name+'_rl']={**variants[name],'collision':coarse_path,
                             'contact_settings':ROOT/'src/product_inputs/rl_contacts.json'}
    save_manifest(root/'asset.json',kind='radial_profile',
                  sources=[Path(__file__),Path(__file__).with_name('rebuild.py'),parameters,
                           *[references/p['texture'][side]['image'] for side in ('top','bottom')],
                           *[ROOT/path for path in p.get('geometry_evidence_files',[])]],
                  variants=variants,
                  units='m',frame='XY centre, Z up; lowest foot contact Z=0',
                  rebuild=['python','-m','src.plates.authored','--parameters',str(parameters)])
    print(json.dumps({'output':str(out),'triangles':len(mesh.faces),'convex_parts':len(parts),
                      'volume_m3':mesh.volume,'offline_checks':'passed'},indent=2))
    return out,p


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters',type=Path,required=True)
    args=parser.parse_args();output,params=build(args.parameters.resolve())
    import subprocess
    import sys
    subprocess.run([sys.executable,str(ROOT/'src/plates/render.py'),params['name']],check=True)
