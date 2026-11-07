"""Export the accepted GLB to portable URDF without altering source geometry.

Run with the project .venv. Engine validation is a separate, independent step.
The numerical dynamics settings are explicit placeholders, not measurements.
"""
from pathlib import Path
import hashlib
import json
import html
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from ..paths import ROOT
OUT = ROOT / 'output/objects/hkqai-instruments/c13534/codex/intermediate/.build_work'
PACKAGE = OUT / 'urdf'
CONFIG = OUT / 'collision_controls.json'
SOURCE = OUT / 'meshes/collision_closed.glb'


def nums(v):
    return ' '.join(f'{float(x):.12g}' for x in v)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def settings_default():
    return {
        'status': 'Numerical placeholders for import/kinematic testing; not measured physical dynamics.',
        'fixed_base_required': True,
        'mass_kg': {'base_link': 1., 'upper_door': 1., 'lower_door': 1.,
                    'upper_handle_turn': .005, 'lower_handle_turn': .005,
                    'upper_handle_pop': .05, 'lower_handle_pop': .05,
                    'mode_selector': .045, 'upper_interior': .5, 'lower_interior': .5},
        'mounting_link_mass_kg': .001,
        'selector_shaft_mass_kg': .005,
        'empty_joint_link_radius_m': .003,
        'joint_defaults': {
            'door': {'effort': 5., 'velocity': 1., 'damping': .05, 'friction': 0.},
            'handle': {'effort': .5, 'velocity': 2., 'damping': .002, 'friction': 0.},
            'selector': {'effort': 10., 'velocity': .05, 'damping': .05, 'friction': 0.}},
        'inertia_method': 'Uniform-density convex-part compound, normalized to placeholder link mass; overlaps may double-count volume. Empty turn frames use a small sphere inertia.',
        'limits_status': 'Copied preview ranges, converted degrees to radians; not measured stops.',
        'joint_effort_units': 'N m for revolute; N for prismatic. Values are numerical limits, not measured actuation forces.',
        'pybullet_collision_margin_m': 0.,
        'latch_interlock': 'Not implemented: URDF represents independent DOFs, not the lift/turn/open state machine.',
    }


def inertia(meshes, mass, radius):
    if not meshes:
        return np.zeros(3), np.eye(3)*(.4*mass*radius**2)
    volumes = np.array([m.volume for m in meshes])
    assert (volumes > 0).all()
    masses = mass*volumes/volumes.sum()
    center = np.sum([w*m.center_mass for w,m in zip(masses,meshes)],axis=0)/mass
    tensor = np.zeros((3,3))
    for weight,m in zip(masses,meshes):
        d = m.center_mass-center
        tensor += m.moment_inertia*(weight/m.volume) + weight*(d@d*np.eye(3)-np.outer(d,d))
    return center, tensor


def add_inertial(link, center, mass, tensor):
    elem = ET.SubElement(link, 'inertial')
    ET.SubElement(elem, 'origin', xyz=nums(center), rpy='0 0 0')
    ET.SubElement(elem, 'mass', value=f'{mass:.12g}')
    ET.SubElement(elem, 'inertia', **{key:f'{tensor[i,j]:.12g}' for key,i,j in
        [('ixx',0,0),('ixy',0,1),('ixz',0,2),('iyy',1,1),('iyz',1,2),('izz',2,2)]})


def generate():
    PACKAGE.mkdir(exist_ok=True)
    (PACKAGE/'meshes').mkdir(exist_ok=True)
    (PACKAGE/'load_pybullet.py').write_bytes(Path(__file__).with_name('urdf_pybullet_loader.py').read_bytes())
    cfg = json.loads(CONFIG.read_text())
    settings_file = PACKAGE/'export_settings.json'
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else settings_default()
    save_json(settings_file, settings)
    scene = trimesh.load(SOURCE, force='scene')
    links = {}
    original_name = lambda name: 'base_link' if name=='fixed' else name
    for name,data in cfg['links'].items():
        target = original_name(name)
        links[target] = dict(data)
        links[target]['parent'] = None if name=='fixed' else original_name(data['parent'])
        links[target]['source_group'] = name
    parts = {}
    meshes = {}
    hardware = []
    for name,info in cfg['parts'].items():
        group = info['group']
        transform,geom = scene.graph.get(frame_to=name, frame_from=group)
        mesh = scene.geometry[geom].copy();mesh.apply_transform(transform)
        assert mesh.is_convex and mesh.is_volume
        # Keep mounting hardware on fixed child links, allowing narrow simulator
        # self-contact filters while retaining its collision with external robots.
        split = info['hinge_hardware'] or name=='selector_shaft'
        parent = original_name(group)
        link_name = name+'_link' if split else parent
        if split:
            links[link_name] = {'parent': parent, 'origin_m': [0,0,0], 'type': 'fixed',
                               'source_group': group, 'mounting_part': name}
        if info['hinge_hardware']:
            hardware.append(link_name)
        color = np.asarray(mesh.visual.vertex_colors[:,:4],dtype=float).mean(0)/255
        filename = 'meshes/'+name+'.obj'
        (PACKAGE/filename).write_text(trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_color=False, include_texture=False))
        meshes[name] = mesh
        parts[name] = {'link':link_name,'source_group':group,'mesh':filename,
                       'triangles':len(mesh.faces),'color_rgba':color.tolist()}
    robot = ET.Element('robot', name='hamamatsu_c13534')
    robot.append(ET.Comment(' Metres, kilograms, radians. Import with a fixed base. '))
    robot.append(ET.Comment(' Inertias, masses, force/velocity limits and damping are provisional. See export_settings.json. '))
    robot.append(ET.Comment(' Apply collision_filters.json to instrument self-contact only; preserve external robot contact. Do not merge fixed mounting links before applying filters. '))
    inertial_values = {}
    for name,info in links.items():
        link = ET.SubElement(robot, 'link', name=name)
        names = [n for n,d in parts.items() if d['link']==name]
        mass = settings['mass_kg'].get(name,settings['selector_shaft_mass_kg'] if name=='selector_shaft_link' else settings['mounting_link_mass_kg'])
        center,tensor = inertia([meshes[n] for n in names], mass, settings['empty_joint_link_radius_m'])
        add_inertial(link,center,mass,tensor)
        inertial_values[name] = {'mass_kg':mass,'com_link_m':center.tolist(),'inertia_kg_m2':tensor.tolist(),'measured':False}
        for part in names:
            for kind in ['visual','collision']:
                element = ET.SubElement(link,kind,name=part)
                ET.SubElement(element,'origin',xyz='0 0 0',rpy='0 0 0')
                geometry = ET.SubElement(element,'geometry')
                ET.SubElement(geometry,'mesh',filename=parts[part]['mesh'],scale='1 1 1')
                if kind=='visual':
                    material = ET.SubElement(element,'material',name=part+'_material')
                    ET.SubElement(material,'color',rgba=nums(parts[part]['color_rgba']))
    joint_map = {}
    for name,info in links.items():
        if info['parent'] is None:
            continue
        joint_name = name+'_joint'
        joint = ET.SubElement(robot,'joint',name=joint_name,type=info['type'])
        ET.SubElement(joint,'parent',link=info['parent'])
        ET.SubElement(joint,'child',link=name)
        ET.SubElement(joint,'origin',xyz=nums(info['origin_m']),rpy='0 0 0')
        if info['type']!='fixed':
            ET.SubElement(joint,'axis',xyz=nums(info['axis']))
            limits = np.array(info['preview_range'],dtype=float)
            if info['position_units']=='degrees':limits=np.deg2rad(limits)
            category = 'selector' if info['type']=='prismatic' else 'door' if name.endswith('_door') else 'handle'
            dynamics = settings['joint_defaults'][category]
            ET.SubElement(joint,'limit',lower=f'{limits[0]:.12g}',upper=f'{limits[1]:.12g}',
                          effort=str(dynamics['effort']),velocity=str(dynamics['velocity']))
            ET.SubElement(joint,'dynamics',damping=str(dynamics['damping']),friction=str(dynamics['friction']))
            joint_map[name] = {'urdf_joint':joint_name,'child_link':name,'axis':info['axis'],
                               'type':info['type'],'lower':float(limits[0]),'upper':float(limits[1]),
                               'units':'radians' if info['type']=='revolute' else 'metres','limits_measured':False}
    ET.indent(robot,space='  ')
    ET.ElementTree(robot).write(PACKAGE/'robot.urdf',encoding='utf-8',xml_declaration=True)
    # Parts in these three links belong to the same rigid instrument body.
    fixed_group = ['base_link','upper_interior','lower_interior']
    filters = {'schema':'link-pair self-collision exclusions; portable URDF cannot enforce these',
        'self_excluded_hardware_links':hardware,
        'explicit_disabled_pairs':[
            {'links':['selector_shaft_link','base_link'],'reason':'Shaft passes into the unmodeled panel bearing.'},
            {'links':['selector_shaft_link','mode_selector'],'reason':'Rigidly attached shaft/knob shapes overlap at their mounting.'}]+
            [{'links':[a,b],'reason':'Same rigid fixed instrument assembly.'} for i,a in enumerate(fixed_group) for b in fixed_group[i+1:]],
        'hardware_policy':'Exclude hinge hardware against this instrument only, matching prior model checks. It remains collidable with external bodies.',
        'do_not_merge_fixed_links':True,'external_robot_collisions_preserved':True}
    save_json(PACKAGE/'collision_filters.json',filters)
    poses={}
    for pose,state in cfg['pose_snapshots'].items():
        poses[pose]={joint_map[n]['urdf_joint']:float(np.deg2rad(v) if joint_map[n]['units']=='radians' else v) for n,v in state.items()}
    save_json(PACKAGE/'joint_states.json',{'units':'radians for revolute, metres for prismatic','poses':poses})
    manifest = {'urdf':'robot.urdf','root_link':'base_link','fixed_base_required':True,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,SOURCE]},
        'parts':parts,'links':links,'moving_joints':joint_map,'inertials':inertial_values,
        'geometry_policy':f'All {len(parts)} source parts preserved as separate visual/collision meshes; no convexification across parts.',
        'latch_behavior':'Independent joints. Lift-turn-open ordering is not mechanically enforced.',
        'internal_motion':'Upper and lower internal assemblies remain fixed.',
        'references':['https://github.com/ros/urdf_tutorial/blob/ros2/urdf/07-physics.urdf',
                      'https://mujoco.readthedocs.io/en/stable/modeling.html#urdf-extensions']}
    save_json(PACKAGE/'manifest.json',manifest)
    return manifest


def xml_fk(robot, positions):
    transforms={'base_link':np.eye(4)}
    remaining=list(robot.findall('joint'))
    while remaining:
        available=[j for j in remaining if j.find('parent').get('link') in transforms]
        assert available,'Joint cycle or disconnected tree'
        for joint in available:
            parent,child=joint.find('parent').get('link'),joint.find('child').get('link')
            origin=joint.find('origin')
            xyz=np.fromstring(origin.get('xyz'),sep=' ');rpy=np.fromstring(origin.get('rpy'),sep=' ')
            t=trimesh.transformations.euler_matrix(*rpy);t[:3,3]=xyz
            if joint.get('type')!='fixed':
                axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
                value=positions.get(joint.get('name'),0.)
                if joint.get('type')=='revolute':t=t@trimesh.transformations.rotation_matrix(value,axis)
                else:
                    move=np.eye(4);move[:3,3]=axis*value;t=t@move
            transforms[child]=transforms[parent]@t
            remaining.remove(joint)
    return transforms


def validate(manifest):
    robot=ET.parse(PACKAGE/'robot.urdf').getroot()
    assert robot.tag=='robot'
    links=robot.findall('link');joints=robot.findall('joint')
    assert len({l.get('name') for l in links})==len(links)
    assert len({j.get('name') for j in joints})==len(joints)
    children={j.find('child').get('link') for j in joints}
    assert {l.get('name') for l in links}-children=={'base_link'}
    assert len(joints)==len(links)-1
    for link in links:
        inertial=link.find('inertial');i=inertial.find('inertia')
        tensor=np.array([[float(i.get('ixx')),float(i.get('ixy')),float(i.get('ixz'))],
                         [float(i.get('ixy')),float(i.get('iyy')),float(i.get('iyz'))],
                         [float(i.get('ixz')),float(i.get('iyz')),float(i.get('izz'))]])
        eigen=np.linalg.eigvalsh(tensor)
        assert float(inertial.find('mass').get('value'))>0
        assert eigen.min()>0 and eigen[-1]<=eigen[:2].sum()+1e-10,link.get('name')
    for joint in joints:
        if joint.get('type')=='fixed':continue
        limits=joint.find('limit');axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
        assert np.isclose(np.linalg.norm(axis),1)
        assert float(limits.get('upper'))>float(limits.get('lower'))
        assert float(limits.get('effort'))>0 and float(limits.get('velocity'))>0
    loaded={}
    for link in links:
        for collision in link.findall('collision'):
            part=collision.get('name');filename=collision.find('geometry/mesh').get('filename')
            q=trimesh.load(PACKAGE/filename,force='mesh',process=False)
            assert q.is_volume and q.is_convex,part
            loaded[part]=q
    assert len(loaded)==len(manifest['parts'])
    poses=json.loads((PACKAGE/'joint_states.json').read_text())['poses']
    errors={}
    for name,state in poses.items():
        transforms=xml_fk(robot,state)
        reference=trimesh.load(OUT/f'meshes/collision_{name}.glb',force='scene')
        worst=0.
        for part,mesh in loaded.items():
            t,g=reference.graph[part]
            target=trimesh.transform_points(reference.geometry[g].vertices,t)
            actual=trimesh.transform_points(mesh.vertices,transforms[manifest['parts'][part]['link']])
            assert actual.shape==target.shape
            worst=max(worst,float(np.max(np.abs(actual-target))))
        assert worst<1e-7,(name,worst)
        errors[name]=worst
    for path,sha in manifest['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
    report={'urdf_links':len(links),'fixed_joints':sum(j.get('type')=='fixed' for j in joints),
        'moving_joints':sum(j.get('type')!='fixed' for j in joints),'visual_meshes':len(robot.findall('link/visual')),
        'collision_meshes':len(loaded),'triangles':sum(len(m.faces) for m in loaded.values()),
        'tree_valid':True,'inertias_positive_and_triangle_inequality_valid':True,
        'inertias_and_limits_physically_measured':False,'all_collision_parts_convex_closed':True,
        'fk_vs_original_glb_max_error_m_by_pose':errors,'source_files_unchanged':True,
        'review_findings':[
            'Both handle turn axes are local -Y, preserving the user-approved direction.',
            'GLB angles are degrees; URDF and joint_states.json use radians. Selector remains metres.',
            'Hardware and selector shaft are fixed child links so mounting self-contact exclusions do not disable door/body or external robot contact.',
            'Import with fixed base and keep mounting fixed links separate until self-contact filters are applied.',
            'Masses/inertias/effort/velocity/damping are placeholders; no physical dynamics or grasp-force accuracy claimed.',
            'Door latching is not enforced. A controller/state machine or explicit latch geometry is needed for that task.',
            'Mechanical stops, exact handle clearance and selector travel remain unmeasured.',
            'Internal assemblies remain fixed and approximate; sample-insertion contact accuracy is unverified.'],
        'engine_validation':{'tested':False}}
    save_json(PACKAGE/'review.json',report)
    return report
