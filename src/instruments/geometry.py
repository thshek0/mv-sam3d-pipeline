"""Instrument geometry: reusable implementation behind the instruments CLI."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault('MPLCONFIGDIR', '/tmp/instrument_model_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image, ImageOps

from .paths import ROOT as REPO
DEFAULT_OUT = REPO / "output"
from .render import Renderer

WHITE = [224, 224, 215, 255]
DARK = [55, 62, 70, 255]
BLACK = [30, 34, 39, 255]
METAL = [159, 172, 183, 255]
GREEN = [62, 118, 104, 255]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()






class Model:
    def __init__(self,p):
        self.p=p;self.parts={};self.groups={};self.evidence={}

    def add(self,name,mesh,group='base_link',color=WHITE,evidence='Image/model estimate'):
        assert mesh.is_volume and mesh.is_convex and np.isfinite(mesh.vertices).all(),name
        mesh.visual.face_colors=color
        self.parts[name]=mesh;self.groups[name]=group;self.evidence[name]=evidence

    def box(self,name,lo,hi,**kw):
        lo,hi=np.array(lo,float),np.array(hi,float)
        assert (hi>lo).all(),name
        mesh=trimesh.creation.box(hi-lo);mesh.apply_translation((lo+hi)/2)
        self.add(name,mesh,**kw)

    def yz(self,name,poly,x0,x1,**kw):
        vertices=[[x,y,z] for x in [x0,x1] for y,z in poly]
        self.add(name,trimesh.convex.convex_hull(vertices),**kw)

    def rounded(self,name,lo,hi,r=.016,**kw):
        # Convex round/chamfer front and back in YZ; simple extrusion in X.
        lo,hi=np.array(lo),np.array(hi);r=min(r,(hi[1]-lo[1])*.24,(hi[2]-lo[2])*.24)
        poly=[]
        for cy,cz,start in [(lo[1]+r,lo[2]+r,180),(hi[1]-r,lo[2]+r,270),
                            (hi[1]-r,hi[2]-r,0),(lo[1]+r,hi[2]-r,90)]:
            for deg in np.linspace(start,start+90,4):
                poly.append([cy+r*np.cos(np.deg2rad(deg)),cz+r*np.sin(np.deg2rad(deg))])
        self.yz(name,poly,lo[0],hi[0],**kw)

    def transform(self,value):
        pivot=np.array(self.p['joint_origin_m']);axis=np.array(self.p['joint_axis'])
        if self.p['joint_type']=='prismatic':
            T=np.eye(4);T[:3,3]=axis*value;return T
        return trimesh.transformations.rotation_matrix(value,axis,point=pivot)

    def scene(self,value=0):
        scene=trimesh.Scene(base_frame='world')
        scene.graph.update(frame_to='base_link',frame_from='world',matrix=np.eye(4))
        pivot=np.array(self.p['joint_origin_m']);T=self.transform(value)
        origin=np.eye(4);origin[:3,3]=pivot
        scene.graph.update(frame_to='cover',frame_from='base_link',matrix=T@origin)
        for name,mesh in self.parts.items():
            q=mesh.copy();group=self.groups[name]
            if group=='cover':q.apply_translation(-pivot)
            scene.add_geometry(q,node_name=name,geom_name=name,parent_node_name=group)
        scene.metadata={'units':'metres','prototype_status':self.p['status'],'joint_position':value,
                        'physical_contact_accuracy_validated':False,'body_dimension_evidence':self.p['body_dimension_evidence']}
        return scene

    def world_parts(self,value=0):
        result={}
        for n,mesh in self.parts.items():
            q=mesh.copy()
            if self.groups[n]=='cover':q.apply_transform(self.transform(value))
            result[n]=q
        return result


def open_value(p):
    if p['model']=='cary60':return p['lid_preview_stroke_m']
    if p['model']=='c16361':return np.deg2rad(p['lid_preview_angle_deg'])
    return p['body_wdh_m'][0]*p['cover_travel_fraction']


def urdf(m,out):
    package=out/'urdf';(package/'meshes').mkdir(parents=True,exist_ok=True)
    expected={name+'.obj' for name in m.parts}
    for obsolete in (package/'meshes').glob('*.obj'):
        if obsolete.name not in expected:
            obsolete.unlink()  # Unreferenced files from this builder's previous draft.
    p=m.p;robot=ET.Element('robot',name=p['model']+'_prototype')
    for group in ['base_link','cover']:
        link=ET.SubElement(robot,'link',name=group)
        inertial=ET.SubElement(link,'inertial');ET.SubElement(inertial,'mass',value='1')
        ET.SubElement(inertial,'inertia',ixx='.01',iyy='.01',izz='.01',ixy='0',ixz='0',iyz='0')
        for name,mesh in m.parts.items():
            if m.groups[name]!=group:continue
            q=mesh.copy()
            if group=='cover':q.apply_translation(-np.array(p['joint_origin_m']))
            (package/'meshes'/f'{name}.obj').write_text(trimesh.exchange.obj.export_obj(q,include_color=False,include_normals=False,include_texture=False))
            for tag in ['visual','collision']:
                item=ET.SubElement(link,tag,name=name)
                geo=ET.SubElement(item,'geometry');ET.SubElement(geo,'mesh',filename=f'meshes/{name}.obj',scale='1 1 1')
                if tag=='visual':
                    material=ET.SubElement(item,'material',name=name+'_color')
                    ET.SubElement(material,'color',rgba=' '.join(str(v/255) for v in mesh.visual.face_colors[0]))
    joint=ET.SubElement(robot,'joint',name='cover_joint',type=p['joint_type'])
    ET.SubElement(joint,'parent',link='base_link');ET.SubElement(joint,'child',link='cover')
    ET.SubElement(joint,'origin',xyz=' '.join(map(str,p['joint_origin_m'])),rpy='0 0 0')
    ET.SubElement(joint,'axis',xyz=' '.join(map(str,p['joint_axis'])))
    ET.SubElement(joint,'limit',lower='0',upper=str(open_value(p)),effort='1',velocity='.2')
    ET.SubElement(joint,'dynamics',damping='.05',friction='0')
    ET.indent(robot)
    path=package/'robot.urdf';ET.ElementTree(robot).write(path,encoding='utf-8',xml_declaration=True)
    (package/'README.txt').write_text('Fixed-base review prototype. Units: metres, kilograms, radians.\n'
        'Mass, inertia, effort, damping and travel limits are numerical placeholders. No latch/interlock.\n'
        'Keep the separate convex collision shapes; do not convexify the whole hollow object.\n'
        'Use URDF_INITIALIZE_SAT_FEATURES in PyBullet. Read review.json for tested contacts and limitations.\n'
        'Dimension evidence: '+p['body_dimension_evidence']+'\n'+p['joint_evidence']+'\n')
    return path


def validate(m,out,path):
    from .physics import pybullet
    bullet=pybullet()
    review={'parts':len(m.parts),'triangles':sum(len(q.faces) for q in m.parts.values()),
            'all_parts_convex_watertight':True,'body_target_mm':(np.array(m.p['body_wdh_m'])*1000).tolist(),
            'closed_bounds_mm':(m.scene().bounds*1000).tolist(),
            'physical_contact_accuracy_validated':False,'engine_tested':False}
    # Source GLB hierarchy and URDF OBJ coordinates must reproduce the same moving pose.
    import xml.etree.ElementTree as ET
    root=ET.parse(path).getroot();maxerr=0.
    joint=root.find('joint')
    axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
    joint_origin=np.eye(4);joint_origin[:3,3]=np.fromstring(joint.find('origin').get('xyz'),sep=' ')
    def xml_transform(value):
        motion=np.eye(4)
        if joint.get('type')=='prismatic':motion[:3,3]=axis*value
        else:motion=trimesh.transformations.rotation_matrix(value,axis)
        return joint_origin@motion
    for value in [0,open_value(m.p)/2,open_value(m.p)]:
        for link in root.findall('link'):
            for visual in link.findall('visual'):
                name=visual.get('name');q=trimesh.load(path.parent/visual.find('geometry/mesh').get('filename'),process=False,force='mesh')
                expected=m.parts[name].vertices
                if link.get('name')=='cover':
                    q.apply_transform(xml_transform(value))
                    expected=trimesh.transform_points(expected,m.transform(value))
                err=float(np.max(abs(q.vertices-expected)));maxerr=max(maxerr,err)
                assert err<1e-7,(name,err)
    review['obj_vs_model_max_error_m']=maxerr
    glb_errors={}
    for label,value in [('closed',0),('open',open_value(m.p))]:
        scene=trimesh.load(out/'meshes'/f'collision_{label}.glb',force='scene')
        errors=[]
        for name,q in m.world_parts(value).items():
            transform,geometry=scene.graph[name]
            actual=trimesh.transform_points(scene.geometry[geometry].vertices,transform)
            errors.append(float(np.max(abs(actual-q.vertices))))
        glb_errors[label]=max(errors)
        assert max(errors)<1e-7
    review['glb_roundtrip_max_error_m']=glb_errors
    if m.p['model']=='c16361':
        body=trimesh.util.concatenate([m.parts[n] for n in ['rear_housing','front_plinth','left_front_housing','right_front_housing']])
    else:body=trimesh.util.concatenate(list(m.parts.values()))
    extent=np.ptp(body.bounds,axis=0)
    np.testing.assert_allclose(extent,m.p['body_wdh_m'],atol=1e-8)
    review['body_extent_measured_mm']=(extent*1000).tolist()
    review['body_target_numeric_check_pass']=True
    if bullet:
        c=bullet.connect(bullet.DIRECT)
        flags=bullet.URDF_USE_SELF_COLLISION|bullet.URDF_USE_SELF_COLLISION_INCLUDE_PARENT|bullet.URDF_INITIALIZE_SAT_FEATURES
        body=bullet.loadURDF(str(path),useFixedBase=True,flags=flags,physicsClientId=c)
        failures=[];worst=0.;fk_error=0.;cached_worst=0.
        for value in np.linspace(0,open_value(m.p),21):
            bullet.resetJointState(body,0,float(value),physicsClientId=c)
            state=bullet.getLinkState(body,0,computeForwardKinematics=True,physicsClientId=c)
            actual=np.eye(4);actual[:3,:3]=np.array(bullet.getMatrixFromQuaternion(state[5])).reshape(3,3);actual[:3,3]=state[4]
            fk_error=max(fk_error,float(np.max(abs(actual-xml_transform(value)))))
            bullet.performCollisionDetection(physicsClientId=c)
            # resetJointState teleports between samples. Persistent manifolds can
            # retain invalid tangential contacts on touching coplanar faces.
            # Query the actual base/cover pair afresh at every sampled pose.
            for cp in bullet.getContactPoints(body,body,physicsClientId=c):cached_worst=max(cached_worst,-cp[8])
            for cp in bullet.getClosestPoints(body,body,0,linkIndexA=-1,linkIndexB=0,physicsClientId=c):
                worst=max(worst,-cp[8])
                if cp[8]<-.0005:failures.append({'joint_position':float(value),'penetration_mm':-cp[8]*1000})
        W,D,H=m.p['body_wdh_m']
        xy=m.p.get('opening_probe_xy_m',{'cary60':[.12,.15],'c16361':[.145,.15],'c13472':[W*.20,D*.40]}[m.p['model']])
        hits={}
        for label,value in [('closed',0),('open',open_value(m.p))]:
            bullet.resetJointState(body,0,value,physicsClientId=c)
            hit=bullet.rayTest([*xy,H+.2],[*xy,0],physicsClientId=c)[0]
            assert hit[0]==body
            hits[label]=hit[3][2]
        assert hits['closed']-hits['open']>.05,hits
        assert fk_error<2e-6,fk_error
        bullet.disconnect(c)
        review.update(engine_tested=True,engine='PyBullet',sampled_states=21,
            maximum_self_penetration_mm=worst*1000,penetrations_over_0_5_mm=failures,
            contact_query='Fresh base/cover closest-points query at each sampled pose',
            persistent_manifold_maximum_penetration_mm=cached_worst*1000,
            xml_vs_engine_max_matrix_error=fk_error,opening_ray_hit_heights_m=hits,
            filters='No self-contact exclusions; fixed parts share the base link.',
            simulator_dynamics='Not calibrated; no physical force/clearance tolerance established.')
        assert not failures,failures[:5]
    write_json(out/'review.json',review)
    write_json(path.parent/'review.json',review)
    return review


def render_views(m,out):
    dest=out/'dimensions';dest.mkdir(exist_ok=True)
    r=Renderer(m.world_parts());p=m.p;W,D,H=np.array(p['body_wdh_m'])*1000
    views=[('front',(1,0,0),(0,0,1),(0,-1,0),W,H),
           ('left_side',(0,-1,0),(0,0,1),(-1,0,0),D,H),
           ('right_side',(0,1,0),(0,0,1),(1,0,0),D,H),
           ('top',(1,0,0),(0,1,0),(0,0,1),W,D),
           ('rear',(-1,0,0),(0,0,1),(0,1,0),W,H)]
    for name,u,v,cam,span_x,span_y in views:
        vertices=np.vstack([q.vertices for q in m.parts.values()])*1000
        xx,yy=vertices@u,vertices@v
        bounds=[float(xx.min()),float(xx.max()),float(yy.min()),float(yy.max())]
        x0,x1,y0,y1=bounds;pad=45
        extent=[x0-5,x1+5,y0-5,y1+5]
        fig,ax=plt.subplots(figsize=(9,8));fig.subplots_adjust(left=.12,right=.94,top=.86,bottom=.20)
        ax.imshow(r.image(u,v,cam,extent,width=1300),extent=extent)
        ax.set(xlim=(x0-pad*1.4,x1+pad*.6),ylim=(y0-pad*1.6,y1+pad*.5),aspect='equal');ax.axis('off')
        color={'M':'#1967b4','U':'#087e62','E':'#a85e10'}[p['dimension_code']]
        # Generated extent is explicitly distinguished from the published body datum.
        ax.annotate('',(x0,y0-22),(x1,y0-22),arrowprops=dict(arrowstyle='<->',color=color))
        ax.text((x0+x1)/2,y0-18,f'{x1-x0:.1f} mm generated extent',ha='center',color=color,fontsize=11)
        ax.annotate('',(x0-25,y0),(x0-25,y1),arrowprops=dict(arrowstyle='<->',color=color))
        ax.text(x0-32,(y0+y1)/2,f'{y1-y0:.1f} mm',ha='center',va='center',rotation=90,color=color,fontsize=11)
        fig.suptitle(p['model'].upper()+' / '+name.replace('_',' ').title(),fontsize=19,weight='bold')
        fig.text(.08,.14,f"Body target: {W:g} × {D:g} × {H:g} mm [{p['dimension_code']}]",fontsize=12,color=color)
        evidence_label={'M':'Published body dimensions; local shapes estimated.',
                        'U':'Approximate user overall dimensions; local shapes estimated.',
                        'E':'ALL dimensions provisional; no verified overall size.'}[p['dimension_code']]
        fig.text(.08,.10,'Actual geometry; closed pose. '+evidence_label,fontsize=10)
        fig.text(.08,.065,'C16361 extent includes the selected sample box and 15.4 mm front projection.' if p['model']=='c16361' else 'Openings, handle details and travel require task-specific measurement.',fontsize=10)
        fig.savefig(dest/(name+'.png'),dpi=150);plt.close(fig)
    # Perspective-free oblique assembly previews and a motion demonstration.
    cam=np.array([1.,-1.6,1.15]);cam/=np.linalg.norm(cam)
    u=np.cross([0,0,1],cam);u/=np.linalg.norm(u);v=np.cross(cam,u)
    all_v=np.vstack([np.vstack([q.vertices for q in m.world_parts(s).values()]) for s in [0,open_value(p)]])*1000
    xx,yy=all_v@u,all_v@v;extent=[xx.min()-25,xx.max()+25,yy.min()-25,yy.max()+25]
    frames=[]
    for val in np.linspace(0,open_value(p),11):
        rgba=Renderer(m.world_parts(val)).image(u,v,cam,extent,width=700)
        im=Image.new('RGBA',(rgba.shape[1],rgba.shape[0]),'white');im.alpha_composite(Image.fromarray(rgba));frames.append(im.convert('RGB'))
    frames[0].save(out/'closed.png');frames[-1].save(out/'open.png')
    frames[0].save(out/'motion.gif',save_all=True,append_images=frames[1:]+frames[-2:0:-1],duration=110,loop=0)
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    for ax,im,title in zip(axes,[frames[0],frames[-1]],['Closed prototype','Open demonstration (travel estimated)']):
        ax.imshow(im);ax.axis('off');ax.set_title(title)
    fig.suptitle(p['model'].upper()+' / geometry review',fontsize=19)
    fig.text(.04,.04,'Dimensions: '+p['body_dimension_evidence']+'. Geometry shown is the exported collision model.',fontsize=11)
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(out/'preview.png',dpi=150);plt.close(fig)
