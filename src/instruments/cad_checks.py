"""Engine-level CSG checks: motion, real openings, attachment and parameter edits."""
import json
import subprocess
from .paths import ROOT
from .storage.diagnostics import workspace, command_log
from pathlib import Path
import numpy as np
import trimesh


def check(dest, executable, model):
    values=json.loads((dest/'parameters_native.json').read_text())['values']
    results={}; meshes={}
    with workspace('native_csg') as tmp:
        def render(name, **definitions):
            path=Path(tmp)/(name+'.stl')
            cmd=[executable,'-o',str(path)]
            for key,value in definitions.items():
                cmd+=['-D',f'{key}={json.dumps(value)}']
            cmd.append(str((dest/'assembly.scad').resolve()))
            run=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
            command_log(path,cmd,run)
            if run.returncode or not path.exists() or 'ERROR:' in run.stderr:
                raise RuntimeError(f'{model}/{name}: {run.stderr}')
            mesh=trimesh.load(path,force='mesh')
            if not mesh.is_watertight or not mesh.is_volume:
                incidence=np.bincount(mesh.edges_unique_inverse)
                bad=mesh.vertices[mesh.edges_unique[incidence!=2]].tolist()
                raise AssertionError((model,name,'Invalid CSG solid',bad[:12],run.stderr))
            meshes[name]=mesh
            results[name]={'definitions':definitions,'bounds_mm':mesh.bounds.tolist(),
                           'watertight':True,'positive_volume':True,'triangles':len(mesh.faces)}
            return mesh

        states=['closed','open'] if model!='c13534' else ['closed','upper_open','lower_open','open','handles_released']
        # Smooth handles differ slightly from their coarse icosphere simulation proxies.
        tolerance=1.0 if model=='c13534' else .03
        for pose in states:
            mesh=render(pose,pose=pose)
            old=trimesh.load(dest.parent/'meshes'/f'collision_{pose}.glb',force='scene')
            error=float(np.max(abs(mesh.bounds-old.bounds*1000)))
            results[pose]['envelope_error_mm']=error
            np.testing.assert_allclose(mesh.bounds,old.bounds*1000,atol=tolerance,err_msg=f'{model}/{pose}')
            mesh.export(dest/f'csg_{pose}.stl')

        if model=='cary60':
            a=render('cover_closed',part='cover')
            b=render('cover_open',part='cover',pose='open')
            np.testing.assert_allclose(b.bounds-a.bounds,[[0,values['cover_travel_mm'],0]]*2,atol=.02)
            assert b.bounds[1,1]<=values['body_depth_mm']
            b=render('shorter_travel',part='cover',pose='open',cover_travel_mm=200)
            np.testing.assert_allclose(b.bounds-a.bounds,[[0,200,0]]*2,atol=.02)
            render('interior',part='holder')
            for name,w,d in [('socket',14,14),('edited_detail',16,18)]:
                m=render(name,part='socket',socket_inner_width_mm=w,socket_inner_depth_mm=d)
                v=values
                expected=(v['socket_outer_width_mm']*v['socket_outer_depth_mm']-w*d)*v['socket_height_mm']
                np.testing.assert_allclose(m.volume,expected,rtol=2e-5)
                np.testing.assert_allclose(m.extents,[v['socket_outer_width_mm'],v['socket_outer_depth_mm'],v['socket_height_mm']],atol=.02)
                results[name]['true_socket_opening_verified']=True
            m=render('deeper_body',body_depth_mm=values['body_depth_mm']+30)
            np.testing.assert_allclose(m.extents,[values['body_width_mm'],values['body_depth_mm']+30,values['body_height_mm']],atol=.03)
            labels=[('closed','Closed'),('open',f"Open: cover slides +Y / {values['cover_travel_mm']:g} mm"),('interior','Holder and open socket'),('edited_detail','Socket edit: clear 16 × 18 mm')]
            shared_report=check_cary_shared(dest,render,values)
        elif model=='c13472':
            a=render('cover_closed',part='cover'); b=render('cover_open',part='cover',pose='open')
            travel=values['body_width_mm']*values['cover_travel_fraction']
            np.testing.assert_allclose(b.bounds-a.bounds,[[travel,0,0]]*2,atol=.02)
            for name,width in [('wall_attachment',values['body_width_mm']),('wider_wall_attachment',values['body_width_mm']+30)]:
                m=render(name,part='side_unit',body_width_mm=width)
                np.testing.assert_allclose(m.bounds[1,0],width*values['inner_wall_fraction'],atol=.02)
                assert len(m.split())==1,'Side assembly must join the wall mount as one solid'
                results[name]['wall_attachment_verified']=True
            render('interior',part='interior')
            for name,diam in [('holder',values['holder_hole_diameter_mm']),('edited_detail',50)]:
                m=render(name,part='holder',holder_hole_diameter_mm=diam)
                center=np.array([values['body_width_mm']*values['stage_center_x_fraction'],values['body_depth_mm']*values['stage_center_y_fraction']])
                radii=np.linalg.norm(m.vertices[:,:2]-center,axis=1)
                ring=m.vertices[np.abs(radii-diam/2)<.02]
                assert len(ring)>=64
                # Boolean triangulation adds nonuniform seam vertices; their
                # arithmetic mean is not the centre of the circular opening.
                np.testing.assert_allclose((ring[:,:2].min(0)+ring[:,:2].max(0))/2,center,atol=.02)
                np.testing.assert_allclose(np.ptp(ring[:,:2],axis=0),[diam,diam],atol=.02)
                assert radii.min()>diam/2-.02
                results[name]['centered_open_hole_verified']=True
            m=render('wider_body',body_width_mm=values['body_width_mm']+30)
            np.testing.assert_allclose(m.extents,[values['body_width_mm']+30,values['body_depth_mm'],values['body_height_mm']],atol=.03)
            labels=[('closed','Closed'),('open','Open: cover slides +X'),('interior','Stage and attached side mechanism'),('edited_detail','Holder edit: hole Ø50 mm, centered')]
        else:
            render('interior',part='interior')
            for name,w,diam in [('lower_floor',values['lower_clear_width_mm'],values['lower_recess_diameter_mm']),('edited_detail',145,52)]:
                m=render(name,part='lower_floor',lower_clear_width_mm=w,lower_recess_diameter_mm=diam)
                height=values['lower_floor_z_mm']-values['lower_recess_bottom_z_mm']
                depth=values['lower_floor_rear_y_mm']-40
                area=64/2*(diam/2)**2*np.sin(2*np.pi/64)
                cap_area=64/2*(diam/2+1)**2*np.sin(2*np.pi/64)
                np.testing.assert_allclose(m.extents,[w,depth,height+2],atol=.02)
                np.testing.assert_allclose(m.volume,(w*depth-area)*height+cap_area*2,rtol=3e-5)
                np.testing.assert_allclose(m.bounds[:,0].mean(),values['door_width_mm']/2,atol=.02)
                results[name]['centered_recess_volume_verified']=True
            for name,spacing in [('pins',values['lower_pin_spacing_mm']),('wider_pin_spacing',50)]:
                m=render(name,part='lower_pins',lower_pin_spacing_mm=spacing)
                components=sorted(m.split(),key=lambda x:x.bounds[0,0]); assert len(components)==2
                centers=np.array([q.bounds.mean(0) for q in components])
                np.testing.assert_allclose(centers[:,0],[values['door_width_mm']/2-spacing/2,values['door_width_mm']/2+spacing/2],atol=.02)
                np.testing.assert_allclose([q.extents[:2] for q in components],[[values['small_pin_diameter_mm']]*2,[values['second_pin_diameter_mm']]*2],atol=.02)
                results[name]['paired_pins_verified']=True
            for name,width in [('body',values['door_width_mm']),('wider_door',values['door_width_mm']+10)]:
                render(name,part='body',door_width_mm=width)
                # The strip starts at the same derived door endpoint; check SCAD geometry by slices.
                door=render(name+'_door',part='upper_door',door_width_mm=width)
                section=door.section(plane_normal=[0,0,1],plane_origin=[0,0,values['body_height_mm']-1])
                assert section is not None
                np.testing.assert_allclose(section.vertices[:,0].max(),width,atol=.02)
                body=meshes[name]; cut=body.section(plane_normal=[0,0,1],plane_origin=[0,0,values['body_height_mm']-1])
                at_seam=cut.vertices[np.abs(cut.vertices[:,0]-width)<.02]
                np.testing.assert_allclose(at_seam[:,1].min(),values['front_y_mm_6'],atol=.02)
                results[name]['flush_strip_seam_verified']=True
            a=render('selector_in',part='selector',pose='closed')
            b=render('selector_out',part='selector',pose='open')
            np.testing.assert_allclose(b.bounds[0,1]-a.bounds[0,1],-values['selector_travel_mm'],atol=.02)
            labels=[('closed','Closed / tapered flush strip'),('open','Both doors and handles released'),('interior','Silver dish, guides and paired pins'),('edited_detail','Lower room edit: width 145 / recess Ø52 mm')]

        if model!='cary60':
            from .shared_validation import check as shared_check
            shared_report=shared_check(dest,render,values)

    version=subprocess.run([executable,'--version'],capture_output=True,text=True)
    report={'model':model,'engine':(version.stderr+version.stdout).strip(),
        'status':'Shared CAD/simulation part and joint definition',
        'checks':results,'envelope_tolerance_mm':tolerance,
        'default_envelope_matches_within_tolerance':True,
        'limitations':['Curves use finer tessellation than collision proxies; QY smooth handle envelopes may differ by up to 1 mm.',
                       'Internal CSG joins can overlap locally to avoid edge-only non-manifold contacts.',
                       'Checks establish CAD consistency, not real-world contact tolerance or mechanical travel limits.']}
    report['shared_backend_checks']=shared_report
    preview(dest,model,meshes,labels)
    (dest/'csg_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def check_cary_shared(dest,render,values):
    """Exercise changed parameters through CAD, convex GLB, textures and URDF."""
    import copy
    import xml.etree.ElementTree as ET
    from . import cary60,geometry,textures,decals
    from .shared_geometry import read_parameter_include
    p=json.loads((dest.parent/'parameters.json').read_text())
    changes=dict(body_width_mm=values['body_width_mm']+20,
                 body_depth_mm=values['body_depth_mm']+30,
                 body_height_mm=values['body_height_mm']+10,
                 chamber_center_x_mm=values['chamber_center_x_mm']+5,
                 socket_inner_width_mm=16,socket_inner_depth_mm=18,cover_travel_mm=200)
    edited=dict(values,**changes)
    with workspace('edited_parameters') as tmp:
        from .inputs import stage_evidence
        work=Path(tmp);(work/'meshes').mkdir();stage_evidence(p['model'],work)
        include=work/'parameters.scad'
        include.write_text('\n'.join(f'{k} = {v};' for k,v in edited.items()))
        p['design_parameters']=read_parameter_include(include,values)
        m=cary60.model(p)
        (work/'parameters.json').write_text(json.dumps(p,indent=2))
        errors={}
        for part in cary60.definition(edited).parts:
            native=render('shared_'+part.name,part=part.name,**changes)
            if part.kind=='subtract':assert len(native.split())==1,(part.name,'Disconnected Boolean solid')
            sim=trimesh.util.concatenate([q for n,q in m.parts.items() if n==part.name or n.startswith(part.name+'__')])
            np.testing.assert_allclose(native.bounds,sim.bounds*1000,atol=.02,err_msg=part.name)
            # Native circles are smoother; straight-edged primitives should agree in volume.
            relative=abs(native.volume-sim.volume*1e9)/(sim.volume*1e9)
            assert relative < (.03 if part.kind in ('rod','rounded_x') else 3e-5),(part.name,relative)
            errors[part.name]=dict(bounds_max_error_mm=float(np.max(abs(native.bounds-sim.bounds*1000))),volume_relative_error=float(relative),bounds_tolerance_mm=.02,volume_tolerance=.03 if part.kind in ('rod','rounded_x') else 3e-5)
        for state,value in [('closed',0),('open',geometry.open_value(p))]:
            m.scene(value).export(work/'meshes'/f'collision_{state}.glb')
            native=render('shared_edited_'+state,pose=state,**changes)
            np.testing.assert_allclose(native.bounds,m.scene(value).bounds*1000,atol=.02)
        path=geometry.urdf(m,work)
        physics=geometry.validate(m,work,path)
        report=textures.apply('cary60',work)
        textures.urdf_visuals('cary60',work,report['coverage'])
        decals.apply('cary60',work)
        for state,value in [('closed',0),('open',geometry.open_value(p))]:
            visual=trimesh.load(work/'meshes'/f'visual_{state}.glb',force='scene')
            for name,mesh in m.world_parts(value).items():
                transform,key=visual.graph[name];q=visual.geometry[key].copy();q.apply_transform(transform)
                np.testing.assert_allclose(q.bounds,mesh.bounds,atol=1e-7,err_msg=name)
                assert q.visual.kind=='texture',name
        robot=ET.parse(path).getroot()
        np.testing.assert_allclose(float(robot.find('joint/limit').get('upper')),edited['cover_travel_mm']/1000)
        actual=json.loads((work/'textures/decals/provenance.json').read_text())['decals'][0]['target_xxz_z_m']
        np.testing.assert_allclose(actual,np.array(cary60.decal_bounds(edited))/1000,atol=1e-9)
        # Preserve a review PNG of the synchronized parameter-change example, not a second model package.
        from .render import Renderer
        import matplotlib.pyplot as plt
        cam=np.array([.6,-1.6,1.2]);cam/=np.linalg.norm(cam);u=np.cross([0,0,1],cam);u/=np.linalg.norm(u);v=np.cross(cam,u)
        worlds=[cary60.model(copy.deepcopy(json.loads((dest.parent/'parameters.json').read_text()))).world_parts(values['cover_travel_mm']/1000),m.world_parts(geometry.open_value(p))]
        points=np.vstack([q.vertices for group in worlds for q in group.values()])*1000
        xx,yy=points@u,points@v;extent=[xx.min()-15,xx.max()+15,yy.min()-15,yy.max()+15]
        fig,axes=plt.subplots(1,2,figsize=(13,6))
        for ax,group,label in zip(axes,worlds,['Current dimensions','Test edit: body +20/+30/+10 mm; travel 200 mm']):
            ax.imshow(Renderer(group).image(u,v,cam,extent,width=850));ax.axis('off');ax.set_title(label)
        fig.suptitle('Cary: one definition drives CAD, GLB and URDF');fig.tight_layout()
        fig.savefig(dest/'csg_shared_parameters.png',dpi=140);plt.close(fig)
    return {'edited_values':changes,'parts_checked':errors,
            'numeric_scad_parameter_import':True,'textured_glb_bounds_match_collision':True,
            'decal_follows_housing':True,'urdf_joint_limit_matches':True,
            'pybullet_sweep_states':physics['sampled_states'],
            'maximum_self_penetration_mm':physics['maximum_self_penetration_mm'],
            'test_changes_published_as_model':False}


def preview(dest,model,meshes,labels):
    import matplotlib.pyplot as plt
    from .render import Renderer
    camera=np.array([.6,-1.6,1.2]);camera/=np.linalg.norm(camera)
    u=np.cross([0,0,1],camera);u/=np.linalg.norm(u);v=np.cross(camera,u)
    figure,axes=plt.subplots(2,2,figsize=(13,10))
    for ax,(name,label) in zip(axes.flat,labels):
        mesh=meshes[name].copy();vertices=mesh.vertices.copy();mesh.apply_scale(.001)
        mesh.visual.face_colors=[150,158,170,255]
        x,y=vertices@u,vertices@v;extent=[x.min()-10,x.max()+10,y.min()-10,y.max()+10]
        ax.imshow(Renderer({'csg':mesh}).image(u,v,camera,extent,width=900));ax.axis('off');ax.set_title(label)
    figure.suptitle(model.upper()+': actual OpenSCAD solids and parameter edits',fontsize=17)
    note='Shared part/joint definition; parameter import rebuilds CAD, textured GLB and URDF.'
    figure.text(.03,.02,note+' Local dimensions remain estimates.',fontsize=10)
    figure.tight_layout(rect=(0,.05,1,.95));figure.savefig(dest/'csg_review.png',dpi=140);plt.close(figure)
