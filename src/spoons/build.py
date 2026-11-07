"""Rebuild corrected spoon visuals and explicit convex colliders in metres.

python -m src.spoons.build [spoon_daiso_pink|spoon_home_black]
"""
from pathlib import Path
import json,sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree,ConvexHull
from shapely.geometry import Polygon,Point
import shapely
import open3d as o3d
from PIL import Image,ImageDraw,ImageFont
from src.instruments.render import Renderer
from src.assets import save_manifest
from src.export import sha256
from src.collision import validate_parts

ROOT=Path(__file__).resolve().parents[2]
PARAMETERS=Path(__file__).parent/'inputs/designs.json'


def json_write(path,data):
    path.write_text(json.dumps(data,indent=2)+'\n')


def ray_scene(mesh):
    s=o3d.t.geometry.RaycastingScene(nthreads=4)
    s.add_triangles(o3d.core.Tensor(mesh.vertices.astype('float32')),o3d.core.Tensor(mesh.faces.astype('uint32')))
    return s


def heights(scene,xy):
    xy=np.asarray(xy);n=len(xy)
    def cast(z,d):
        rays=np.c_[xy,np.full(n,z),np.tile([0,0,d],(n,1))].astype('float32')
        dist=scene.cast_rays(o3d.core.Tensor(rays),nthreads=4)['t_hit'].numpy()
        return z+d*dist
    return cast(.5,-1),cast(-.5,1)


def surface_shell(xy,top,bottom,nr,na,rim_round=0):
    vertices=np.vstack([np.c_[xy,top],np.c_[xy,bottom]])
    count=len(xy);faces=[]
    # A single centre vertex followed by rings.
    for j in range(na):faces.append([0,1+j,1+(j+1)%na])
    for k in range(nr-1):
        a=1+k*na;b=a+na
        for j in range(na):
            q=(j+1)%na;faces.extend([[a+j,b+j,b+q],[a+j,b+q,a+q]])
    faces=np.asarray(faces);faces=np.vstack([faces,faces[:,::-1]+count])
    rim=1+(nr-1)*na;sides=[]
    previous=np.arange(rim,rim+na)
    if rim_round:
        edge=xy[rim:];direction=edge-xy[0];direction/=np.linalg.norm(direction,axis=1,keepdims=True)
        mid=(top[rim:]+bottom[rim:])/2;half=(top[rim:]-bottom[rim:])/2
        for angle in np.linspace(0,np.pi,9)[1:-1]:
            ring=np.c_[edge+direction*rim_round*np.sin(angle),mid+half*np.cos(angle)]
            current=np.arange(len(vertices),len(vertices)+na);vertices=np.vstack([vertices,ring])
            for j in range(na):
                q=(j+1)%na;sides.extend([[previous[j],current[j],current[q]],[previous[j],current[q],previous[q]]])
            previous=current
    current=np.arange(rim,rim+na)+count
    for j in range(na):
        q=(j+1)%na;sides.extend([[previous[j],current[j],current[q]],[previous[j],current[q],previous[q]]])
    mesh=trimesh.Trimesh(vertices,np.vstack([faces,sides]),process=True)
    mesh.fix_normals()
    if mesh.volume<0:mesh.invert()
    assert mesh.is_volume
    return mesh


def crown_geometry():
    points=[[-21,12],[-12,7],[-6,8],[0,20],[6,8],[12,7],[21,12],[16,-12],[-16,-12]]
    polygon=Polygon(np.array(points)/1000)
    for x,y in [(-21,12),(0,20),(21,12)]:polygon=polygon.union(Point(x/1000,y/1000).buffer(.0032))
    return polygon.buffer(.0006).buffer(-.0006)


def pink_boundary(theta,p):
    # The rounded lip grows outward to the requested outer envelope.
    x=np.sin(theta)*(1-.22*np.cos(theta));x*=(p['width_mm']-2*p['rim_round_mm'])/2000/1.022913
    y=np.cos(theta)*(p['bowl_length_mm']-2*p['rim_round_mm'])/2000+.0015
    return np.c_[x,y]


def pink_profile(r,p):
    """Bowl height and radial derivative, smoothly flattening into the lip."""
    r=np.asarray(r);f=r**2.4;derivative=2.4*r**1.4
    start=p['rim_flat_start'];h=1-start;u=np.clip((r-start)/h,0,1)
    v=start**2.4;dv=2.4*start**1.4
    transition=(2*u**3-3*u**2+1)*v+(u**3-2*u**2+u)*h*dv+(-2*u**3+3*u**2)
    slope=((6*u*u-6*u)*v+(3*u*u-4*u+1)*h*dv+(-6*u*u+6*u))/h
    return np.where(r>start,transition,f),np.where(r>start,slope,derivative)


def pink_surface_gradient(xy,r,p):
    """Analytic gradient of the undecorated inner bowl in world XY."""
    ax=(p['width_mm']-2*p['rim_round_mm'])/2000/1.022913
    by=(p['bowl_length_mm']-2*p['rim_round_mm'])/2000
    c=np.clip((xy[:,1]-.0015)/np.maximum(r*by,1e-12),-1,1)
    s=np.sign(xy[:,0])*np.sqrt(np.maximum(0,1-c*c))
    bx=ax*s*(1-.22*c);yy=by*c;dx=ax*(c*(1-.22*c)+.22*s*s);dy=-by*s
    determinant=bx*dy-yy*dx
    grad_r=np.c_[dy,-dx]/np.where(abs(determinant)>1e-12,determinant,1)[:,None]
    _,slope=pink_profile(r,p)
    return grad_r*(slope*(p['rim_height_mm']-p['floor_height_mm'])/1000)[:,None]


def pink_levels(xy,r,p,relief=True):
    profile,_=pink_profile(r,p)
    top=p['floor_height_mm']/1000+(p['rim_height_mm']-p['floor_height_mm'])/1000*profile
    if relief:
        shape=crown_geometry();points=shapely.points(xy)
        signed=shapely.distance(points,shape.boundary)*np.where(shapely.contains(shape,points),1,-1)
        blend=np.clip((signed+.0007)/.0014,0,1);blend=blend*blend*(3-2*blend)
        top=top-p['crown_depth_mm']/1000*blend
    # A fixed vertical gap made steep sides paper-thin. Compensate for slope
    # so thickness is approximately constant normal to the material.
    gradient=pink_surface_gradient(xy,r,p)
    thickness=p['wall_normal_mm']/1000*np.sqrt(1+np.sum(gradient**2,axis=1))
    return top,top-thickness


def socket_ring(y,angles,p,inner=False):
    """Loft the open socket into the shell, burying its far end in the wall."""
    start,end=np.array(p['socket_blend_y_mm'])/1000
    u=np.clip((y-start)/(end-start),0,1);blend=u*u*(3-2*u)
    rx,rz=(.012,.0075) if not inner else (.010,.0055)
    rx=rx*(1-blend)+(p['socket_flare_halfwidth_mm']/1000-(.002 if inner else 0))*blend
    x=rx*np.cos(angles);xy=np.c_[x,np.full(len(x),y)]
    ax=(p['width_mm']-2*p['rim_round_mm'])/2000/1.022913
    by=(p['bowl_length_mm']-2*p['rim_round_mm'])/2000
    yy=(y-.0015)/by;r=np.full(len(x),max(abs(yy),.001))
    for _ in range(20):r=np.sqrt((x/(ax*(1-.22*yy/r)))**2+yy**2)
    top,bottom=pink_levels(xy,np.minimum(r,1),p,relief=False)
    centre=(1-blend)*(.033+(y+.076)/.032*(-.004))+blend*(top+bottom)/2
    radius=rz*(1-blend)+(.0008 if not inner else .0003)*blend
    return np.c_[x,np.full(len(x),y),centre+radius*np.sin(angles)]


def tube_socket(p):
    # Dense, smoothly tapered loft; the mouth and pin hole remain open.
    na=64;ny=49;verts=[]
    for inner in (False,True):
        for y in np.linspace(-.076,p['socket_blend_y_mm'][1]/1000,ny):
            verts.extend(socket_ring(y,np.arange(na)*2*np.pi/na,p,inner))
    verts=np.array(verts);faces=[];count=ny*na
    for side in (0,1):
        for k in range(ny-1):
            for j in range(na):
                a=side*count+k*na+j;b=side*count+k*na+(j+1)%na;c=b+na;d=a+na
                faces.extend([[a,b,c],[a,c,d]] if side==0 else [[a,c,b],[a,d,c]])
    for k in (0,ny-1):
        for j in range(na):
            a=k*na+j;b=k*na+(j+1)%na
            faces.extend([[a,a+count,b+count],[a,b+count,b]])
    mesh=trimesh.Trimesh(verts,faces,process=True);mesh.fix_normals()
    if mesh.volume<0:mesh.invert()
    hole=trimesh.creation.cylinder(radius=.0021,height=.02,sections=32)
    hole.apply_translation([0,-.066,.021])
    mesh=trimesh.boolean.difference([mesh,hole],engine='manifold')
    return mesh


def build_pink(p):
    nr=p['radial_sections'];na=p['angular_sections'];theta=np.arange(na)*2*np.pi/na
    boundary=pink_boundary(theta,p);center=np.array([0,.0015])
    radii=np.arange(1,nr+1)/nr
    xy=np.vstack([center,np.concatenate([center+r*(boundary-center) for r in radii])])
    r=np.r_[0,np.repeat(radii,na)];top,bottom=pink_levels(xy,r,p)
    shell=surface_shell(xy,top,bottom,nr,na,p['rim_round_mm']/1000);socket=tube_socket(p)
    mesh=trimesh.boolean.union([shell,socket],engine='manifold')
    # The buried tube end can enclose a tiny sealed air pocket. It is not part
    # of the open handle socket and is unnecessary for this solid RL asset.
    mesh=max(mesh.split(),key=lambda part:part.volume)
    neck_report=fair_neck(mesh,p)
    mesh.visual.vertex_colors=np.tile(p['rgba'],(len(mesh.vertices),1))
    parts={};nt=p['collision_angular_sections'];rings=p['collision_rings']
    for j in range(nt):
        angles=np.linspace(j*2*np.pi/nt,(j+1)*2*np.pi/nt,3);edge=pink_boundary(angles,p)
        for k,(r0,r1) in enumerate(zip(rings[:-1],rings[1:])):
            rr=np.linspace(r0,r1,3);q=np.concatenate([center+v*(edge-center) for v in rr]);rv=np.repeat(rr,3)
            a,b=pink_levels(q,rv,p,relief=False)
            cloud=np.vstack([np.c_[q,a],np.c_[q,b]])
            if r1==1:
                direction=edge-center;direction/=np.linalg.norm(direction,axis=1,keepdims=True)
                t,z=pink_levels(edge,np.ones(len(edge)),p,relief=False)
                for angle in np.linspace(0,np.pi,7):
                    cloud=np.vstack([cloud,np.c_[edge+direction*p['rim_round_mm']/1000*np.sin(angle),(t+z)/2+(t-z)/2*np.cos(angle)]])
            parts[f'bowl_{j:02}_{k:02}']=trimesh.convex.convex_hull(cloud)
    # Follow the curved neck with short convex sectors, retaining the opening.
    for i in range(8):
        a=np.linspace(i*2*np.pi/8,(i+1)*2*np.pi/8,5)
        levels=np.linspace(-.076,p['socket_blend_y_mm'][1]/1000,5)
        for k,(lo,hi) in enumerate(zip(levels[:-1],levels[1:])):
            v=np.concatenate([socket_ring(y,a,p,inner) for y in np.linspace(lo,hi,5) for inner in (False,True)])
            parts[f'socket_{i:02}_{k:02}']=trimesh.convex.convex_hull(v)
    # Crown stamp is shallow; use a small number of blocks on the exterior.
    # A convex base and three tips approximate the external stamp only.
    crown_polys=[Polygon([[-.016,-.012],[.016,-.012],[.020,.011],[.012,.007],[.006,.008],[0,.019],[-.006,.008],[-.012,.007],[-.020,.011]])]
    from shapely.ops import triangulate
    for i,triangle in enumerate(triangulate(crown_polys[0])):
        if not crown_polys[0].covers(triangle.representative_point()):continue
        q=np.array(triangle.exterior.coords)[:-1];r=np.linalg.norm((q-center)/np.array([.05,.0585]),axis=1)
        a,b=pink_levels(q,r,p,relief=False);b-=p['crown_depth_mm']/1000
        parts[f'crown_{i:02}']=trimesh.convex.convex_hull(np.vstack([np.c_[q,a-.002],np.c_[q,b]]))
    probes=np.c_[xy[1:][radii.repeat(na)<.82],top[1:][radii.repeat(na)<.82]+.004]
    # The visible socket occupies this rear strip; it is not open bowl space.
    probes=probes[~((probes[:,1]<-.042)&(abs(probes[:,0])<.014))]
    # Measure through steep wall regions, away from crown/socket features.
    angles=np.linspace(-1.35,1.35,13);edge=pink_boundary(angles,p)
    sample_xy=np.concatenate([center+v*(edge-center) for v in (.65,.75,.85)])
    sample_r=np.repeat([.65,.75,.85],len(angles));z,_=pink_levels(sample_xy,sample_r,p)
    normal=np.c_[-pink_surface_gradient(sample_xy,sample_r,p),np.ones(len(sample_r))]
    normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    rays=np.c_[np.c_[sample_xy,z]-normal*.00005,-normal].astype('float32')
    thickness=ray_scene(mesh).cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()+.00005
    if not(np.isfinite(thickness).all() and thickness.min()>.0026):
        raise ValueError('Pink material thickness check failed: '+str(thickness))
    return mesh,parts,{'construction':'Solid bowl with slope-corrected material thickness, rounded lip, crown stamp and hollow socket lofted smoothly into the rear wall. Collision uses coarse shell cells and curved socket sectors; crown recess and socket pin hole are simplified.','neck_fairing':neck_report,'nominal_wall_normal_mm':p['wall_normal_mm'],'rounded_lip_outward_mm':p['rim_round_mm'],'measured_steep_wall_thickness_mm':{'minimum':float(thickness.min()*1000),'maximum':float(thickness.max()*1000),'samples':len(thickness)},'cavity_probes':probes}


def polynomial(x,y):
    return np.c_[np.ones(len(x)),x,y,x*x,x*y,y*y,x**3,x*x*y,x*y*y,y**3]


def fair_neck(mesh,p):
    """Local surface fairing with a smooth falloff and fixed outer geometry."""
    from trimesh.smoothing import laplacian_calculation
    original=mesh.vertices.copy();lo,hi=np.array(p['neck_blend_y_mm'])/1000
    t=np.clip((original[:,1]-lo)/(hi-lo),0,1)
    weight=np.sin(np.pi*t)**2;weight[(t==0)|(t==1)]=0
    if 'neck_halfwidth_mm' in p:
        x=np.clip(abs(original[:,0])/(p['neck_halfwidth_mm']/1000),0,1)
        weight*=((1-x*x)**2)
    laplacian=laplacian_calculation(mesh)
    xyz=original.copy()
    for _ in range(p['neck_smoothing_iterations']):
        # Paired Taubin steps suppress reconstruction noise without narrowing
        # the neck as repeated averaging alone would do.
        steps=(.3,) if p.get('neck_fairing_method')=='average' else (.5,-.53)
        for step in steps:xyz+=step*weight[:,None]*(laplacian@xyz-xyz)
    mesh.vertices=xyz
    displacement=np.linalg.norm(xyz-original,axis=1)
    assert np.array_equal(xyz[weight==0],original[weight==0])
    assert mesh.is_volume
    return {'y_interval_mm':p['neck_blend_y_mm'],'vertices':int((weight>0).sum()),
            'maximum_displacement_mm':float(displacement.max()*1000),
            'outside_neck_displacement_mm':0.0}


def build_black(p):
    from .black import build as build_authored_black
    return build_authored_black(p)


def preview(mesh,parts,out,name):
    m=mesh.copy();m.apply_translation(-m.bounds.mean(0));r=Renderer({'mesh':m},smooth=True)
    board=Image.new('RGB',(1800,720),'#f7f7f7');draw=ImageDraw.Draw(board);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    for i,(label,cam) in enumerate([('Inside / top',(0,0,1)),('Oblique',(.75,-.8,.8)),('Side',(1,0,.02)),('Underside',(0,0,-1))]):
        c=np.array(cam,float);c/=np.linalg.norm(c);u=np.cross([0,1,0],c);u/=np.linalg.norm(u);v=np.cross(c,u);span=m.extents.max()*1000*1.18
        im=Image.fromarray(r.image(u,v,c,[-span*.42,span*.42,-span*.55,span*.55],width=440));board.paste(im,(i*450,55),im);draw.text((i*450+12,15),label,font=font,fill='#222222')
    draw.text((12,665),name+' | metres | separate simplified convex collision assembly',font=font,fill='#222222');board.save(out/(name+'_preview.png'))
    # Side section: directly reveals the hollow bowl and thin wall.
    cross=mesh.section(plane_origin=[0,0,0],plane_normal=[1,0,0])
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    if cross:
        for line in cross.discrete:ax.plot(line[:,1]*1000,line[:,2]*1000,color='#bd537e' if 'pink' in name else '#252525',lw=1)
    ax.set_aspect('equal');ax.set_xlabel('Length Y (mm)');ax.set_ylabel('Height Z (mm)');ax.set_title(name+' — centre section, open cavity');fig.savefig(out/'section.png',dpi=150,bbox_inches='tight');plt.close(fig)


def build(name):
    p=json.loads(PARAMETERS.read_text())[name];root=ROOT/'output/objects/hestia-rl'/name/'codex';out=root/'products';out.mkdir(parents=True,exist_ok=True)
    mesh,parts,report=build_pink(p) if name=='spoon_daiso_pink' else build_black(p)
    shift=np.array([0,0,-mesh.bounds[0,2]]);mesh.apply_translation(shift)
    for m in parts.values():m.apply_translation(shift)
    probes=report.pop('cavity_probes')+shift
    validate_parts(parts)
    # Cavity clearance is a geometric constraint, not a rendering judgement.
    from src.products import probe_collision
    occupied=probe_collision(parts,probes*1000)
    if occupied.any():raise ValueError(f'{name}: collision fills {occupied.sum()} of {len(probes)} cavity probes')
    if not(mesh.is_volume and mesh.is_winding_consistent):raise ValueError('Invalid visual volume')
    visual=out/(name+'_m.glb');mesh.export(visual)
    if name=='spoon_home_black':
        import shutil
        from .black import VISUAL, COLLISION
        shutil.copy2(VISUAL,visual)
    scene=trimesh.Scene()
    for key,m in parts.items():scene.add_geometry(m,node_name=key,geom_name=key)
    collision=out/(name+'_collision_m.glb');scene.export(collision)
    if name=='spoon_home_black':shutil.copy2(COLLISION,collision)
    loaded=trimesh.load(visual,force='mesh');np.testing.assert_allclose(loaded.bounds,mesh.bounds,atol=3e-8)
    np.testing.assert_array_equal(loaded.visual.vertex_colors,mesh.visual.vertex_colors)
    surface=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(loaded.vertices),o3d.utility.Vector3iVector(loaded.faces))
    intersections=len(surface.get_self_intersecting_triangles())
    if intersections:raise ValueError(f'{name}: {intersections} intersecting triangle pairs')
    components=len(loaded.split())
    if components!=1:raise ValueError(f'{name}: {components} disconnected surface components')
    physics={'mass_kg':p['mass_kg'],'friction':[.6,.005,.0001],'solref':[.004,1],'solimp':[.99,.999,.0001],'priority':1,'margin':.0005,'provenance':'Estimated placeholder mass and friction; not measured.'};json_write(out/'physics.json',physics)
    report.update(units='metres',visual_triangles=len(mesh.faces),collision_parts=len(parts),collision_triangles=sum(len(m.faces) for m in parts.values()),watertight=bool(loaded.is_watertight),connected_surface_components=components,self_intersecting_triangle_pairs=intersections,cavity_probes_free=len(probes),colour_reload_exact=True,bounds_m=mesh.bounds.tolist(),dimensions_m=mesh.extents.tolist(),physical_dimensions_estimated=True,mujoco_runtime_tested=False)
    report['dependencies']={'authoring':'Accepted visual/collision mesh sources and explicit provenance' if name=='spoon_home_black' else 'Python geometry and JSON profiles; original HEIC references',
                            'sam3d_inference_required':False,'saved_accepted_mesh_required':name=='spoon_home_black',
                            'historically_sam3d_derived':name=='spoon_home_black',
                            'experimental_method_directory_required':False,
                            'runtime':'Self-contained MuJoCo OBJ/MTL/PNG/XML delivery'}
    json_write(out/'parameters.json',p);json_write(out/'validation.json',report)
    sources=[Path(__file__),PARAMETERS]
    if name=='spoon_home_black':
        from .black import sources as black_sources
        sources.extend(black_sources())
    sources.extend(sorted((ROOT/'dataset/hestia-rl'/name).glob('*.HEIC')))
    save_manifest(root/'asset.json',kind='accepted_mesh_spoon' if name=='spoon_home_black' else 'parametric_spoon',sources=sources,variants={name:{'visual':visual,'collision':collision,'physics':out/'physics.json'}},units='m',frame='X width, Y length (bowl toward +Y), Z up, lowest visual point Z=0',rebuild=['python','-m','src.spoons.build',name])
    (out/'README.txt').write_text(name+'\nEditable recipe: src/spoons/inputs/designs.json\nRebuild: python -m src.spoons.build '+name+'\nVisual and convex collision GLBs use metres. Keep collision pieces separate in MuJoCo.\nDimensions, thickness, mass and friction are estimates; see parameters.json.\nOriginal captures are retained. Black uses accepted mesh sources with explicit historical provenance; neither spoon needs a SAM3D directory or inference.\n')
    preview(mesh,parts,out,name);print(name,json.dumps(report),flush=True)
    return root

if __name__=='__main__':
    for name in sys.argv[1:] or ['spoon_daiso_pink','spoon_home_black']:build(name)
