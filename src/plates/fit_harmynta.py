"""Inspect the November RGB-D archive and fit a compact rotational HARMYNTA profile.

Run with --write to update the editable input, then run src.plates.authored.
Only the two close underside views fit the body shape. Other accepted inverted
views are validation data. Upright/low-angle silhouettes are not table circles.
RGB/depth correspondence is assumed from the paired files, not independently
calibrated: the archive has one intrinsic set and no color-depth extrinsics.
"""
import argparse
import os
os.environ['OMP_NUM_THREADS']='4'
import zipfile,io,json
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
import open3d as o3d
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'work/hestia-rl/plate_harmynta_white_22cm/codex/refit_20261107'
ARCHIVE=ROOT/'dataset/hestia-rl/plate_harmynta_white_22cm/raw/captures_20261107_1213_plate.zip'
INPUT=ROOT/'src/plates/inputs/plate_harmynta_white_22cm.json'

def rays(uv,m):
    # Inverse Brown-Conrady, following librealsense rs2_deproject_pixel_to_point.
    # https://github.com/realsenseai/librealsense/blob/master/src/rs.cpp
    xo=(uv[...,0]-m['cx'])/m['fx'];yo=(uv[...,1]-m['cy'])/m['fy']
    x=xo.copy();y=yo.copy();k1,k2,p1,p2,k3=m['distortion_coeffs']
    for _ in range(10):
        r2=x*x+y*y;inv=1/(1+((k3*r2+k2)*r2+k1)*r2)
        xq=x/inv;yq=y/inv
        x=(xo-2*p1*xq*yq-p2*(r2+2*xq*xq))*inv
        y=(yo-2*p2*xq*yq-p1*(r2+2*yq*yq))*inv
    return np.stack([x,y,np.ones_like(x)],axis=-1)


def project(points,m):
    """Inverse-Brown projection paired with rays(), per librealsense."""
    x=points[:,0]/points[:,2];y=points[:,1]/points[:,2]
    k1,k2,p1,p2,k3=m['distortion_coeffs'];r2=x*x+y*y
    scale=1+k1*r2+k2*r2*r2+k3*r2*r2*r2;x=x*scale;y=y*scale
    xx=x+2*p1*x*y+p2*(r2+2*x*x);yy=y+2*p2*x*y+p1*(r2+2*y*y)
    return np.c_[xx*m['fx']+m['cx'],yy*m['fy']+m['cy']]

def inspect_archive():
    OUT.mkdir(parents=True,exist_ok=True)
    z=zipfile.ZipFile(ARCHIVE)
    board=Image.new('RGB',(1280,300*5),'white');draw=ImageDraw.Draw(board)
    fig,axs=plt.subplots(4,3,figsize=(14,13));reports=[];clouds={}
    for i,n in enumerate(sorted(n for n in z.namelist() if n.endswith('_rgb.png'))):
        rgb=np.array(Image.open(io.BytesIO(z.read(n))))
        depth=np.array(Image.open(io.BytesIO(z.read(n.replace('_rgb.png','_depth.png')))))
        m=json.loads(z.read(n.replace('_rgb.png','.json')));depth=depth*m['depth_scale_m_per_unit']*1000
        uv=np.stack(np.meshgrid(np.arange(640),np.arange(480)),axis=-1);ray=rays(uv,m);xyz=ray*depth[...,None]
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        contours,_=cv2.findContours((gray>110).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cs=[c for c in contours if cv2.contourArea(c)>4000]
        contour=min(cs,key=lambda c: np.linalg.norm(c.reshape(-1,2).mean(0)-[320,250]))
        mask=np.zeros(gray.shape,'uint8');cv2.drawContours(mask,[contour],-1,1,-1)
        outer=cv2.dilate(mask,np.ones((71,71),'uint8'));inner=cv2.dilate(mask,np.ones((15,15),'uint8'))
        sel=(outer>0)&(inner==0)&(depth>100)&(depth<900)
        points=xyz[sel][::3]
        pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points));o3d.utility.random.seed(42)
        plane,idx=pc.segment_plane(distance_threshold=1.5,ransac_n=3,num_iterations=400)
        normal=np.array(plane[:3]);offset=plane[3]
        if normal[2]>0:normal=-normal;offset=-offset
        u=np.cross(normal,[0,1,0]);u/=np.linalg.norm(u);v=np.cross(normal,u)
        ctr_uv=contour.reshape(-1,2)[::3]
        ctr_uv=ctr_uv[(ctr_uv[:,0]>1)&(ctr_uv[:,0]<638)&(ctr_uv[:,1]>1)&(ctr_uv[:,1]<478)]
        rr=rays(ctr_uv,m);edge=rr*(-offset/(rr@normal))[:,None]
        xy=np.c_[edge@u,edge@v]
        opt=least_squares(lambda p: np.linalg.norm(xy-p[:2],axis=1)-p[2],np.r_[xy.mean(0),110],loss='soft_l1',f_scale=1)
        for _ in range(3):
            error=np.abs(np.linalg.norm(xy-opt.x[:2],axis=1)-opt.x[2])
            good=error<max(1.5,3*np.median(error))
            opt=least_squares(lambda p: np.linalg.norm(xy[good]-p[:2],axis=1)-p[2],opt.x,loss='soft_l1',f_scale=1)
        center=opt.x[0]*u+opt.x[1]*v-offset*normal
        delta=xyz-center;h=delta@normal;rad=np.sqrt(np.maximum((delta*delta).sum(2)-h*h,0))
        safe=cv2.erode(mask,np.ones((9,9),'uint8'))>0
        # Exclude missing measurements, dark markers/print, flying edge points.
        valid=safe&(depth>100)&(gray>110)&(rad<108)&(h>-3)&(h<46)
        local_z=40-h
        table_err=np.abs(points[idx]@normal+offset)
        report=dict(index=i,file=Path(n).stem,plane=np.asarray(plane).tolist(),normal=normal.tolist(),center_mm=center.tolist(),
                    rim_radius_mm=float(opt.x[2]),rim_radial_median_error_mm=float(np.median(abs(opt.fun))),
                    table_inlier_fraction=len(idx)/len(points),table_median_error_mm=float(np.median(table_err)),
                    plate_valid_fraction=float(((depth>0)&(mask>0)).sum()/mask.sum()))
        reports.append(report)
        clouds[f'r_{i}']=rad[valid];clouds[f'z_{i}']=local_z[valid];clouds[f'uv_{i}']=uv[valid]
        preview=rgb.copy();cv2.drawContours(preview,[contour],-1,(0,255,0),2)
        preview[valid]=(preview[valid]*.65+np.array([0,90,180])*.35).astype('uint8')
        tile=Image.fromarray(preview);tile.thumbnail((426,270));xx=i%3*426;yy=i//3*300;board.paste(tile,(xx,yy+28))
        draw.text((xx+5,yy+3),f'{i} R={opt.x[2]:.1f} edge err={np.median(abs(opt.fun)):.1f}',fill='black')
        if i>=3:
            ax=axs.ravel()[i-3];ax.scatter(rad[valid][::8],local_z[valid][::8],s=.3,alpha=.2)
            for lo in np.arange(4,108,4):
                a=local_z[valid&(rad>=lo)&(rad<lo+4)]
                if len(a)>30:ax.plot(lo+2,np.median(a),'r.')
            ax.set(xlim=(0,110),ylim=(-5,45),title=f'{i} {Path(n).stem[9:15]}',xlabel='radius mm',ylabel='upright z mm');ax.grid()
        print(f'View {i}: rim radius {opt.x[2]:.2f} mm; radial residual {np.median(abs(opt.fun)):.2f} mm',flush=True)
    board.save(OUT/'depth_masks.jpg');fig.tight_layout();fig.savefig(OUT/'depth_profiles.png',dpi=120)
    (OUT/'capture_inspection.json').write_text(json.dumps(reports,indent=2)+'\n');np.savez(OUT/'radial_samples.npz',**clouds)
    z.close()
    plt.close(fig)


def fit_profile(write=False):
    samples=np.load(OUT/'radial_samples.npz')
    captures=json.loads((OUT/'capture_inspection.json').read_text())
    baseline=OUT/'before/input.json'
    if not baseline.exists():
        raise FileNotFoundError('Save the original input at '+str(baseline)+' before fitting.')
    p=json.loads(baseline.read_text());old=np.array(p['profile_radius_top_bottom_mm'])
    # User measurements. Direct width is redundant and inconsistent with the
    # diameters, so it is recorded rather than silently changing either diameter.
    inner,outer,height=87.7/2,102.5/2,5.0
    train=[9,12];held=[4,5,6,10,11]
    radii=[];medians=[];weights=[];views=[];recess=[]
    for j,i in enumerate(train):
        r=samples[f'r_{i}'];zz=samples[f'z_{i}']
        recess.append(float(np.median(zz[(r>10)&(r<35)])-np.median(zz[(r>45)&(r<50)])))
        for lo in range(58,102,4):
            values=zz[(r>=lo)&(r<lo+4)]
            median=np.median(values);mad=np.median(abs(values-median))
            radii.append(lo+2);medians.append(median);weights.append(1/max(2,mad));views.append(j)
    radii=np.array(radii);medians=np.array(medians);weights=np.array(weights);views=np.array(views)
    def wall(r,exponent):
        return height+(38.7-height)*np.clip((r-outer)/(110-outer),0,1)**exponent
    def residual(x):
        return np.r_[(wall(radii,x[0])-medians-x[1:][views])*weights,x[1:]*.025]
    solution=least_squares(residual,[1.4,2,2],bounds=([1.05,-8,-8],[2.5,8,8]),loss='soft_l1')
    exponent=float(solution.x[0])
    # The central recess is not the external ring height: estimate its relief
    # against the ring in the same view, cancelling table-plane depth offsets.
    recess_depth=round(float(np.clip(np.median(recess),1,5)),2)
    radial=np.unique(np.r_[old[:,0],0,35,40,inner,inner+.12,inner+.35,
                          outer-.35,outer-.12,outer,np.arange(outer+2.5,108,2.5),108,109,109.7,110])
    # Keep the existing inner dish; only sample its shape-preserving interpolant.
    top=PchipInterpolator(old[:,0],old[:,1])(radial)
    ring_r=np.array([0,inner,inner+.12,inner+.35,outer-.35,outer-.12,outer])
    ring_z=np.array([recess_depth,recess_depth,.5,0,0,.5,height])
    bottom=np.where(radial<=outer,np.interp(radial,ring_r,ring_z),wall(radial,exponent))
    if not np.all(top>bottom):raise ValueError('Fitted underside intersects the inner dish')
    profile=np.round(np.c_[radial,top,bottom],5)
    p['profile_radius_top_bottom_mm']=profile.tolist();p['profile_tolerance_mm']=.1;p['angular_sections']=64
    p['foot_ring_measurements_mm']={
        'outer_diameter':102.5,'inner_diameter':87.7,'height':5,
        'direct_width':7.82,'diameter_derived_width':7.4,'width_discrepancy':.42,
        'source':'User caliper measurements supplied 2026-11-07; diameters and ring height are hard constraints.',
        'height_datum':'External foot ring from contact z=0 to outer wall attachment z=5 mm.',
        'edge_treatment':'0.35 mm radial edge transitions lie within the nominal ring envelope; contact land is narrower.',
        'central_recess_depth_estimate':recess_depth}
    p['profile_fit']={'body_exponent':exponent,'body_start_radius_mm':outer,
        'body_start_height_mm':height,'body_end_radius_mm':110,'body_end_height_mm':38.7,
        'training_views':[captures[i]['file'] for i in train],
        'held_out_views':[captures[i]['file'] for i in held],
        'report':str((OUT/'fit_report.json').relative_to(ROOT))}
    p['status_label']='Measured foot ring; depth-constrained underside; nominal 220 x 40 mm envelope'
    p['generate_rl_variant']=True
    p['geometry_evidence_files']=[str(ARCHIVE.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT)),
        str((OUT/'fit_report.json').relative_to(ROOT)),str(baseline.relative_to(ROOT))]
    p['evidence']['profile']=(
        '2026-11-07 revision: user-measured foot diameters 102.5/87.7 mm and external ring height 5 mm. '
        'Smooth power-law underside fit to robust radial depth medians in two close inverted captures; '
        'five other inverted views held out. Central recess depth is estimated relative to the foot in depth. '
        'Inner dish retained from the previous photo-authored profile. Nominal 220 x 40 mm envelope retained. '
        'Depth offsets, glaze noise, inferred thickness and rounded edge dimensions remain uncertain.')
    p['evidence']['depth_calibration']=(
        'Archive metadata: D405, 640x480, 0.1 mm depth encoding units (not accuracy), inverse Brown-Conrady. '
        'Depth deprojection follows librealsense rs2_deproject_pixel_to_point. Table planes and projected '
        'rim circles independently locate each inverted view. No rigid fusion across plate flips. '
        'Paired RGB/depth registration is assumed; extrinsics and separate color intrinsics are absent. '
        'Per-view fitted vertical offsets are nuisance alignment terms, not sensor calibration.')
    rows=[]
    for i in train+held:
        rr=samples[f'r_{i}'];zz=samples[f'z_{i}'];valid=(rr>=58)&(rr<104)
        row={'view':i,'file':captures[i]['file'],'role':'fit' if i in train else 'held_out','samples':int(valid.sum())}
        for label,pr in [('previous',old),('revised',profile)]:
            error=np.interp(rr[valid],pr[:,0],pr[:,2])-zz[valid]
            shift=float(np.median(error));aligned=abs(error-shift)
            row[label]={'raw_median_abs_error_mm':float(np.median(abs(error))),
                        'vertical_offset_mm':shift,'offset_corrected_median_abs_error_mm':float(np.median(aligned)),
                        'offset_corrected_p90_abs_error_mm':float(np.percentile(aligned,90))}
        rows.append(row)
    report={'method':'Measured rotational foot + robust low-parameter underside depth fit',
        'body_exponent':exponent,'fit_vertical_offsets_mm':solution.x[1:].tolist(),
        'central_recess_depth_estimates_mm':recess,'central_recess_depth_mm':recess_depth,
        'hard_dimensions_mm':p['foot_ring_measurements_mm'],'per_view_comparison':rows,
        'evaluation_scope':'Outer underside at radii 58-104 mm. Each model/view gets its own median vertical alignment. '
            'Errors are scatter against sensor observations, not calibrated object accuracy; foot is enforced from calipers.',
        'excluded_views':{'0-2':'Upright; tabletop does not coincide with rim; existing inner profile retained.',
            '3,7,8':'Oblique silhouette is not a clean rim circle; plane/circle recovery is unreliable.'},
        'limitations':[p['evidence']['depth_calibration'],'Top, wall thickness, lip and ring edge rounding remain estimates.',
            'Only rigid rotational geometry is fitted; surface decoration is not rotationally symmetric.']}
    (OUT/'fit_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'candidate_input.json').write_text(json.dumps(p,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    for label,pr,color in [('Previous',old,'#999999'),('Revised',profile,'#2166ac')]:
        axs[0].plot(pr[:,0],pr[:,2],color=color,label=label+' underside')
        axs[0].plot(pr[:,0],pr[:,1],color=color,ls='--',alpha=.6)
        axs[1].plot(pr[:,0],pr[:,2],color=color,label=label)
    for j,i in enumerate(train):
        sel=views==j
        axs[0].scatter(radii[sel],medians[sel]+solution.x[j+1],s=15,label=f'View {i}, aligned medians')
    axs[0].set(xlabel='Radius (mm)',ylabel='Height (mm)',title='Underside fit; dashed = inner dish')
    axs[1].set(xlim=(38,62),ylim=(-1,10),xlabel='Radius (mm)',ylabel='Height (mm)',title='Measured foot ring; central recess estimated')
    for ax in axs:ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'profile_comparison.png',dpi=150);plt.close(fig)
    # Project actual contact-land circles to the source photos. Each model uses
    # its independently evaluated depth offset; this is an alignment diagnostic,
    # not an independent calibrated measurement of the foot.
    board=Image.new('RGB',(1280,530),'white');draw=ImageDraw.Draw(board)
    with zipfile.ZipFile(ARCHIVE) as archive:
        names=sorted(n for n in archive.namelist() if n.endswith('_rgb.png'))
        for j,i in enumerate(train):
            rgb=np.array(Image.open(io.BytesIO(archive.read(names[i]))))
            meta=json.loads(archive.read(names[i].replace('_rgb.png','.json')))
            normal=np.array(captures[i]['normal']);center=np.array(captures[i]['center_mm'])
            u=np.cross(normal,[0,1,0]);u/=np.linalg.norm(u);v=np.cross(normal,u)
            angles=np.linspace(0,2*np.pi,257)
            for label,pr,color in [('previous',old,(230,70,190)),('revised',profile,(0,210,230))]:
                ring=pr[pr[:,2]==0,0];offset=rows[j][label]['vertical_offset_mm']
                for radius in (ring.min(),ring.max()):
                    xyz=center+(40+offset)*normal+radius*(np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v)
                    pixels=project(xyz,meta)
                    cv2.polylines(rgb,[np.rint(pixels).astype('int32')],True,color,1,cv2.LINE_AA)
            board.paste(Image.fromarray(rgb),(640*j,50))
            draw.text((640*j+10,10),f'View {i}: cyan = revised contact edges; magenta = previous',fill='black')
            draw.text((640*j+10,28),'Estimated pose + depth offset; projection is a diagnostic',fill='black')
    board.save(OUT/'foot_photo_comparison.jpg')
    if write:INPUT.write_text(json.dumps(p,indent=2)+'\n')
    print(json.dumps({'body_exponent':exponent,'recess_depth_mm':recess_depth,'updated_input':write,'comparison':rows},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--reuse-inspection',action='store_true',help='Reuse saved depth samples from the same archive')
    args=parser.parse_args()
    if not args.reuse_inspection:inspect_archive()
    fit_profile(args.write)
