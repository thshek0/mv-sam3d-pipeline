"""Fit a rotational underside spline to manually reviewed photo silhouettes.

The model is in mm, inverted for the photographs. Camera poses are inferred
from the known rim and foot circles. EXIF supplies an approximate focal length.
This does not calibrate the phone lens or validate physical fixture contact.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT/'output/objects/hestia-rl/plate_harmynta_white_22cm/codex'
WORK = ASSET/'intermediate/curvature_refit_20261107'
INPUT = ROOT/'src/plates/inputs/plate_harmynta_white_22cm.json'
EDGES = ROOT/'src/plates/inputs/plate_harmynta_white_22cm_photo_edges.json'
REFERENCES = ASSET/'intermediate/print_fit_review_20261107'
KNOTS = np.array([51.25,60,70,80,90,100,106,110])
ANGLES = np.linspace(0,2*np.pi,2048,endpoint=False)


def rotation(pose):
    beta,roll=np.deg2rad(pose[:2]);c,s=np.cos(beta),np.sin(beta)
    cr,sr=np.cos(roll),np.sin(roll)
    return np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]]) @ np.array([[1,0,0],[0,c,-s],[0,s,c]])


def project(points,pose,focal=699):
    camera=points@rotation(pose).T+pose[2:5]
    return camera[:,:2]/camera[:,2,None]*focal+np.array([378,504])


def circle(radius,height,pose):
    return project(np.c_[radius*np.cos(ANGLES),radius*np.sin(ANGLES),np.full(len(ANGLES),height)],pose)


def distance(observed,predicted):
    return cKDTree(predicted).query(observed)[0]


def anchor_errors(pose,observations):
    return np.r_[distance(observations['foot'],circle(50.9,40,pose)),
                 distance(observations['rim'],circle(109,0,pose))]


def profile(values):
    return PchipInterpolator(KNOTS,np.r_[5,values,38.7])


def envelope(curve,pose,ys,side):
    """Visible silhouette of the union of projected outer-body cross-sections.

    Intersect each circle's projected filled disk with image rows, then take
    the exterior envelope. This excludes self-occluded apparent contours.
    """
    r=np.r_[np.linspace(51.25,110,500),50.9,51.13,109,109.7,110]
    h=np.r_[40-curve(r[:500]),40,39.5,0,.2,.8]
    n=rotation(pose)[:,2];centers=np.asarray(pose[2:5])+h[:,None]*n
    d=centers@n
    q=d[:,None,None]**2*np.eye(3)-d[:,None,None]*(centers[:,:,None]*n[None,None,:]+n[None,:,None]*centers[:,None,:])
    q+=(np.sum(centers*centers,axis=1)-r*r)[:,None,None]*np.outer(n,n)
    kinv=np.array([[1/699,0,-378/699],[0,1/699,-504/699],[0,0,1]])
    c=kinv.T@q@kinv
    yy=np.asarray(ys)[None,:]
    aa=c[:,0,0,None];bb=2*(c[:,0,1,None]*yy+c[:,0,2,None])
    cc=c[:,1,1,None]*yy*yy+2*c[:,1,2,None]*yy+c[:,2,2,None]
    disc=bb*bb-4*aa*cc
    valid=disc>=0;root=np.sqrt(np.maximum(disc,0))
    lo=(-bb-root)/(2*aa);hi=(-bb+root)/(2*aa)
    if side=='left':return np.min(np.where(valid,lo,np.inf),axis=0)
    return np.max(np.where(valid,hi,-np.inf),axis=0)


def wall_errors(curve,pose,obs):
    points=np.asarray(obs['wall']);side='left' if np.mean(points[:,0])<378 else 'right'
    pred=envelope(curve,pose,points[:,1],side)
    slope=(envelope(curve,pose,points[:,1]+.5,side)-envelope(curve,pose,points[:,1]-.5,side))
    return (pred-points[:,0])/np.sqrt(1+slope*slope)


def pose_fit(obs):
    foot=np.mean(obs['foot'],axis=0);best=None
    for elevation in [-10,5,20,35]:
        d=240.;a=np.deg2rad(elevation)
        tx=(foot[0]-378)*(d-40*np.sin(a))/699
        ty=(foot[1]-504)*(d-40*np.sin(a))/699+40*np.cos(a)
        init=[90+elevation,0,tx,ty,d]
        fit=least_squares(lambda p: anchor_errors(p,obs),init,
            bounds=([40,-20,-300,-200,130],[155,20,300,300,700]),
            loss='soft_l1',f_scale=2,max_nfev=350,diff_step=1e-4)
        score=np.mean(anchor_errors(fit.x,obs)**2)
        if best is None or score<best[0]:best=(score,fit.x)
    return best[1]


def make_input(before,curve,report,write):
    """Make a closed thin shell while keeping measured ring and envelope fixed."""
    from src.plates.rebuild import keep_profile
    prior=np.array(before['profile_radius_top_bottom_mm'])
    r=np.unique(np.r_[prior[:,0],KNOTS,np.arange(51.25,109,.5)])
    previous_top=PchipInterpolator(prior[:,0],prior[:,1])(r)
    previous_bottom=PchipInterpolator(prior[:,0],prior[:,2])(r)
    # Preserve the explicitly authored ring's short straight edge transitions.
    bottom=np.interp(r,prior[:,0],prior[:,2]);outer=r>=51.25
    bottom[outer]=curve(r[outer])
    # Interior wall follows the corrected underside with the previous estimated
    # vertical thickness. Blend at the foot and lip to retain the floor/rim.
    inner_blend=np.clip((r-51.25)/8.75,0,1);inner_blend=inner_blend**2*(3-2*inner_blend)
    lip_blend=np.clip((109-r)/3,0,1);lip_blend=lip_blend**2*(3-2*lip_blend)
    top=previous_top+(bottom-previous_bottom)*inner_blend*lip_blend
    candidate=np.round(np.c_[r,top,bottom],6)
    assert np.all(candidate[:,1]>candidate[:,2])
    assert top.max()==40 and bottom.min()==0 and r[-1]==110
    p=json.loads(json.dumps(before));p['profile_radius_top_bottom_mm']=candidate.tolist()
    p['profile_tolerance_mm']=.1
    ti,te=keep_profile(candidate[:,0],candidate[:,1],.1)
    bi,be=keep_profile(candidate[:,0],candidate[:,2],.1)
    # A closed revolved polygon with two axis vertices has this many triangles.
    triangles=2*p['angular_sections']*(len(ti)+len(bi)-2)
    assert triangles<=5000,triangles
    report['shell']={'inner_wall':'Tracks corrected outer wall using previous inferred thickness; blended into original floor and lip.',
        'measured_envelope_mm':[220,220,40],'predicted_visual_triangles':triangles,
        'profile_simplification_height_error_mm':max(te,be),'inner_wall_is_measured':False}
    observations=json.loads(EDGES.read_text())['images']
    simplified=lambda rr: np.interp(rr,candidate[bi,0],candidate[bi,2])
    for name,view in report['views'].items():
        err=np.abs(wall_errors(simplified,view['pose'],observations[name]))
        view['simplified_profile']={'median_px':float(np.median(err)),'p90_px':float(np.percentile(err,90)),
            'scope':'Simplified meridional profile; excludes the small angular polygon approximation.'}
    p['profile_fit']={'method':'Smooth shape-preserving cubic spline fitted to perspective photo silhouettes',
        'knots_radius_mm':report['knots_radius_mm'],'knots_bottom_mm':report['knots_bottom_mm'],
        'training_views':[n for n,v in report['views'].items() if v['role']=='fit'],
        'held_out_views':[n for n,v in report['views'].items() if v['role']=='held_out'],
        'report':str((WORK/'photo_fit_report.json').relative_to(ROOT))}
    p['status_label']='Photo-fitted underside curvature; measured foot ring'
    p['evidence']['profile']=(
        '2026-11-07 second revision: underside spline fitted to traced standalone inverted-photo silhouettes '
        'IMG_9972 and IMG_9977, with IMG_9974 and IMG_9978 held out. Perspective cameras estimated from '
        'known foot/rim circles and approximate EXIF focal length. Foot diameters 102.5/87.7 mm and '
        'external ring height 5 mm remain fixed; nominal envelope remains 220 x 40 mm. '
        'Interior wall follows revised underside at the prior estimated thickness, blending into the '
        'original floor and lip. Central recess remains the earlier 1.99 mm depth estimate. '
        'Gold-print photos support a fit mismatch but do not calibrate the shape or prove a corrected physical fit.')
    p['geometry_evidence_files']=list(dict.fromkeys(p.get('geometry_evidence_files',[])+[
        'dataset/hestia-rl/plate_harmynta_white_22cm/More plate-20261107T052054Z-1-001.zip',
        str(Path(__file__).relative_to(ROOT)),str(EDGES.relative_to(ROOT)),
        str((WORK/'photo_fit_report.json').relative_to(ROOT)),str((WORK/'before/input.json').relative_to(ROOT))]))
    (WORK/'candidate_input.json').write_text(json.dumps(p,indent=2)+'\n')
    if write:INPUT.write_text(json.dumps(p,indent=2)+'\n')
    return candidate


def run(write=False):
    WORK.mkdir(exist_ok=True)
    annotations=json.loads(EDGES.read_text());obs=annotations['images']
    before=json.loads((WORK/'before/input.json').read_text())
    prior=np.array(before['profile_radius_top_bottom_mm'])
    oldcurve=PchipInterpolator(prior[prior[:,0]>=51.25,0],prior[prior[:,0]>=51.25,2])
    poses={}
    for name,o in obs.items():
        poses[name]=pose_fit(o)
        print(name,'anchor median px',np.median(anchor_errors(poses[name],o)),'pose',poses[name],flush=True)
    train=[n for n in obs if obs[n]['role']=='fit']
    initial=np.interp(KNOTS[1:-1],prior[:,0],prior[:,2])
    x0=np.r_[initial,*[poses[n] for n in train]]
    def residual(x):
        curve=profile(x[:6]);errors=[]
        for j,n in enumerate(train):
            pose=x[6+5*j:11+5*j]
            errors.extend(anchor_errors(pose,obs[n])/1.5)
            errors.extend(wall_errors(curve,pose,obs[n])/1.5)
        slopes=np.diff(np.r_[5,x[:6],38.7])/np.diff(KNOTS)
        errors.extend(np.minimum(slopes,0)*100)
        errors.extend(np.diff(slopes)*5.0)
        return np.array(errors)
    lower=np.r_[[5.05,5.1,5.2,10,20,28],np.tile([40,-20,-300,-200,130],len(train))]
    upper=np.r_[[19,27,33,37,38,38.6],np.tile([155,20,300,300,700],len(train))]
    fit=least_squares(residual,x0,bounds=(lower,upper),loss='soft_l1',f_scale=2,max_nfev=450,diff_step=1e-4)
    curve=profile(fit.x[:6])
    for j,n in enumerate(train):poses[n]=fit.x[6+5*j:11+5*j]
    assert np.all(np.diff(np.r_[5,fit.x[:6],38.7])>0),'Underside must rise monotonically outside the foot'
    report={'knots_radius_mm':KNOTS.tolist(),'knots_bottom_mm':np.r_[5,fit.x[:6],38.7].tolist(),
            'optimization_success':bool(fit.success),'evaluations':fit.nfev,'views':{},
            'limitations':['Hand-traced silhouettes and approximate EXIF camera intrinsics.',
                'Fit cameras use foot and rim circles; held-out cameras use circle anchors only.',
                'No fixture CAD or confirmed gap reading; physical seating is not validated.']}
    board=Image.new('RGB',(1512,1116),'white');draw=ImageDraw.Draw(board)
    for j,(n,o) in enumerate(obs.items()):
        pose=poses[n]
        metrics={}
        for label,c in [('previous',oldcurve),('revised',curve)]:
            er=np.abs(wall_errors(c,pose,o))
            metrics[label]={'median_px':float(np.median(er)),'p90_px':float(np.percentile(er,90))}
        report['views'][n]={'role':o['role'],'pose':pose.tolist(),'anchor_median_px':float(np.median(anchor_errors(pose,o))),**metrics}
        im=np.array(Image.open(REFERENCES/(n+'.jpg')).resize((756,1008)))
        for label,c,color in [('previous',oldcurve,(230,140,30)),('revised',curve,(0,180,240))]:
            yy=np.linspace(360,890,1400)
            for side in ['left','right']:
                xx=envelope(c,pose,yy,side);valid=np.isfinite(xx)&(xx>-500)&(xx<1256)
                pts=np.c_[xx[valid],yy[valid]]
                for p in pts:cv2.circle(im,tuple(np.rint(p).astype(int)),1,color,-1,cv2.LINE_AA)
        for key in ['foot','rim','wall']:
            for p in o[key]:cv2.circle(im,tuple(p),3,(255,40,110),1,cv2.LINE_AA)
        for radius,h in [(50.9,40),(109,0)]:
            pts=circle(radius,h,pose);cv2.polylines(im,[np.rint(pts).astype('int32')],True,(60,220,60),1,cv2.LINE_AA)
        xx=j%2*756;yy=j//2*558
        board.paste(Image.fromarray(im).crop((0,360,756,890)),(xx,yy+28))
        draw.text((xx+10,yy+5),f'{n} {o["role"]}: orange=before, cyan=new; pink=traced; green=pose anchors',fill='black')
    board.save(WORK/'silhouette_comparison.jpg',quality=95)
    candidate=make_input(before,curve,report,write)
    (WORK/'photo_fit_report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(WORK/'curve_fit.npz',radius=KNOTS,bottom=np.r_[5,fit.x[:6],38.7])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
    original=np.array(json.loads((ASSET/'intermediate/refit_20261107/before/input.json').read_text())['profile_radius_top_bottom_mm'])
    for pr,label,color in [(original,'Original','#969696'),(prior,'Previous revision','#d88922'),(candidate,'New photo fit','#1674b8')]:
        x=np.r_[-pr[:0:-1,0],pr[:,0]]
        t=np.r_[pr[:0:-1,1],pr[:,1]];b=np.r_[pr[:0:-1,2],pr[:,2]]
        axs[0].plot(x,b,color=color,label=label);axs[0].plot(x,t,color=color,ls='--',alpha=.65)
        axs[1].plot(pr[:,0],pr[:,2],color=color,label=label)
    axs[0].set(xlim=(-113,113),ylim=(-3,43),title='Full authored cross-section (dashed = inside)',xlabel='Distance from centre (mm)',ylabel='Height (mm)')
    axs[0].set_aspect('equal',adjustable='box')
    axs[1].set(xlim=(45,110),ylim=(-1,41),title='Underside curvature detail',xlabel='Radius (mm)',ylabel='Height (mm)')
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(WORK/'cross_section_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true')
    args=parser.parse_args();run(args.write)
