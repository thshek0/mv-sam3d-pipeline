from pathlib import Path
import sys,json,numpy as np,cv2
from scipy.optimize import least_squares
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from inspection_helpers import tags,unproject
cv2.setNumThreads(4);rng=np.random.default_rng(48)

def fitplane(pts):
 if len(pts)>3500:pts=pts[rng.choice(len(pts),3500,replace=False)]
 best=None;score=0
 for j in range(250):
  p=pts[rng.choice(len(pts),3,replace=False)];n=np.cross(p[1]-p[0],p[2]-p[0]);ln=np.linalg.norm(n)
  if ln<1e-10:continue
  n/=ln;d=-p[0]@n;inside=np.abs(pts@n+d)<.002
  if inside.sum()>score:best=inside;score=inside.sum()
 if best is None:raise ValueError('no plane')
 for _ in range(3):
  center=pts[best].mean(0);_,_,vh=np.linalg.svd(pts[best]-center,full_matrices=False);n=vh[-1];d=-center@n;best=np.abs(pts@n+d)<.0025
 if n@center>0:n=-n;d=-d
 return n,d,float(best.mean()),float(np.median(np.abs(pts[best]@n+d)))*1000

results={}
for name in ['plate_maehata_pop_blue_23cm','plate_strimmig_pale_grey_green_21cm','plate_paradisisk_off_white_20cm']:
 out=ROOT/'work/hestia-rl'/name/'codex';data=np.load(out/'depth/sensor.npz');records=[]
 for i,z in enumerate(data['depth']):
  rgb=np.array(Image.open(out/'scene/images'/f'{i}.png'));mask=np.array(Image.open(out/'scene/masks'/f'{i}.png'))[:,:,3]>0;k=data['intrinsics'][i]
  inside=cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,5);outside=cv2.distanceTransform((~mask).astype('uint8'),cv2.DIST_L2,5)
  tm=np.zeros(mask.shape,np.uint8);det=tags(rgb)
  for tid in set(t for t,j in det):
   p=np.array([det[(tid,j)] for j in range(4)]);p=p.mean(0)+(p-p.mean(0))*1.3;cv2.fillConvexPoly(tm,np.rint(p).astype('int32'),255)
  ring=(inside>=4)&(inside<=11)&(z>0)&(tm==0);collar=(outside>=12)&(outside<=28)&(z>0)
  if ring.sum()<100 or collar.sum()<150:records.append({'view':i,'accepted':False,'reason':'insufficient measured rim/table pixels'});continue
  y,x=np.nonzero(ring);rp=unproject(np.c_[x,y],z[y,x],k);n,d,frac,mad=fitplane(rp)
  y,x=np.nonzero(collar);tp=unproject(np.c_[x,y],z[y,x],k);tn,td,tfrac,tmad=fitplane(tp)
  contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);contour=max(contours,key=cv2.contourArea).reshape(-1,2);rays=unproject(contour,np.ones(len(contour)),k);zz=-d/(rays@n);xyz=rays*zz[:,None]
  axis=np.cross(n,[0,1,0]);axis/=np.linalg.norm(axis);axis2=np.cross(n,axis);xy=np.c_[xyz@axis,xyz@axis2];initial=np.r_[xy.mean(0),np.linalg.norm(xy-xy.mean(0),axis=1).mean()]
  fit=least_squares(lambda p:np.linalg.norm(xy-p[:2],axis=1)-p[2],initial,loss='soft_l1',f_scale=.001)
  residual=np.abs(np.linalg.norm(xy-fit.x[:2],axis=1)-fit.x[2])*1000
  angle=float(np.rad2deg(np.arccos(np.clip(n@tn,-1,1))));height=(xyz@tn+td)*1000
  # Inner rim band is slightly below the outer lip; expose this systematic limitation.
  accepted=frac>.5 and tfrac>.5 and np.median(residual)<2.5 and angle<8 and 0<np.median(height)<80
  records.append({'view':i,'accepted':bool(accepted),'diameter_mm':float(2*fit.x[2]*1000),'rim_height_above_table_mm':float(np.median(height)),'rim_plane_inlier_fraction':frac,'table_plane_inlier_fraction':tfrac,'rim_plane_median_residual_mm':mad,'table_plane_median_residual_mm':tmad,'circle_median_residual_mm':float(np.median(residual)),'plane_angle_deg':angle,'rim_plane_camera':{'normal':n.tolist(),'offset_m':d},'table_plane_camera':{'normal':tn.tolist(),'offset_m':td}})
 accepted=[r for r in records if r['accepted']]
 dims=np.array([[r['diameter_mm'],r['rim_height_above_table_mm']] for r in accepted])
 summary={'accepted_views':len(accepted),'total_views':len(records),'status':'CAPTURE_DERIVED_ESTIMATE_NOT_VERIFIED_PHYSICAL_MEASUREMENT','method':'Fit measured rim/table planes, intersect silhouette rays with rim plane, then robust circle fit. Independent of generated mesh scale.','limitations':['Rim plane uses a band inside the outer edge and can underestimate true lip height.','Masked silhouette can miss a thin sidewall. Glossy depth, calibration and plane-fit error remain.','No supplied caliper measurements or certified dimensional tolerance.','Median/range across views are repeatability diagnostics, not absolute accuracy bounds.']}
 if len(dims):summary.update(median_diameter_mm=float(np.median(dims[:,0])),diameter_p10_p90_mm=np.percentile(dims[:,0],[10,90]).tolist(),median_rim_height_mm=float(np.median(dims[:,1])),rim_height_p10_p90_mm=np.percentile(dims[:,1],[10,90]).tolist())
 results[name]={'summary':summary,'views':records};print(name,summary,flush=True)
target=ROOT/'output/comparisons/plates/dimension_audit.json';target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(results,indent=2))
