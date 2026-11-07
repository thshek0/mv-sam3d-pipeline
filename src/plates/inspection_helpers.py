"""Shared mesh and capture-inspection helpers for the consolidated plates workflow."""
from pathlib import Path
import sys,os,json,subprocess,numpy as np,cv2,trimesh
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
os.environ['OMP_NUM_THREADS']='4'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from post import _clean,to_o3d,preview_mesh
import open3d as o3d
cv2.setNumThreads(4)

def audit(m):
 counts=np.bincount(m.edges_unique_inverse);om=to_o3d(m)
 return {'vertices':len(m.vertices),'triangles':len(m.faces),'boundary_edges':int((counts==1).sum()),'nonmanifold_edges':int((counts>2).sum()),'watertight':bool(m.is_watertight),'winding_consistent':bool(m.is_winding_consistent),'positive_volume':bool(m.is_volume),'components':len(m.split(only_watertight=False)),'self_intersecting_triangle_pairs':len(om.get_self_intersecting_triangles()) if len(m.faces)<=50000 else None,'bounds_mm':(m.bounds*1000).tolist(),'extents_mm':(m.extents*1000).tolist()}

def ray_scene(m):
 s=o3d.t.geometry.RaycastingScene();s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(to_o3d(m)));return s

def tags(rgb):
 p=cv2.aruco.DetectorParameters();p.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_APRILTAG
 d=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),p)
 cs,ids,_=d.detectMarkers(rgb)
 return {} if ids is None else {(int(t),j):xy for t,c in zip(ids.ravel(),cs) for j,xy in enumerate(c.reshape(4,2))}

def unproject(uv,z,k):return np.column_stack([(uv[:,0]-k[0,2])*z/k[0,0],(uv[:,1]-k[1,2])*z/k[1,1],z])

def reference(out):
 npz=np.load(out/'depth/sensor.npz');rgbs=[np.array(Image.open(p)) for p in sorted((out/'scene/images').glob('*.png'),key=lambda p:int(p.stem))];masks=[np.array(Image.open(out/'scene/masks'/f'{i}.png'))[:,:,3]>0 for i in range(len(rgbs))]
 k=npz['intrinsics'][0];z=npz['depth'][0];dets=[tags(rgb) for rgb in rgbs];landmarks={}
 for key,uv in dets[0].items():
  x,y=np.rint(uv).astype(int);patch=z[max(0,y-2):y+3,max(0,x-2):x+3];valid=patch[patch>0]
  if len(valid):landmarks[key]=unproject(uv[None],np.array([np.median(valid)]),k)[0]
 poses=[np.eye(4)];pose_reports=[{'view':0,'reference':True}]
 for i in range(1,len(rgbs)):
  common=sorted(landmarks.keys()&dets[i].keys());pose=None;rep={'view':i,'common_tag_corners':len(common)}
  if len(common)>=6:
   xyz=np.array([landmarks[c] for c in common]);uv=np.array([dets[i][c] for c in common]);ok,r,t,inliers=cv2.solvePnPRansac(xyz,uv,npz['intrinsics'][i],None,iterationsCount=300,reprojectionError=5,flags=cv2.SOLVEPNP_EPNP)
   if ok and inliers is not None and len(inliers)>=6:
    ii=inliers.ravel();r,t=cv2.solvePnPRefineLM(xyz[ii],uv[ii],npz['intrinsics'][i],None,r,t);proj=cv2.projectPoints(xyz,r,t,npz['intrinsics'][i],None)[0].reshape(-1,2);err=np.linalg.norm(proj-uv,axis=1)
    rep.update(reprojection_median_px=float(np.median(err)),reprojection_p90_px=float(np.percentile(err,90)),inliers=len(ii))
    if np.median(err)<4 and np.percentile(err,90)<8:
     pose=np.eye(4);pose[:3,:3]=cv2.Rodrigues(r)[0];pose[:3,3]=t.ravel()
  rep['accepted_for_comparison']=pose is not None;poses.append(pose);pose_reports.append(rep)
 return npz,rgbs,masks,poses,pose_reports

def render_compare(mesh,npz,rgbs,masks,poses,out,label):
 scene=ray_scene(mesh);stats=[];tiles=[]
 for i,(rgb,mask,pose) in enumerate(zip(rgbs,masks,poses)):
  if pose is None:continue
  k=npz['intrinsics'][i].astype(float);rays=o3d.t.geometry.RaycastingScene.create_rays_pinhole(k,pose,504,504)
  z=scene.cast_rays(rays)['t_hit'].numpy();hit=np.isfinite(z);valid=hit&mask&(npz['depth'][i]>0)
  er=np.abs(z[valid]-npz['depth'][i][valid])*1000
  stat={'view':i,'silhouette_iou':float((hit&mask).sum()/max(1,(hit|mask).sum())),'observed_mask_coverage':float((hit&mask).sum()/mask.sum()),'depth_comparison_pixels':int(valid.sum()),'depth_median_error_mm':float(np.median(er)) if len(er) else None,'depth_p90_error_mm':float(np.percentile(er,90)) if len(er) else None};stats.append(stat)
  im=rgb.copy();im[mask&~hit]=[240,60,40];im[hit&~mask]=[0,180,255];over=hit&mask;im[over]=(im[over]*.6+np.array([40,210,80])*.4).astype('uint8');tile=Image.fromarray(im);tile.thumbnail((252,252));tiles.append((tile,stat))
 board=Image.new('RGB',(1008,285*((len(tiles)+3)//4)),'white');d=ImageDraw.Draw(board)
 for j,(tile,st) in enumerate(tiles):x=j%4*252;y=j//4*285;board.paste(tile,(x,y+30));d.text((x+5,y+5),f"View {st['view']+1} IoU {st['silhouette_iou']:.2f}",fill='black')
 board.save(out/f'{label}_comparison.jpg')
 return stats
