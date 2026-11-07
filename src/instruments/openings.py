"""Instrument openings: reusable implementation behind the instruments CLI."""
import numpy as np
import trimesh


def circular_holes_plate(lo,hi,holes,z0,z1):
 """Convex grid cells and annular sectors around nonoverlapping circular bores."""
 boxes=[(x-d/2,y-d/2,x+d/2,y+d/2) for x,y,d in holes]
 for i,box in enumerate(boxes):
  assert lo[0]<box[0]<box[2]<hi[0] and lo[1]<box[1]<box[3]<hi[1], 'Hole outside plate'
  for other in boxes[:i]:
   assert box[2]<=other[0] or other[2]<=box[0] or box[3]<=other[1] or other[3]<=box[1], 'Hole bounding boxes overlap'
 xs=sorted({lo[0],hi[0],*(v for box in boxes for v in (box[0],box[2]))})
 ys=sorted({lo[1],hi[1],*(v for box in boxes for v in (box[1],box[3]))})
 rectangles=[]
 for ya,yb in zip(ys,ys[1:]):
  start=None
  for xa,xb in zip(xs,xs[1:]):
   x,y=(xa+xb)/2,(ya+yb)/2
   keep=not any(a<x<c and b<y<d for a,b,c,d in boxes)
   if keep and start is None:start=xa
   if not keep and start is not None:rectangles.append([start,ya,xa,yb]);start=None
  if start is not None:rectangles.append([start,ya,xs[-1],yb])
 merged=[]
 for rectangle in rectangles:
  previous=next((q for q in merged if q[0]==rectangle[0] and q[2]==rectangle[2] and q[3]==rectangle[1]),None)
  if previous is None:merged.append(rectangle)
  else:previous[3]=rectangle[3]
 parts=[]
 for xa,ya,xb,yb in merged:
  mesh=trimesh.creation.box([xb-xa,yb-ya,z1-z0]);mesh.apply_translation([(xa+xb)/2,(ya+yb)/2,(z0+z1)/2]);parts.append(mesh)
 for (x,y,d),box in zip(holes,boxes):parts+=circle_plate(box[:2],box[2:],(x,y),d/2,z0,z1)
 return parts

def circle_plate(lo,hi,center,radius,z0,z1,segments=32):
 """Rectangle minus a polygonal circular aperture, decomposed into convex sectors."""
 cx,cy=center;x0,y0=lo;x1,y1=hi
 angles=list(np.linspace(0,2*np.pi,segments,endpoint=False))
 angles += [np.arctan2(y-cy,x-cx)%(2*np.pi) for x,y in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]]
 angles=sorted(set(round(a,12) for a in angles));result=[]
 for i,a in enumerate(angles):
  b=angles[(i+1)%len(angles)]
  points=[]
  for angle in [a,b]:
   dx,dy=np.cos(angle),np.sin(angle)
   tx=((x1 if dx>0 else x0)-cx)/dx if abs(dx)>1e-10 else np.inf
   ty=((y1 if dy>0 else y0)-cy)/dy if abs(dy)>1e-10 else np.inf
   outer=min(tx,ty)
   for r in (radius,outer):points.append([cx+r*dx,cy+r*dy])
  if np.linalg.matrix_rank(np.asarray(points)-np.asarray(points).mean(0),tol=1e-10)<2:continue
  vertices=[[x,y,z] for x,y in points for z in (z0,z1)]
  mesh=trimesh.convex.convex_hull(vertices)
  if mesh.volume>1e-12:result.append(mesh)
 return result

def perforated_plate(lo,hi,z0,z1,circle,slots):
 """Grid-decompose rectangular slots plus one circle; merge plain grid cells."""
 cx,cy,r=circle
 excluded=[*slots,(cx-r,cy-r,cx+r,cy+r)]
 xs=sorted(set([lo[0],hi[0]]+[v for q in excluded for v in (q[0],q[2])]))
 ys=sorted(set([lo[1],hi[1]]+[v for q in excluded for v in (q[1],q[3])]))
 rectangles=[]
 for ya,yb in zip(ys,ys[1:]):
  start=None
  for xa,xb in zip(xs,xs[1:]):
   x,y=(xa+xb)/2,(ya+yb)/2
   keep=lo[0]<=x<=hi[0] and lo[1]<=y<=hi[1] and not any(q[0]<x<q[2] and q[1]<y<q[3] for q in excluded)
   if keep and start is None:start=xa
   if not keep and start is not None:rectangles.append([start,ya,xa,yb]);start=None
  if start is not None:rectangles.append([start,ya,xs[-1],yb])
 merged=[]
 for q in rectangles:
  prev=next((p for p in merged if p[0]==q[0] and p[2]==q[2] and p[3]==q[1]),None)
  if prev is None:merged.append(q)
  else:prev[3]=q[3]
 result=[]
 for xa,ya,xb,yb in merged:
  mesh=trimesh.creation.box([xb-xa,yb-ya,z1-z0]);mesh.apply_translation([(xa+xb)/2,(ya+yb)/2,(z0+z1)/2]);result.append(mesh)
 result += circle_plate((cx-r,cy-r),(cx+r,cy+r),(cx,cy),r,z0,z1)
 return result
