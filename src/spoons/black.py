"""Reproduce the user-preferred spoon from explicit permanent mesh sources.

Historically SAM3D-derived; no SAM3D directory, inference or weights are needed.
The rejected procedural loft must not replace this accepted shape implicitly.
"""
from pathlib import Path
import json
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from src.export import sha256
from src.products_mujoco import load_parts

SOURCE=Path(__file__).resolve().parents[2]/'dataset/hestia-rl/spoon_home_black/accepted_geometry'
VISUAL=SOURCE/'spoon_home_black_m.glb'
COLLISION=SOURCE/'spoon_home_black_collision_m.glb'


def sources():
    return [Path(__file__),VISUAL,COLLISION,SOURCE/'provenance.json']


def build(p):
    from .build import ray_scene, heights
    provenance=json.loads((SOURCE/'provenance.json').read_text())
    for name,digest in provenance['files'].items():
        if sha256(SOURCE/name)!=digest:raise ValueError('Accepted source changed; review and update provenance first: '+name)
    mesh=trimesh.load(VISUAL,force='mesh');parts=load_parts(COLLISION)
    cy=p['support_center_y_mm']/1000;center=np.array([0,cy])
    eq=ConvexHull(mesh.vertices[mesh.vertices[:,1]>.052,:2]).equations
    angles=np.arange(32)*2*np.pi/32;directions=np.c_[np.cos(angles),np.sin(angles)];den=directions@eq[:,:2].T
    t=np.where(den>1e-10,-(center@eq[:,:2].T+eq[:,2])/den,np.inf).min(1)*.995
    edge=center+directions*t[:,None];q=np.concatenate([center+r*(edge-center) for r in (.1,.3,.5,.7)])
    top,_=heights(ray_scene(mesh),q)
    return mesh,parts,{'construction':'User-preferred previous geometry restored exactly: smoothed neck, original body thickness and straight transverse support bar.',
        'provenance':provenance['origin'],'source_sha256':provenance['files'],
        'remaining_shape_issue':'Handle width profile is the previous accepted version; a further width-only edit is deferred after the rejected full-loft redesign.',
        'cavity_probes':np.c_[q,top+.004]}
