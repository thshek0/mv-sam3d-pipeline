"""Shared authoring changes must reach actual openings, motion and metadata."""
import numpy as np
import pytest

from src.instruments.cary60 import model,parameters,definition,decal_bounds
from src.instruments.shared_geometry import read_parameter_include, convex_difference


def test_subtractive_cavity_has_empty_space_and_nonoverlapping_volume():
    pieces=convex_difference(['block',[[0,0,0],[20,20,20]]], [
        ['block',[[4,4,10],[16,16,21]]],
        ['block',[[8,4,10],[18,16,21]]]])
    # Overlapping cutters must remove their union, not count the overlap twice.
    np.testing.assert_allclose(sum(q.volume for q in pieces),8000-(1440+1200-960))
    assert all(q.is_volume and q.is_convex for q in pieces)
    def contains(q,point):
        return np.all(np.einsum('ij,ij->i',point-q.triangles_center,q.face_normals)<1e-8)
    assert not any(contains(q,np.array([10,10,15])) for q in pieces)
    assert any(contains(q,np.array([10,10,5])) for q in pieces)


def source_parameters():
    return {'model':'cary60','body_wdh_m':[.477,.567,.196],
            'body_dimension_evidence':'Published nominal body envelope','status':'test',
            'lid_preview_stroke_m':.230,
            'detail_parameters':{'holder_center_xy_m':[.120,.122],
                'moving_cover_y_m':[.012,.305],'fixed_cover_y_m':[.305,.535],
                'cover_thickness_m':.003,'holder_plate_wh_m':[.105,.045]}}


def test_edited_opening_and_cover_reach_mesh_and_joint_metadata():
    p=source_parameters();v,_=parameters(p)
    p['design_parameters']=dict(v,socket_inner_width_mm=16,socket_inner_depth_mm=18,
                                chamber_center_x_mm=125,cover_travel_mm=200,body_width_mm=497)
    m=model(p)
    assert p['joint_axis']==[0,1,0]
    assert p['lid_preview_stroke_m']==.2
    assert p['body_wdh_m'][0]==.497
    socket=[q for n,q in m.parts.items() if n.startswith('socket_shell__')]
    expected=(v['socket_outer_width_mm']*v['socket_outer_depth_mm']-16*18)*v['socket_height_mm']
    np.testing.assert_allclose(sum(q.volume for q in socket)*1e9,expected)
    center=np.array([.100,.122,.130])
    assert not any(np.all(np.einsum('ij,ij->i',center-q.triangles_center,q.face_normals)<=1e-10) for q in socket)
    assert max(q.bounds[1,0] for q in socket)<.125
    assert m.parts['holder_adjuster_knob'].bounds[0,0]>.125
    for name,q in m.parts.items():
        if name.startswith('lid_shell__'):
            np.testing.assert_allclose(m.world_parts(.2)[name].bounds-q.bounds,[[0,.2,0]]*2)
    assert all(q.is_convex and q.is_volume for q in m.parts.values())
    # Even after updating the derived display fields, the canonical parameters persist.
    again,_=parameters(p);assert again==p['design_parameters']


@pytest.mark.parametrize('bad',[
    'body_width_mm = 1/0;', 'body_width_mm = nan;', 'body_width_mm = 1e999;',
    'unknown = 1;', 'body_width_mm = 10; body_width_mm = 20;',
    'body_width_mm = __import__("os");', 'include <other.scad>;', '',
])
def test_parameter_import_rejects_code_unknown_or_incomplete_values(tmp_path,bad):
    path=tmp_path/'parameters.scad';path.write_text(bad)
    with pytest.raises(ValueError):read_parameter_include(path,{'body_width_mm':477})


def test_parameter_include_roundtrip_and_invalid_geometry(tmp_path):
    v,_=parameters(source_parameters());path=tmp_path/'parameters.scad'
    path.write_text('// numeric dimensions\n'+'\n'.join(f'{k} = {value};' for k,value in v.items()))
    assert read_parameter_include(path,v)==v
    with pytest.raises(ValueError,match='retract'):definition(dict(v,cover_travel_mm=500))
    with pytest.raises(ValueError,match='socket width'):definition(dict(v,socket_inner_width_mm=30))


def test_photo_label_follows_its_housing():
    v,_=parameters(source_parameters());old=decal_bounds(v)
    np.testing.assert_allclose(old,[215,330,106,147])
    changed=dict(v,body_width_mm=517,body_height_mm=216,chamber_center_x_mm=130)
    x0,x1,z0,z1=decal_bounds(changed)
    assert 205<x0<x1<517 and 0<z0<z1<216
    assert x0>old[0] and z1>old[3]


def test_cary_grip_has_two_rounded_and_two_square_corners():
    from src.instruments.shared_geometry import resolve
    v,_=parameters(source_parameters())
    part=next(p for p in definition(v).parts if p.name=='lid_shell')
    polygon=np.array(resolve(part.args[1][0][1][0],v))
    lo,hi=polygon.min(0),polygon.max(0)
    def inside(point):
        edges=np.roll(polygon,-1,axis=0)-polygon
        delta=point-polygon
        return np.all(edges[:,0]*delta[:,1]-edges[:,1]*delta[:,0]>=-1e-7)
    assert inside(lo+[1,1]) and inside(hi-[1,1])
    assert not inside([lo[0]+1,hi[1]-1])
    assert not inside([hi[0]-1,lo[1]+1])
