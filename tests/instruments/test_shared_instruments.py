"""Dimension and joint changes must propagate into actual instrument geometry."""
import numpy as np
import pytest
from src.instruments import design
from src.instruments.shared_geometry import read_parameter_include

def source(name):
 return dict(model=name,detail_parameters={},status='test',body_dimension_evidence='test')

def edited(name,**changes):
 p=source(name);v,_=design.parameters(p);v.update(changes);p['design_parameters']=v
 return design.build(p),v

def test_tau_opening_depth_center_and_hinge_follow_parameters():
 m,v=edited('c16361',clear_width_mm=210,hole_diameter_mm=12,chamber_depth_mm=50,overall_height_mm=263.3,open_angle_deg=80)
 pieces=[q for n,q in m.parts.items() if n.startswith('perforated_floor__')]
 vertices=np.vstack([q.vertices for q in pieces])*1000
 np.testing.assert_allclose(np.ptp(vertices[:,0]),210)
 np.testing.assert_allclose(vertices[:,2].max(),263.3-v['lid_height_mm']-50)
 center=np.array([v['chamber_center_x_mm'],(-v['front_projection_mm']+v['wall_mm']+v['chamber_rear_y_mm'])/2])
 radii=np.linalg.norm(vertices[:,:2]-center,axis=1)
 np.testing.assert_allclose(radii.min(),6,atol=1e-7)
 assert np.all(radii>=6-1e-7)
 pivot=np.r_[m.p['joint_origin_m'],1]
 np.testing.assert_allclose(m.transform(np.deg2rad(80))@pivot,pivot)
 assert m.p['joint_axis']==[-1,0,0]
 assert not any(name.startswith('floor_fastener_') for name in m.parts)
 for angle in np.arange(8)*np.pi/4:
  xy=center+v['floor_hole_pattern_radius_mm']*np.array([np.cos(angle),np.sin(angle)])
  point=np.r_[xy,263.3-v['lid_height_mm']-50-v['floor_thickness_mm']/2]/1000
  assert not any(np.all(np.einsum('ij,ij->i',point-q.triangles_center,q.face_normals)<=1e-10) for q in pieces)
 expected=(210*(v['chamber_rear_y_mm']+v['front_projection_mm']-v['wall_mm'])-np.pi*(6**2+8*(v['floor_small_hole_diameter_mm']/2)**2))*v['floor_thickness_mm']
 np.testing.assert_allclose(sum(q.volume for q in pieces)*1e9,expected,rtol=1e-4)

def test_orientation_motor_moves_with_wall_and_hole_remains_centered():
 m,v=edited('c13472',body_width_mm=665,holder_hole_diameter_mm=48,cover_travel_fraction=.30)
 wall=665*v['inner_wall_fraction']/1000
 np.testing.assert_allclose(m.parts['side_motor_body'].bounds[1,0],wall-v['motor_wall_inset_mm']/1000)
 np.testing.assert_allclose(m.parts['side_wall_mount'].bounds[1,0],wall)
 assert m.parts['side_motor_body'].bounds[1,0]>m.parts['side_wall_mount'].bounds[0,0]
 center=np.array([665*v['stage_center_x_fraction'],v['body_depth_mm']*v['stage_center_y_fraction']])
 vertices=np.vstack([q.vertices for n,q in m.parts.items() if n.startswith('holder_frame__')])*1000
 np.testing.assert_allclose(np.linalg.norm(vertices[:,:2]-center,axis=1).min(),24,atol=1e-7)
 for name,q in m.parts.items():
  if name.startswith('cover_shell__'):
   np.testing.assert_allclose(m.world_parts(.1995)[name].bounds-q.bounds,[[.1995,0,0]]*2)

def test_qy_room_pins_selector_and_nested_handle_frames():
 m,v=edited('c13534',door_width_mm=230,lower_clear_width_mm=135,lower_pin_spacing_mm=44,door_open_angle_deg=85,selector_travel_mm=35)
 np.testing.assert_allclose(m.parts['lower_right_infill'].bounds[0,0]-m.parts['lower_left_infill'].bounds[1,0],.135)
 pin_centers=[m.parts[n].bounds.mean(0)[0] for n in ('lower_small_pin','lower_second_pin')]
 np.testing.assert_allclose(pin_centers,[.115-.022,.115+.022])
 assert m.links['upper_handle_pop']['parent']=='upper_handle_turn'
 assert m.links['upper_handle_turn']['parent']=='upper_door'
 assert m.links['upper_handle_turn']['axis']==[0,-1,0]
 assert m.demo_states['open']['upper_door']==85
 np.testing.assert_allclose(m.demo_states['open']['mode_selector'],.035)
 before=m.transforms({})['mode_selector'];after=m.transforms({'mode_selector':.035})['mode_selector']
 np.testing.assert_allclose(after[:3,3]-before[:3,3],[0,-.035,0])
 # The generated turn frame stays anchored to the corresponding wider door.
 turn=m.transforms({})['upper_handle_turn']
 np.testing.assert_allclose(turn[0,3],.230*v['latch_x_fraction'])
 # Inner black panels must travel with their respective door, not the fixed chamber.
 for label in ('upper','lower'):
  panels=[n for n in m.parts if n.startswith(label+'_inner_lining__')]
  assert panels
  closed=m.scene({});opened=m.scene({label+'_door':85})
  for name in panels:
   assert m.info[name]['group']==label+'_door'
   assert np.mean(m.parts[name].visual.vertex_colors[:,:3])<40
   before,_=closed.graph[name];after,_=opened.graph[name]
   assert not np.allclose(before,after)
   np.testing.assert_allclose(after,m.transforms({label+'_door':85})[label+'_door'])

@pytest.mark.parametrize('name',['c16361','c13472','c13534'])
def test_all_parameter_schemas_roundtrip_numeric_scad(name,tmp_path):
 v,_=design.parameters(source(name));path=tmp_path/'parameters.scad'
 path.write_text('\n'.join(f'{key} = {value};' for key,value in v.items()))
 assert read_parameter_include(path,v)==v

@pytest.mark.parametrize('name,change',[
 ('c16361',{'hole_diameter_mm':100}),('c13472',{'cover_travel_fraction':.8}),
 ('c13534',{'lower_clear_width_mm':400}),('c13534',{'front_x_fraction_3':.01})])
def test_impossible_parameter_edits_are_rejected(name,change):
 with pytest.raises(ValueError):edited(name,**change)

@pytest.mark.parametrize('name',['cary60','c16361','c13472','c13534'])
def test_bootstrap_inputs_build_geometry_without_previous_outputs(name):
 from src.instruments.inputs import initial_parameters,evidence
 p=initial_parameters(name)
 m=design.build(p)
 assert m.parts
 assert all(part.is_volume and part.is_convex for part in m.parts.values())
 assert evidence(name)['model']==name
 # A caller's edits must not mutate the saved defaults for later builds.
 p['design_parameters']['body_width_mm']+=1
 assert initial_parameters(name)['design_parameters']['body_width_mm']!=p['design_parameters']['body_width_mm']

def test_evidence_rejects_invalid_crop_and_external_review_path(tmp_path,monkeypatch):
 import json
 from src.instruments import inputs
 monkeypatch.setattr(inputs,'DIRECTORY',tmp_path)
 folder=tmp_path/'test';folder.mkdir()
 config={'schema_version':1,'model':'test','materials':{'white':{'review_image':'panel.jpg','crop_pixels':[10,0,5,20]}}}
 (folder/'evidence.json').write_text(json.dumps(config))
 with pytest.raises(ValueError,match='crop'):inputs.evidence('test')
 config['materials']['white'].update(review_image='../panel.jpg',crop_pixels=[0,0,5,20])
 (folder/'evidence.json').write_text(json.dumps(config))
 with pytest.raises(ValueError,match='relative'):inputs.evidence('test')
