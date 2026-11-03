"""QY shared solids, seven moving joints and observed demonstration states."""
from .shared_geometry import Assembly,Expr,resolve
from .geometry import WHITE,DARK,BLACK,METAL

CONSTRAINTS=[
 ('door_width_mm > lower_clear_width_mm+12 and door_width_mm+strip_width_mm < body_width_mm','Invalid tower/room widths'),
 ('base_height_mm+lower_door_height_mm+2*door_clearance_mm < body_height_mm-upper_door_height_mm','Door heights overlap'),
 ('lower_recess_diameter_mm > 0 and lower_recess_diameter_mm < lower_clear_width_mm','Invalid lower recess'),
 ('upper_dish_inner_bottom_diameter_mm > 0 and upper_dish_inner_top_diameter_mm < upper_dish_outer_diameter_mm','Invalid upper dish'),
 ('lower_pin_spacing_mm+small_pin_diameter_mm < lower_clear_width_mm','Pins exceed chamber width'),
 ('latch_x_fraction >= front_x_fraction_5 and latch_x_fraction < 1','Latch outside modeled front segment'),
 ('door_open_angle_deg > 0 and door_open_angle_deg <= 100 and selector_travel_mm > 0','Invalid demonstration motion'),
 ('front_x_fraction_0 == 0 and front_x_fraction_6 == 1','Profile endpoints must meet door sides'),
]+[(f'front_x_fraction_{i+1} > front_x_fraction_{i}','Front profile X must increase') for i in range(6)]

def definition(values):
 p={k:Expr(k) for k in values};W=p['body_width_mm'];D=p['body_depth_mm'];H=p['body_height_mm'];dw=p['door_width_mm'];tw=dw+p['strip_width_mm']
 bh=p['base_height_mm'];cf=p['control_front_y_mm'];rear=p['tower_rear_y_mm'];hy=rear-p['rear_strip_depth_mm'];t=p['door_thickness_mm'];gap=p['door_clearance_mm']
 lower0=bh;lower1=bh+p['lower_door_height_mm'];upper0=H-p['upper_door_height_mm'];floor=p['lower_floor_z_mm'];stage=p['lower_stage_z_mm']
 f=[[p[f'front_x_fraction_{i}']*dw,p[f'front_y_mm_{i}']] for i in range(7)];seam=f[-1][1]
 full=f+[[dw,hy+gap/2],[0,hy+gap/2]];doorpoly=f+[[dw,hy-gap/2],[0,hy-gap/2]]
 rx0=dw+p['recess_left_fraction']*p['strip_width_mm'];rx1=dw+p['recess_right_fraction']*p['strip_width_mm'];rz0=p['recess_bottom_z_mm'];rz1=p['recess_top_z_mm'];panel=cf+p['recess_back_offset_mm']
 cx=dw/2;lx=cx-p['lower_clear_width_mm']/2;rx=cx+p['lower_clear_width_mm']/2;py=p['lower_pin_y_mm'];ucx=dw*p['upper_dish_x_fraction'];ucy=p['upper_dish_y_mm'];dish=upper0+p['upper_dish_above_door_bottom_mm']
 a=Assembly(values,CONSTRAINTS)
 a.link('fixed','world','fixed')
 pivot=[-p['hinge_axis_outboard_mm'],hy,0]
 def box(n,lo,hi,group='fixed',color=WHITE,tags=('body',),**kw):a.add(n,'block',[lo,hi],group,tags,color,**kw)
 def prism(n,poly,z0,z1,group='fixed',color=WHITE,tags=('body',)):a.add(n,'xy_prism',[poly,z0,z1],group,tags,color)
 def rod(n,c,d,h,group='fixed',color=METAL,axis='z',tags=('body',),**kw):a.add(n,'rod',[c,d,h,axis],group,tags,color,**kw)
 def strip_y(x):return seam+(cf-seam)*(x-dw)/p['strip_width_mm']
 box('right_housing',[tw,cf,bh],[W,D,p['rear_housing_height_mm']],color=[66,73,78,255])
 box('rear_housing',[0,rear-.01,bh],[tw+.01,D,p['rear_housing_height_mm']],color=[66,73,78,255])
 box('base_support',[37,63,0],[W-37,D-40,bh],color=DARK)
 prism('tower_middle_band',full,lower1+gap,upper0-gap)
 box('tower_rear_wall',[0,hy+gap/2,bh],[dw,rear,H])
 for n,x0,x1,z0,z1 in [('control_top',dw,tw,rz1,H),('control_bottom',dw,tw,bh,rz0),('control_left',dw,rx0,rz0,rz1),('control_right',rx1,tw,rz0,rz1)]:
  prism(n,[[x0,strip_y(x0)],[x1,strip_y(x1)],[x1,rear],[x0,rear]],z0,z1)
 box('control_recess_back',[rx0,panel,rz0],[rx1,rear,rz1],color=DARK)
 inner=[[Expr(f'min(max({x.code},{(t+gap).code}),{(dw-t-gap).code})'),y+t+gap] for x,y in f]+[[dw-t-gap,hy],[t+gap,hy]]
 prism('chamber_floor',inner,bh,bh+p['chamber_floor_thickness_mm'],color=DARK)
 states={'closed':{},'handles_released':{},'upper_open':{},'lower_open':{},'open':{}}
 for label,z0,z1 in [('upper',upper0,H),('lower',lower0,lower1)]:
  door=label+'_door';turn=label+'_handle_turn';pop=label+'_handle_pop';tags=(door,)
  a.link(door,'fixed','revolute',pivot,(0,0,-1),p['door_open_angle_deg'])
  x=dw*p['latch_x_fraction'];z=z0+(z1-z0)*p[label+'_latch_height_fraction']
  fy=f[5][1]+(f[6][1]-f[5][1])*(x-f[5][0])/(f[6][0]-f[5][0])
  hp=[x,fy-p['latch_projection_mm']-p['handle_pivot_front_offset_mm'],z+p['handle_pivot_above_center_mm']]
  a.link(turn,door,'revolute',[x-y for x,y in zip(hp,pivot)],(0,-1,0),p['handle_turn_angle_deg'])
  a.link(pop,turn,'revolute',axis=(-1,0,0),limit=p['handle_pop_angle_deg'])
  for state in ('handles_released',label+'_open','open'):states[state].update({pop:p['handle_pop_angle_deg'],turn:p['handle_turn_angle_deg']})
  for state in (label+'_open','open'):states[state][door]=p['door_open_angle_deg']
  for i,(u,v) in enumerate(zip(f[:-1],f[1:])):prism(f'{label}_front_{i:02}',[u,v,[v[0],v[1]+t],[u[0],u[1]+t]],z0,z1,door,tags=tags)
  box(label+'_side',[0,f[0][1],z0],[t,hy-gap/2,z1],door,tags=tags)
  box(label+'_closing_return',[dw-t,seam,z0],[dw,cf+t,z1],door,tags=tags)
  if label=='upper':prism('upper_top',doorpoly,H-t,H,door,tags=tags)
  a.add(label+'_latch_plate','capsule_y',[[x,fy+(1-p['latch_projection_mm'])/2,z],p['latch_plate_width_mm'],p['latch_plate_height_mm'],p['latch_projection_mm']+1],door,tags,DARK)
  a.add(label+'_handle_grip','ellipsoid',[[0,-p['handle_thickness_mm']/2,-p['handle_length_mm']/2],[p['handle_width_mm']/2,p['handle_thickness_mm']/2,p['handle_length_mm']/2]],pop,tags,BLACK,local=True)
  box(label+'_handle_root',[-6,-6,-12],[6,.8,0],pop,DARK,tags,local=True)
  rod(label+'_handle_key_face',[0,-p['handle_thickness_mm']-.3,-48],10,1.5,pop,axis='y',tags=tags,local=True)
  for i in range(2):
   zz=z0+(z1-z0)*p[f'{label}_hinge_fraction_{i}']
   rod(f'{label}_hinge_{i}_pin',[pivot[0],hy,zz],2*p['hinge_radius_mm'],p['hinge_height_mm'],hardware=True)
   for moving in (False,True):
    lo=hy-p['hinge_leaf_half_span_mm'] if moving else hy+gap/2;hi=hy-gap/2 if moving else hy+p['hinge_leaf_half_span_mm']
    box(f'{label}_hinge_{i}_'+('moving_leaf' if moving else 'fixed_leaf'),[-p['hinge_leaf_thickness_mm'],lo,zz-p['hinge_height_mm']/2],[.2,hi,zz+p['hinge_height_mm']/2],door if moving else 'fixed',METAL,tags if moving else ('body',),hardware=True)
 sx=(rx0+rx1)/2;sz=rz0+p['selector_height_fraction']*(rz1-rz0)
 a.link('mode_selector','fixed','prismatic',[sx,panel,sz],(0,-1,0),p['selector_travel_mm']);states['open']['mode_selector']=p['selector_travel_mm']
 face=p['selector_face_offset_mm'];thick=p['selector_head_thickness_mm'];tags=('selector',)
 rod('selector_head',[0,-face+thick/2,0],p['selector_head_diameter_mm'],thick,'mode_selector',BLACK,'y',tags,local=True)
 rod('selector_neck',[0,-(face-thick+1)/2,0],p['selector_neck_diameter_mm'],face-thick-1,'mode_selector',DARK,'y',tags,local=True)
 rod('selector_shaft',[0,(p['selector_travel_mm']+2)/2,0],p['selector_shaft_diameter_mm'],p['selector_travel_mm']+10,'mode_selector',METAL,'y',tags,local=True)
 a.add('selector_bezel','radial_profile',[[sx,panel,sz],[[4.3,-1],[6,-1],[6,0],[4.3,0]],'y',12],tags=tags,color=DARK,group='fixed')
 for label in ('upper','lower'):a.link(label+'_interior','fixed','fixed')
 u='upper_interior';l='lower_interior';it=('interior',)
 box('upper_sample_tray',[12,40,upper0],[dw-7,244,upper0+4],u,DARK,it)
 for label,z0,z1 in [('upper',upper0+4,H-7),('lower',floor,lower1-2)]:
  box(label+'_back_lining',[6,249.99,z0],[dw-3,hy+gap/2,z1],label+'_interior',DARK,it)
  box(label+'_right_lining',[dw-6,40,z0],[dw-3,250,z1],label+'_interior',DARK,it)
 box('lower_left_infill',[6,40,floor-2],[lx,250,lower1-2],l,DARK,it);box('lower_right_infill',[rx,40,floor-2],[dw-6,250,lower1-2],l,DARK,it)
 box('lower_upper_stage',[lx,77,stage],[rx,211,stage+p['lower_stage_thickness_mm']],l,DARK,it)
 box('lower_rear_column',[cx-33.5,212,floor],[cx+33.5,241,stage],l,DARK,it)
 a.add('lower_recess_wall','plate_hole',[[lx,40],[rx,p['lower_floor_rear_y_mm']],[cx,py],p['lower_recess_diameter_mm'],p['lower_recess_bottom_z_mm'],floor],l,('lower_floor','interior'),DARK)
 rod('lower_recess_bottom',[cx,py,p['lower_recess_bottom_z_mm']-1],p['lower_recess_diameter_mm']+2,2,l,BLACK,tags=('lower_floor','interior'))
 ro=p['upper_dish_outer_diameter_mm']/2;r0=p['upper_dish_inner_bottom_diameter_mm']/2;r1=p['upper_dish_inner_top_diameter_mm']/2;top=dish+p['upper_dish_depth_mm']
 # 0.01 mm overlap with the tray avoids circular edge-only contact after CSG union.
 a.add('upper_dish_wall','radial_profile',[[ucx,ucy,0],[[r0,dish],[r1,top],[ro,top],[ro,dish-3.01]],'z',12],u,('upper_dish','interior'),METAL)
 rod('upper_dish_bottom',[ucx,ucy,dish-1.5],2*r1,3,u,tags=('upper_dish','interior'))
 rod('upper_post_base',[ucx,ucy,dish+6],34,12,u,tags=('upper_dish','interior'))
 rod('upper_post_collar',[ucx,ucy,dish+18],18,12,u,tags=('upper_dish','interior'))
 rod('upper_sample_post',[ucx,ucy,dish+(24+p['upper_post_above_floor_mm'])/2],6,p['upper_post_above_floor_mm']-24,u,tags=('upper_dish','interior'))
 for i,sign in enumerate((-1,1)):
  xx=cx+sign*p['lower_guide_spacing_mm']/2
  rod(f'lower_guide_{i}',[xx,p['lower_guide_y_mm'],(floor+p['lower_guide_top_z_mm'])/2],p['lead_screw_diameter_mm'] if i==0 else p['right_guide_diameter_mm'],p['lower_guide_top_z_mm']-floor,l,tags=it)
  rod(f'lower_guide_foot_{i}',[xx,p['lower_guide_y_mm'],floor+6],16,12,l,BLACK,tags=it)
  rod('lower_small_pin' if i==0 else 'lower_second_pin',[cx+sign*p['lower_pin_spacing_mm']/2,py,stage-p['lower_pin_length_mm']/2],p['small_pin_diameter_mm'] if i==0 else p['second_pin_diameter_mm'],p['lower_pin_length_mm'],l,tags=('lower_pins','interior'))
 rod('lower_central_piece',[cx,py,p['central_piece_bottom_z_mm']+p['central_piece_height_mm']/2],p['central_piece_diameter_mm'],p['central_piece_height_mm'],l,tags=it)
 rod('lower_bearing_flange',[cx,py,stage-5],p['bearing_outer_diameter_mm'],10,l,tags=it)
 rod('lower_bearing_step',[cx,py,stage-12.5],40,5,l,tags=it);rod('lower_bearing_adjuster',[cx+22,101,stage-6],8,12,l,axis='x',tags=it)
 a.states=states
 return a

def model(p,v):
 p.update(lower_clear_width_m=v['lower_clear_width_mm']/1000,lead_screw_diameter_m=v['lead_screw_diameter_mm']/1000,
  right_guide_diameter_m=v['right_guide_diameter_mm']/1000,small_pin_diameter_m=v['small_pin_diameter_mm']/1000,
  central_piece_diameter_m=v['central_piece_diameter_mm']/1000,lower_center_x_m=v['door_width_mm']/2000)
 cx=v['door_width_mm']/2
 p.update(lower_guide_xy_m=[[(cx+s*v['lower_guide_spacing_mm']/2)/1000,v['lower_guide_y_mm']/1000] for s in (-1,1)],
  central_piece_bottom_z_m=v['central_piece_bottom_z_mm']/1000,central_piece_height_m=v['central_piece_height_mm']/1000,
  small_pin_xy_m=[(cx-v['lower_pin_spacing_mm']/2)/1000,v['lower_pin_y_mm']/1000],small_pin_length_m=v['lower_pin_length_mm']/1000)
 p['detail_parameters'].update(floor_hole_radius_m=v['lower_recess_diameter_mm']/2000,floor_recess_bottom_z_m=v['lower_recess_bottom_z_mm']/1000,
  second_pin_diameter_m=v['second_pin_diameter_mm']/1000,bearing_outer_radius_m=v['bearing_outer_diameter_mm']/2000)
 a=definition(v);m=a.model(p)
 m.demo_states={name:{key:float(resolve(value,v))/(1000 if a.links[key]['type']=='prismatic' else 1) for key,value in state.items()} for name,state in a.states.items()}
 return m

def configurations(p,m):
 """Derived compatibility metadata for the existing URDF and dimension exporters."""
 v=p['design_parameters'];c={};d={'control_strip_front_shape':'tapered_flush'}
 controls={'parameters':c};doors={'parameters':d}
 c.update(handle_pop_preview_degrees=v['handle_pop_angle_deg'],handle_turn_preview_degrees=v['handle_turn_angle_deg'],knob_stroke_preview_m=v['selector_travel_mm']/1000)
 d.update(body_width_m=v['body_width_mm']/1000,body_depth_m=v['body_depth_mm']/1000,body_height_m=v['body_height_mm']/1000,
  tower_width_m=(v['door_width_mm']+v['strip_width_mm'])/1000,door_allocation_width_m=v['door_width_mm']/1000,strip_allocation_width_m=v['strip_width_mm']/1000,
  control_front_y_m=v['control_front_y_mm']/1000,tower_rear_y_m=v['tower_rear_y_mm']/1000,
  front_profile_xy_m=[[v[f'front_x_fraction_{i}']*v['door_width_mm']/1000,v[f'front_y_mm_{i}']/1000] for i in range(7)],
  upper_door_height_m=v['upper_door_height_mm']/1000,lower_door_height_m=v['lower_door_height_mm']/1000,base_height_m=v['base_height_mm']/1000,
  door_depth_reference_x_m=v['door_width_mm']*.94/1000)
 # These are exported views of the shared definition, not geometry inputs.
 transforms=m.transforms({})
 doors['parts']=[]
 for name,mesh in m.parts.items():
  q=mesh.copy();q.apply_transform(transforms[m.info[name]['group']])
  doors['parts'].append(dict(name=name,bounds_closed_m=q.bounds.tolist(),**m.info[name]))
 controls.update(links=m.links,parts=m.info,pose_snapshots=m.demo_states,authoring='Derived from shared definition; edit parameters.json/design_parameters')
 doors['authoring']='Derived from shared definition; edit parameters.json/design_parameters'
 return controls,doors
