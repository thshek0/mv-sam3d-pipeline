"""Tau solids and lid joint, shared by native CAD and simulation."""
import numpy as np
from .shared_geometry import Assembly,Expr
from .geometry import DARK,BLACK,METAL

CONSTRAINTS=[
 ('clear_width_mm > grip_width_mm and hole_diameter_mm > 0 and hole_diameter_mm < fastener_radius_mm','Invalid chamber/grip/hole'),
 ('overall_height_mm-lid_height_mm-chamber_depth_mm-floor_thickness_mm > plinth_height_mm','Chamber below plinth'),
 ('chamber_depth_mm > grip_depth_mm and chamber_rear_y_mm < body_depth_mm','Invalid chamber depth'),
 ('chamber_center_x_mm-(clear_width_mm+2*wall_mm)/2 >= front_projection_mm','Left chamber curve exceeds body'),
 ('chamber_center_x_mm+(clear_width_mm+2*wall_mm)/2+front_projection_mm <= body_width_mm','Right chamber curve exceeds body'),
 ('open_angle_deg > 0 and open_angle_deg <= 100','Unsupported lid demonstration angle'),
]

def definition(values):
 p={k:Expr(k) for k in values};W=p['body_width_mm'];D=p['body_depth_mm'];H=p['overall_height_mm']-p['lid_height_mm']
 t=p['wall_mm'];cx=p['chamber_center_x_mm'];xa=cx-(p['clear_width_mm']+2*t)/2;xb=cx+(p['clear_width_mm']+2*t)/2
 ya=-p['front_projection_mm'];yb=p['chamber_rear_y_mm'];floor=H-p['chamber_depth_mm'];ft=p['floor_thickness_mm'];cy=(ya+t+yb)/2
 a=Assembly(values,CONSTRAINTS,p['open_angle_deg'],axis=(-1,0,0),origin=[xa,yb,H],joint_type='revolute')
 def box(n,lo,hi,color=DARK,tags=('body',)):a.add(n,'block',[lo,hi],tags=tags,color=color)
 def rod(n,center,d,h,axis='z'):a.add(n,'rod',[center,d,h,axis],tags=('hardware',),color=METAL)
 box('rear_housing',[0,yb,0],[W,D,H]);box('front_plinth',[0,0,0],[W,yb,p['plinth_height_mm']])
 box('left_front_housing',[0,0,p['plinth_height_mm']],[xa,yb,H]);box('right_front_housing',[xb,0,p['plinth_height_mm']],[W,yb,H])
 box('chamber_left_wall',[xa,ya,floor-ft],[xa+t,yb,H]);box('chamber_right_wall',[xb-t,ya,floor-ft],[xb,yb,H])
 box('front_enclosure_panel',[xa,ya,p['plinth_height_mm']],[xb,ya+t,floor-ft])
 box('chamber_front',[xa,ya,floor-ft],[xb,ya+t,H-p['grip_depth_mm']])
 gx0=cx-p['grip_width_mm']/2;gx1=cx+p['grip_width_mm']/2
 box('grip_left',[xa,ya,H-p['grip_depth_mm']],[gx0,ya+t,H]);box('grip_right',[gx1,ya,H-p['grip_depth_mm']],[xb,ya+t,H])
 box('grip_recess_back',[gx0-t,ya+t,H-p['grip_depth_mm']-t],[gx1+t,ya+2*t,H],BLACK)
 for side,edge,sign in [('left',xa,-1),('right',xb,1)]:a.add('front_curved_'+side,'quarter_disc',[edge,p['front_projection_mm'],p['plinth_height_mm'],H,sign],color=DARK)
 a.add('perforated_floor','plate_hole',[[xa+t,ya+t],[xb-t,yb],[cx,cy],p['hole_diameter_mm'],floor-ft,floor],tags=('floor',),color=BLACK)
 a.add('sample_box_lid','rounded_x',[[xa,ya,H],[xb,yb,H+p['lid_height_mm']],p['lid_corner_radius_mm']],group='cover',tags=('lid','cover'),color=DARK)
 for i in range(8):
  angle=2*np.pi*i/8
  rod(f'floor_fastener_{i}',[cx+p['fastener_radius_mm']*float(np.cos(angle)),cy+p['fastener_radius_mm']*float(np.sin(angle)),floor+p['fastener_height_mm']/2],p['fastener_diameter_mm'],p['fastener_height_mm'])
 for i,x in enumerate([xa+p['knob_inset_mm'],xb-p['knob_inset_mm']]):
  rod(f'rear_adjuster_shaft_{i}',[x,yb-p['knob_shaft_offset_mm'],H-p['knob_below_rim_mm']],p['knob_shaft_diameter_mm'],p['knob_shaft_length_mm'],'y')
  rod(f'rear_adjuster_knob_{i}',[x,yb-p['knob_offset_mm'],H-p['knob_below_rim_mm']],p['knob_diameter_mm'],p['knob_length_mm'],'y')
 for i,x in enumerate([xa+p['latch_inset_mm'],xb-p['latch_inset_mm']]):box(f'front_latch_mount_{i}',[x-p['latch_width_mm']/2,ya+t,floor],[x+p['latch_width_mm']/2,ya+t+p['latch_depth_mm'],floor+p['latch_height_mm']],METAL,('hardware',))
 return a

def model(p,values):
 p.pop('holder',None)
 H=values['overall_height_mm']-values['lid_height_mm'];cx=values['chamber_center_x_mm'];w=values['clear_width_mm']+2*values['wall_mm']
 p.update(body_wdh_m=[values['body_width_mm']/1000,values['body_depth_mm']/1000,H/1000],overall_height_m=values['overall_height_mm']/1000,
  sample_box_height_m=values['lid_height_mm']/1000,lid_preview_angle_deg=values['open_angle_deg'],wall_m=values['wall_mm']/1000,
  chamber_x_m=[(cx-w/2)/1000,(cx+w/2)/1000],chamber_y_m=[-values['front_projection_mm']/1000,values['chamber_rear_y_mm']/1000],
  chamber_floor_z_m=(H-values['chamber_depth_mm'])/1000,opening_probe_xy_m=[cx/1000,.15])
 p['detail_parameters']['hole_diameter_m']=values['hole_diameter_mm']/1000
 p['detail_parameters']['central_hole_xy_m']=[cx/1000,(-values['front_projection_mm']+values['wall_mm']+values['chamber_rear_y_mm'])/2000]
 return definition(values).model(p)
