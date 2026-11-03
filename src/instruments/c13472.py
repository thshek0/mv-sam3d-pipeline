"""Orientation system solids and sliding hood, shared by both export backends."""
import numpy as np
from .shared_geometry import Assembly,Expr
from .geometry import WHITE,DARK,BLACK,METAL

CONSTRAINTS=[
 ('cover_left_fraction+cover_width_fraction+cover_travel_fraction <= 1 and cover_travel_fraction > 0','Hood exceeds body envelope'),
 ('holder_hole_diameter_mm > 0 and holder_hole_diameter_mm < holder_collar_diameter_mm and holder_collar_diameter_mm < holder_frame_width_mm','Invalid holder opening'),
 ('body_height_mm*base_height_fraction+wall_mm+holder_above_floor_mm+4 < body_height_mm*shoulder_height_fraction','Holder exceeds hood'),
 ('motor_mount_thickness_mm > motor_wall_inset_mm','Motor mount must overlap motor end'),
]

def definition(values):
 p={k:Expr(k) for k in values};W=p['body_width_mm'];D=p['body_depth_mm'];H=p['body_height_mm'];t=p['wall_mm']
 base=H*p['base_height_fraction'];shoulder=H*p['shoulder_height_fraction'];bevel=D*p['front_bevel_fraction'];rear=D*p['rear_position_fraction'];floor=base+t
 wall=W*p['inner_wall_fraction'];xa=W*p['cover_left_fraction'];xb=xa+W*p['cover_width_fraction'];x=W*p['stage_center_x_fraction'];y=D*p['stage_center_y_fraction']
 hz=floor+p['holder_above_floor_mm'];z=floor+56.6;mx=wall-p['motor_wall_inset_mm']-44;my=y+15;mz=floor+p['motor_above_floor_mm']
 a=Assembly(values,CONSTRAINTS,W*p['cover_travel_fraction'],axis=(1,0,0))
 def box(n,lo,hi,color=WHITE,group='base_link',tags=('body',)):a.add(n,'block',[lo,hi],group,tags,color)
 def yz(n,points,x0,x1,color=WHITE,group='base_link',tags=('body',)):a.add(n,'yz_prism',[points,x0,x1],group,tags,color)
 def rod(n,c,d,h,axis='z',tags=('interior',),color=METAL):a.add(n,'rod',[c,d,h,axis],tags=tags,color=color)
 box('base_housing',[0,0,0],[W,D,base])
 yz('left_wall',[[0,base],[rear,base],[rear,H],[bevel,H],[0,shoulder]],0,W*.032)
 box('back_housing',[0,rear-.01,base-.01],[W,D,H],DARK)
 yz('right_inner_housing',[[10,base],[rear,base],[rear,H-10],[bevel+10,H-10],[10,shoulder-10]],wall,W)
 box('chamber_floor',[W*.032,12,base],[wall,rear,floor],BLACK)
 box('side_black_wall_lining',[wall-1,10,floor],[wall,rear,floor+181.6],BLACK)
 box('front_rail',[W*.032,0,base],[W,10,base+8],METAL);box('back_rail',[W*.032,rear-4,H-9],[W,rear,H-4],METAL)
 gx=xa+W*.045;gw=W*.026;z0=base+10
 for n,lo,hi in [('cover_front_left',[xa,0,z0],[gx,t,shoulder]),('cover_front_right',[gx+gw,0,z0],[xb,t,shoulder]),
  ('cover_front_bottom',[gx,0,z0],[gx+gw,t,z0+25]),('cover_front_top',[gx,0,shoulder-35],[gx+gw,t,shoulder]),
  ('cover_grip_back',[gx-1,t,z0+24],[gx+gw+1,2*t,shoulder-34])]:box(n,lo,hi,DARK,'cover',('cover',))
 yz('cover_chamfer',[[0,shoulder],[bevel,H],[bevel,H-t],[t,shoulder]],xa,xb,DARK,'cover',('cover',))
 box('cover_top',[xa,bevel-.01,H-t],[xb,rear-4,H],DARK,'cover',('cover',))
 rod('stage_rotary_base',[x,y,floor+6],2*p['stage_radius_mm'],12,color=DARK)
 rod('stage_rotary_top',[x,y,floor+17],1.8*p['stage_radius_mm'],10)
 for n,lo,hi in [('stage_pedestal',[x-40,y-40,floor+22],[x+40,y+40,z]),('stage_lower_slide',[x-52,y-48,z],[x+52,y+48,z+14]),
  ('stage_upper_slide',[x-47,y-44,z+18],[x+47,y+44,z+32]),('holder_support_left',[x-38,y-35,z+32],[x-27,y+35,hz-8]),('holder_support_right',[x+27,y-35,z+32],[x+38,y+35,hz-8])]:box(n,lo,hi,DARK,tags=('interior',))
 for i,yy in enumerate([y-32,y+32]):rod(f'stage_rail_{i}',[x,yy,z+16],8,105,'x')
 for i,(pos,axis,vec) in enumerate([([x-74,y-35,z+25],'x',[1,0,0]),([x+35,y-70,z+6],'y',[0,1,0])]):
  rod(f'micrometer_barrel_{i}',pos,14,30,axis);rod(f'micrometer_knob_{i}',[v-d*22 for v,d in zip(pos,vec)],18,14,axis);rod(f'micrometer_spindle_{i}',[v+d*22 for v,d in zip(pos,vec)],4,25,axis)
 r=p['holder_frame_width_mm']/2;ri=p['holder_hole_diameter_mm']/2;ro=p['holder_collar_diameter_mm']/2
 a.add('holder_frame','plate_hole',[[x-r,y-r],[x+r,y+r],[x,y],2*ri,hz-8,hz],tags=('holder','interior'),color=DARK)
 a.add('holder_collar','radial_profile',[[x,y,0],[[ri,hz],[ro,hz],[ro,hz+4],[ri,hz+4]],'z',16],tags=('holder','interior'),color=METAL)
 for i,(xx,yy) in enumerate([(x-37,y-37),(x+37,y-37),(x-37,y+37),(x+37,y+37)]):rod(f'holder_frame_screw_{i}',[xx,yy,hz+2],8,4,tags=('holder','interior'))
 tags=('side_unit','interior')
 rod('side_rotary_plate',[mx,my,mz],96,9,'x',tags,DARK);rod('side_rotary_hub',[mx-20,my,mz],40,32,'x',tags,BLACK)
 rod('side_motor_body',[mx+24,my,mz],2*p['motor_radius_mm'],40,'x',tags,DARK)
 box('side_motor_foot',[mx+6,my-25,floor],[mx+45,my+25,mz-10],DARK,tags=tags)
 box('side_wall_mount',[wall-p['motor_mount_thickness_mm'],my-38,mz-40],[wall,my+38,mz+40],BLACK,tags=tags)
 for i,ang in enumerate(np.linspace(0,2*np.pi,4,endpoint=False)):rod(f'side_plate_screw_{i}',[mx-6,my+40*float(np.cos(ang)),mz+40*float(np.sin(ang))],6,3,'x',tags)
 return a

def model(p,v):
 p.pop('holder',None)
 p.update(body_wdh_m=[v[k]/1000 for k in ('body_width_mm','body_depth_mm','body_height_mm')],wall_m=v['wall_mm']/1000,
  cover_left_fraction=v['cover_left_fraction'],cover_width_fraction=v['cover_width_fraction'],cover_travel_fraction=v['cover_travel_fraction'])
 d=p['detail_parameters'];d.update(stage_center_xy_m=[v['body_width_mm']*v['stage_center_x_fraction']/1000,v['body_depth_mm']*v['stage_center_y_fraction']/1000],
  holder_inner_radius_m=v['holder_hole_diameter_mm']/2000,holder_outer_radius_m=v['holder_collar_diameter_mm']/2000,
  holder_top_z_m=(v['body_height_mm']*v['base_height_fraction']+v['wall_mm']+v['holder_above_floor_mm'])/1000,
  side_wall_mount_x_m=v['body_width_mm']*v['inner_wall_fraction']/1000)
 return definition(v).model(p)
