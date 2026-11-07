"""One Cary part/joint recipe consumed by both native CAD and simulation exports."""
from .shared_geometry import Assembly,Expr
from .geometry import WHITE,DARK,BLACK,METAL
import math


CONSTRAINTS=[
    ('chamber_center_x_mm-cover_width_mm/2 > 0 and chamber_center_x_mm+cover_width_mm/2 < body_width_mm','Chamber must fit inside body'),
    ('cover_travel_mm > 0 and cover_front_y_mm+cover_length_mm+cover_travel_mm <= body_depth_mm-receiver_rear_margin_mm','Cover must retract inside body'),
    ('socket_inner_width_mm > 0 and socket_inner_width_mm < socket_outer_width_mm','Invalid socket width'),
    ('socket_inner_depth_mm > 0 and socket_inner_depth_mm < socket_outer_depth_mm','Invalid socket depth'),
    ('chamber_floor_z_mm+holder_plate_above_floor_mm+socket_height_mm < body_height_mm-cover_thickness_mm','Holder exceeds cover clearance'),
    ('grip_depth_mm > 0 and grip_depth_mm < cover_thickness_mm and grip_width_mm < cover_width_mm','Invalid grip pocket'),
    ('grip_front_offset_mm > 0 and grip_front_offset_mm+grip_length_mm < cover_length_mm','Grip outside lid'),
    ('grip_corner_radius_mm > 0 and 2*grip_corner_radius_mm < min(grip_width_mm,grip_length_mm)','Invalid grip corner radius'),
    ('holder_plate_above_floor_mm > 12 and receiver_gap_mm > 0 and receiver_thickness_mm > 0','Invalid holder/receiver dimensions'),
]


def definition(values):
    p={k:Expr(k) for k in values}
    W,D,H=(p[k] for k in ('body_width_mm','body_depth_mm','body_height_mm'))
    cx=p['chamber_center_x_mm'];xa=cx-p['cover_width_mm']/2;xb=cx+p['cover_width_mm']/2
    ya=p['cover_front_y_mm'];yb=ya+p['cover_length_mm'];floor=p['chamber_floor_z_mm'];wall=p['chamber_wall_mm']
    x=cx;y=p['holder_center_y_mm'];z=floor+p['holder_plate_above_floor_mm']
    a=Assembly(values,CONSTRAINTS,p['cover_travel_mm'])
    def box(name,lo,hi,color=WHITE,tags=('body',),group='base_link'):
        a.add(name,'block',[lo,hi],group,tags,color)
    def rounded(name,lo,hi,r,color=WHITE):a.add(name,'rounded_x',[lo,hi,r],color=color)
    def rod(name,center,d,length):a.add(name,'rod',[center,d,length,'z'],tags=('holder',),color=METAL)
    rounded('right_housing',[xb,0,0],[W,D,H],p['housing_radius_mm'])
    rounded('left_end_housing',[0,0,0],[xa,D,H],p['housing_radius_mm'])
    a.add('compartment_shell','subtract',[
        ['block',[[xa,ya,0],[xb,D,H-9]]],
        [['block',[[xa+wall,ya-1,floor],[xb-wall,D-p['rear_wall_depth_mm'],H]]]]],color=DARK)
    rounded('removable_front_door',[xa,0,12],[xb,22,H-9],9,DARK)
    receiver_top=H-p['cover_thickness_mm']-p['receiver_gap_mm']
    box('fixed_cover_receiver',[xa,yb,receiver_top-p['receiver_thickness_mm']],
        [xb,D-p['receiver_rear_margin_mm'],receiver_top],DARK)
    gx0=cx-p['grip_width_mm']/2;gx1=cx+p['grip_width_mm']/2
    gy0=ya+p['grip_front_offset_mm'];gy1=gy0+p['grip_length_mm'];gz=H-p['grip_depth_mm']
    # Plan view, X right / Y up: only the upper-left and lower-right corners round.
    r=p['grip_corner_radius_mm']
    grip=[[gx0,gy0],[gx1-r,gy0]]
    grip += [[gx1-r+r*round(math.cos(math.radians(-90+i*90/16)),12),
              gy0+r+r*round(math.sin(math.radians(-90+i*90/16)),12)] for i in range(1,17)]
    grip += [[gx1,gy1],[gx0+r,gy1]]
    grip += [[gx0+r+r*round(math.cos(math.radians(90+i*90/16)),12),
              gy1-r+r*round(math.sin(math.radians(90+i*90/16)),12)] for i in range(1,17)]
    a.add('lid_shell','subtract',[
        ['block',[[xa,ya,H-p['cover_thickness_mm']],[xb,yb,H]]],
        [['xy_prism',[grip,gz,H+1]]]],group='cover',tags=('cover',),color=DARK)
    a.add('holder_foot','xy_prism',[[[x-60,y-25],[x-60,y+25],[x+60,y+25],[x+60,y-25],[x,y-65]],floor,floor+8],tags=('holder',),color=BLACK)
    for name,lo,hi in [
        ('holder_bridge_left',[x-50,y-19,floor+8],[x-38,y+19,z-4]),
        ('holder_bridge_right',[x+38,y-19,floor+8],[x+50,y+19,z-4])]:box(name,lo,hi,BLACK,('holder',))
    box('holder_plate',[x-p['holder_plate_width_mm']/2,y-p['holder_plate_depth_mm']/2,z-4],
        [x+p['holder_plate_width_mm']/2,y+p['holder_plate_depth_mm']/2,z],METAL,('holder',))
    sx=x-p['socket_offset_x_mm'];wi=p['socket_inner_width_mm'];di=p['socket_inner_depth_mm']
    wo=p['socket_outer_width_mm'];do=p['socket_outer_depth_mm'];top=z+p['socket_height_mm']
    a.add('socket_shell','subtract',[
        ['block',[[sx-wo/2,y-do/2,z],[sx+wo/2,y+do/2,top]]],
        [['block',[[sx-wi/2,y-di/2,z-1],[sx+wi/2,y+di/2,top+1]]]]],tags=('holder','socket'),color=BLACK)
    rod('holder_lower_clamp_knob',[x,y-25,floor+10.5],20,5)
    rod('holder_adjuster_shaft',[x+25,y,z+8],6,16)
    rod('holder_adjuster_knob',[x+25,y,z+18.5],14,5)
    rod('holder_threaded_post',[x+40,y-2,z+10],6,20)
    for i,xx in enumerate([x-47,x+47]):rod(f'holder_plate_screw_{i}',[xx,y+16,z+1.5],6,3)
    for label,xx in [('left',xa+12),('right',xb-13)]:
        box('inner_'+label+'_rail',[xx-2,ya,floor+56],[xx+2,ya+255,floor+101],DARK)
    return a


def parameters(p):
    from .design import parameters as shared_parameters
    return shared_parameters(p)


def model(p):
    values,_=parameters(p)
    p['design_parameters']=values
    p['authoring']='Shared Cary part/joint definition; design_parameters is authoritative (mm/degrees)'
    p['body_wdh_m']=[values[k]/1000 for k in ('body_width_mm','body_depth_mm','body_height_mm')]
    d=p['detail_parameters'];ya=values['cover_front_y_mm'];yb=ya+values['cover_length_mm']
    d.update(moving_cover_y_m=[ya/1000,yb/1000],fixed_cover_y_m=[yb/1000,(values['body_depth_mm']-values['receiver_rear_margin_mm'])/1000],
             cover_thickness_m=values['cover_thickness_mm']/1000,
             holder_center_xy_m=[values['chamber_center_x_mm']/1000,values['holder_center_y_mm']/1000],
             holder_plate_wh_m=[values['holder_plate_width_mm']/1000,values['holder_plate_depth_mm']/1000])
    p.pop('holder',None)  # Superseded legacy dimensions; design_parameters owns the holder now.
    xa=values['chamber_center_x_mm']-values['cover_width_mm']/2;xb=xa+values['cover_width_mm'];H=values['body_height_mm']
    p['lid_bounds_m']=[[xa/1000,ya/1000,(H-values['cover_thickness_mm'])/1000],[xb/1000,yb/1000,H/1000]]
    p['cavity_left_x_m']=(xa+values['chamber_wall_mm'])/1000
    p['cavity_whd_m']=[(values['cover_width_mm']-2*values['chamber_wall_mm'])/1000,
        (H-9-values['chamber_floor_z_mm'])/1000,(values['body_depth_mm']-values['rear_wall_depth_mm']-ya)/1000]
    p['opening_probe_xy_m']=[values['chamber_center_x_mm']/1000,(ya+min(68,values['cover_travel_mm']/2))/1000]
    return definition(values).model(p)


def decal_bounds(values):
    """Keep the photographed identity label attached to its housing panel."""
    x0=values['chamber_center_x_mm']+values['cover_width_mm']/2
    width=values['body_width_mm']-x0;h=values['body_height_mm']
    return [x0+width*20/282,x0+width*135/282,h*106/196,h*147/196]
