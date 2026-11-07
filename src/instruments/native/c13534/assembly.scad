// Quantaurus-QY. PDF p8: body envelope; IMG_9845: moving black door interiors.
// IMG_9866/9867/9869: low left plinth. Functional diameters follow user readings.
include <parameters_native.scad>
include <common.scad>
/* [View and motion] */
pose="closed"; // [closed,open,custom,handles_released,upper_open,lower_open]
animate_motion=false;
upper_door_opening=0; // [0:0.01:1]
lower_door_opening=0; // [0:0.01:1]
upper_handle_turn_opening=0; // [0:0.01:1]
lower_handle_turn_opening=0; // [0:0.01:1]
upper_handle_pop_opening=0; // [0:0.01:1]
lower_handle_pop_opening=0; // [0:0.01:1]
mode_selector_opening=0; // [0:0.01:1]
/* [Geometry] */
part="all";
details=true;
fillets=true;
simplified=false;
/* [Hidden] */
$fn=64;
BODY=[body_width_mm,body_depth_mm,body_height_mm];
TOWER_W=door_width_mm+strip_width_mm;
HINGE_Y=tower_rear_y_mm-rear_strip_depth_mm;
HINGE=[-hinge_axis_outboard_mm,HINGE_Y,0];
LOWER_Z=base_height_mm;
LOWER_TOP=LOWER_Z+lower_door_height_mm;
UPPER_Z=BODY.z-upper_door_height_mm;
CENTER_X=door_width_mm/2;
ROOM_LEFT=CENTER_X-lower_clear_width_mm/2;
ROOM_RIGHT=CENTER_X+lower_clear_width_mm/2;
RECESS_LEFT=door_width_mm+recess_left_fraction*strip_width_mm;
RECESS_RIGHT=door_width_mm+recess_right_fraction*strip_width_mm;
PANEL_Y=control_front_y_mm+recess_back_offset_mm;
FRONT_PROFILE=[
    [front_x_fraction_0*door_width_mm,front_y_mm_0],
    [front_x_fraction_1*door_width_mm,front_y_mm_1],
    [front_x_fraction_2*door_width_mm,front_y_mm_2],
    [front_x_fraction_3*door_width_mm,front_y_mm_3],
    [front_x_fraction_4*door_width_mm,front_y_mm_4],
    [front_x_fraction_5*door_width_mm,front_y_mm_5],
    [door_width_mm,front_y_mm_6]
];
// Insets are deliberately short expressions, with no expanded trigonometric literals.
function plan(front_inset=0,left=0,right=door_width_mm,rear=HINGE_Y) =
    concat([for(p=FRONT_PROFILE) [min(max(p.x,left),right),p.y+front_inset]],[[right,rear],[left,rear]]);
function joint_fraction(label,kind,manual) = animate_motion ? 1-abs(2*$t-1) :
    pose=="custom" ? manual : pose=="open" ? 1 :
    kind=="door" ? (pose==str(label,"_open") ? 1 : 0) :
    (pose=="handles_released" || pose==str(label,"_open") ? 1 : 0);
function door_fraction(label) = joint_fraction(label,"door",label=="upper" ? upper_door_opening : lower_door_opening);
function turn_fraction(label) = joint_fraction(label,"handle",label=="upper" ? upper_handle_turn_opening : lower_handle_turn_opening);
function pop_fraction(label) = joint_fraction(label,"handle",label=="upper" ? upper_handle_pop_opening : lower_handle_pop_opening);
function selector_fraction() = animate_motion ? 1-abs(2*$t-1) : pose=="custom" ? mode_selector_opening : pose=="open" ? 1 : 0;
assert(lower_left_infill_height_mm>0 && lower_floor_z_mm+lower_left_infill_height_mm<lower_stage_z_mm);
assert(LOWER_TOP+2*door_clearance_mm<UPPER_Z);

module fixed_housing() {
    color("#42494e") {
        component("right_housing",["body"]) block([TOWER_W-EPS,control_front_y_mm,base_height_mm-EPS],BODY-[0,0,BODY.z-rear_housing_height_mm]);
        component("rear_housing",["body"]) block([0,tower_rear_y_mm-EPS,base_height_mm-EPS],[TOWER_W+EPS,BODY.y,rear_housing_height_mm]);
    }
    component("base_support",["body"]) color(DARK) block([37,63,0],[BODY.x-37,BODY.y-40,base_height_mm]);
    color(WHITE) {
        component("tower_middle_band",["body"])
            xy_prism(plan(rear=HINGE_Y+door_clearance_mm/2+EPS),LOWER_TOP+door_clearance_mm,UPPER_Z-door_clearance_mm);
        component("tower_rear_wall",["body"])
            block([0,HINGE_Y+door_clearance_mm/2,base_height_mm-EPS],[door_width_mm+EPS,tower_rear_y_mm-EPS,BODY.z]);
        component("control_frame",["body"]) render(convexity=10) difference() {
            xy_prism([[door_width_mm,FRONT_PROFILE[6].y],[TOWER_W,control_front_y_mm],
                      [TOWER_W,tower_rear_y_mm],[door_width_mm,tower_rear_y_mm]],base_height_mm-EPS,BODY.z);
            block([RECESS_LEFT,-100,recess_bottom_z_mm],[RECESS_RIGHT,tower_rear_y_mm+EPS,recess_top_z_mm]);
        }
    }
    color(DARK) {
        component("control_recess_back",["body"])
            block([RECESS_LEFT-EPS,PANEL_Y,recess_bottom_z_mm-EPS],[RECESS_RIGHT+EPS,tower_rear_y_mm,recess_top_z_mm+EPS]);
        component("chamber_floor",["body"])
            xy_prism(plan(door_thickness_mm+door_clearance_mm,door_thickness_mm+door_clearance_mm,
                          door_width_mm-door_thickness_mm-door_clearance_mm),base_height_mm,base_height_mm+chamber_floor_thickness_mm);
    }
}

// Door geometry is local in Z. A single parent hinge transform carries shell,
// lining, latch and the handle's nested turn/pop transforms.
module door_shell(height,roof) {
    t=door_thickness_mm;
    render(convexity=10) difference() {
        xy_prism(plan(rear=HINGE_Y-door_clearance_mm/2),0,height);
        xy_prism(plan(t,t,door_width_mm-t,HINGE_Y+EPS),-EPS,roof ? height-t : height+EPS);
        block([door_width_mm-t,control_front_y_mm+t,-EPS],
              [door_width_mm+EPS,HINGE_Y+EPS,roof ? height-t : height+EPS]);
    }
}
module door_inner_lining(height,roof) {
    t=door_thickness_mm; lining=door_lining_thickness_mm; edge=door_lining_edge_inset_mm;
    end=door_width_mm-t-edge;
    // Evaluate this solid before the moving transform: prevents F5 phantom faces.
    render(convexity=10) difference() {
        xy_prism(plan(t-EPS,t-EPS,end,HINGE_Y-door_clearance_mm/2-edge),edge,roof ? height-t+EPS : height-edge);
        xy_prism(plan(t+lining,t+lining,end,HINGE_Y+EPS),-EPS,roof ? height-t-lining : height+EPS);
    }
}
module handle_grip() {
    if(fillets && !simplified)
        translate([0,-handle_thickness_mm/2,-handle_length_mm/2])
            scale([handle_width_mm,handle_thickness_mm,handle_length_mm]/2) sphere(r=1);
    else block([-handle_width_mm/2,-handle_thickness_mm,-handle_length_mm],[handle_width_mm/2,0,0]);
}
module door_latch(label,height) {
    x=door_width_mm*latch_x_fraction;
    z=height*(label=="upper" ? upper_latch_height_fraction : lower_latch_height_fraction);
    y=lookup(x,FRONT_PROFILE);
    component(str(label,"_latch_plate"),[str(label,"_door"),"doors"]) color(DARK)
        capsule_y([x,y+(1-latch_projection_mm)/2,z],latch_plate_width_mm,latch_plate_height_mm,latch_projection_mm+1);
    translate([x,y-latch_projection_mm-handle_pivot_front_offset_mm,z+handle_pivot_above_center_mm])
        rotate(a=handle_turn_angle_deg*turn_fraction(label),v=[0,-1,0])
            rotate(a=handle_pop_angle_deg*pop_fraction(label),v=[-1,0,0]) {
                component(str(label,"_handle_grip"),[str(label,"_door"),"doors","handles"]) color(BLACK) handle_grip();
                component(str(label,"_handle_root"),[str(label,"_door"),"doors","handles"]) color(DARK)
                    block([-6,-6,-12],[6,0.8,0]);
                if(show_details()) component(str(label,"_handle_key_face"),[str(label,"_door"),"doors","handles"]) color(METAL)
                    rod([0,-handle_thickness_mm-0.3,-48],10,1.5,"y");
            }
}
module door_hinges(label,z0,height,moving) {
    fractions=label=="upper" ? [upper_hinge_fraction_0,upper_hinge_fraction_1] : [lower_hinge_fraction_0,lower_hinge_fraction_1];
    if(show_details()) component(str(label,moving ? "_moving_hinges" : "_fixed_hinges"),
                                moving ? [str(label,"_door"),"doors"] : ["body"])
        color(METAL) for(f=fractions) {
            z=z0+height*f;
            if(!moving) rod([HINGE.x,HINGE.y,z],2*hinge_radius_mm,hinge_height_mm);
            block([HINGE.x-hinge_leaf_thickness_mm/2,HINGE.y+(moving ? -hinge_leaf_half_span_mm : door_clearance_mm/2),z-hinge_height_mm/2],
                  [0.2,HINGE.y+(moving ? -door_clearance_mm/2 : hinge_leaf_half_span_mm),z+hinge_height_mm/2]);
        }
}
module door_assembly(label,z0,height,roof) {
    door_hinges(label,z0,height,false);
    pivot_at(HINGE,door_open_angle_deg*door_fraction(label),[0,0,-1]) {
        door_hinges(label,z0,height,true);
        translate([0,0,z0]) {
            component(str(label,"_shell"),[str(label,"_door"),"doors"]) color(WHITE) door_shell(height,roof);
            component(str(label,"_inner_lining"),[str(label,"_door"),"doors"]) color(BLACK) door_inner_lining(height,roof);
            door_latch(label,height);
        }
    }
}

module chamber_linings() {
    color(DARK) for(upper=[false,true]) {
        label=upper ? "upper" : "lower";
        z0=upper ? UPPER_Z+4 : lower_floor_z_mm;
        z1=upper ? BODY.z-7 : LOWER_TOP-2;
        component(str(label,"_back_lining"),["interior",str(label,"_interior")])
            block([6,250-EPS,z0],[door_width_mm-3,HINGE_Y+door_clearance_mm/2,z1]);
        component(str(label,"_right_lining"),["interior",str(label,"_interior")])
            block([door_width_mm-6,40,z0],[door_width_mm-3,250,z1]);
    }
}
module upper_sample_dish() {
    dish_z=UPPER_Z+upper_dish_above_door_bottom_mm;
    center=[door_width_mm*upper_dish_x_fraction,upper_dish_y_mm,dish_z];
    component("upper_sample_tray",["interior","upper_interior"]) color(DARK)
        block([12,40,UPPER_Z],[door_width_mm-7,244,UPPER_Z+4]);
    translate(center) color(METAL) {
        component("upper_dish",["interior","upper_interior","upper_dish_wall"])
            render(convexity=10) difference() {
                translate([0,0,-3-EPS]) cylinder(d=upper_dish_outer_diameter_mm,h=upper_dish_depth_mm+3+EPS);
                cylinder(d1=upper_dish_inner_bottom_diameter_mm,d2=upper_dish_inner_top_diameter_mm,h=upper_dish_depth_mm+EPS);
            }
        component("upper_sample_post",["interior","upper_interior"]) {
            rod([0,0,6],34,12);
            rod([0,0,18],18,12);
            rod([0,0,(24+upper_post_above_floor_mm)/2],6,upper_post_above_floor_mm-24);
        }
    }
}
module lower_chamber_structure() {
    color(DARK) {
        // The small left plinth supports the screw foot, not the entire room wall.
        component("lower_left_infill",["interior","lower_interior"])
            block([6,40,lower_floor_z_mm-2],[ROOM_LEFT,250,lower_floor_z_mm+lower_left_infill_height_mm]);
        component("lower_right_infill",["interior","lower_interior"])
            block([ROOM_RIGHT,40,lower_floor_z_mm-2],[door_width_mm-6,250,LOWER_TOP-2]);
        component("lower_upper_stage",["interior","lower_interior"])
            block([ROOM_LEFT,77,lower_stage_z_mm],[ROOM_RIGHT,211,lower_stage_z_mm+lower_stage_thickness_mm]);
        component("lower_rear_column",["interior","lower_interior"])
            block([CENTER_X-33.5,212,lower_floor_z_mm],[CENTER_X+33.5,241,lower_stage_z_mm]);
        component("lower_recess_wall",["interior","lower_interior","lower_floor"])
            render(convexity=10) difference() {
                block([ROOM_LEFT,40,lower_recess_bottom_z_mm],[ROOM_RIGHT,lower_floor_rear_y_mm,lower_floor_z_mm]);
                translate([CENTER_X,lower_pin_y_mm,lower_recess_bottom_z_mm-EPS])
                    cylinder(d=lower_recess_diameter_mm,h=lower_floor_z_mm-lower_recess_bottom_z_mm+2*EPS);
            }
    }
    component("lower_recess_bottom",["interior","lower_interior","lower_floor"]) color(BLACK)
        rod([CENTER_X,lower_pin_y_mm,lower_recess_bottom_z_mm-1],lower_recess_diameter_mm+2,2);
}
module lower_sample_mechanism() {
    color(METAL) {
        component("lower_guides",["interior","lower_interior"])
            for(side=[-1,1]) rod([CENTER_X+side*lower_guide_spacing_mm/2,lower_guide_y_mm,
                                 (lower_floor_z_mm+lower_guide_top_z_mm)/2],
                                side<0 ? lead_screw_diameter_mm : right_guide_diameter_mm,
                                lower_guide_top_z_mm-lower_floor_z_mm);
        component("lower_pins",["interior","lower_interior"])
            for(side=[-1,1]) rod([CENTER_X+side*lower_pin_spacing_mm/2,lower_pin_y_mm,lower_stage_z_mm-lower_pin_length_mm/2],
                                side<0 ? small_pin_diameter_mm : second_pin_diameter_mm,lower_pin_length_mm);
        component("lower_central_piece",["interior","lower_interior"])
            rod([CENTER_X,lower_pin_y_mm,central_piece_bottom_z_mm+central_piece_height_mm/2],central_piece_diameter_mm,central_piece_height_mm);
        component("lower_bearing",["interior","lower_interior"]) {
            rod([CENTER_X,lower_pin_y_mm,lower_stage_z_mm-5],bearing_outer_diameter_mm,10);
            rod([CENTER_X,lower_pin_y_mm,lower_stage_z_mm-12.5],40,5);
            if(show_details()) rod([CENTER_X+22,101,lower_stage_z_mm-6],8,12,"x");
        }
    }
    component("lower_guide_feet",["interior","lower_interior"]) color(BLACK)
        for(side=[-1,1]) rod([CENTER_X+side*lower_guide_spacing_mm/2,lower_guide_y_mm,lower_floor_z_mm+6],16,12);
}
module selector_assembly() {
    origin=[(RECESS_LEFT+RECESS_RIGHT)/2,PANEL_Y,recess_bottom_z_mm+selector_height_fraction*(recess_top_z_mm-recess_bottom_z_mm)];
    component("selector_bezel",["selector"]) color(DARK)
        translate(origin) rotate([90,0,0]) ring([0,0,0],12,8.6,1);
    translate(origin+[0,-selector_travel_mm*selector_fraction(),0]) {
        component("selector_head",["selector"]) color(BLACK)
            rod([0,-selector_face_offset_mm+selector_head_thickness_mm/2,0],selector_head_diameter_mm,selector_head_thickness_mm,"y");
        component("selector_neck",["selector"]) color(DARK)
            rod([0,-(selector_face_offset_mm-selector_head_thickness_mm+1)/2,0],selector_neck_diameter_mm,selector_face_offset_mm-selector_head_thickness_mm-1,"y");
        component("selector_shaft",["selector"]) color(METAL)
            rod([0,(selector_travel_mm+2)/2,0],selector_shaft_diameter_mm,selector_travel_mm+10,"y");
    }
}
module c13534() {
    fixed_housing();
    door_assembly("upper",UPPER_Z,upper_door_height_mm,true);
    door_assembly("lower",LOWER_Z,lower_door_height_mm,false);
    chamber_linings();
    upper_sample_dish();
    lower_chamber_structure();
    lower_sample_mechanism();
    selector_assembly();
}
c13534();
