// Cary 60. Authoritative native model; source dimensions in parameters_native.scad.
// Brochure pp15/16: envelope and chamber. Cutaway p5: holder handedness.
include <parameters_native.scad>
include <common.scad>
/* [View and motion] */
pose = "closed"; // [closed,open,custom]
opening = 0; // [0:0.01:1]
animate_motion = false;
/* [Geometry] */
part = "all";
details = true;
fillets = true;
simplified = false;
/* [Hidden] */
$fn = 64;
BODY = [body_width_mm,body_depth_mm,body_height_mm];
CHAMBER_LEFT = chamber_center_x_mm-cover_width_mm/2;
CHAMBER_RIGHT = CHAMBER_LEFT+cover_width_mm;
CHAMBER_REAR = BODY.y-rear_wall_depth_mm;
LID_ORIGIN = [CHAMBER_LEFT,cover_front_y_mm,BODY.z-cover_thickness_mm];
LID_SIZE = [cover_width_mm,cover_length_mm,cover_thickness_mm];
HOLDER_ORIGIN = [chamber_center_x_mm,holder_center_y_mm,chamber_floor_z_mm];
// Small hardware dimensions are appearance estimates, separate from functional sizes.
FOOT_HALF_WIDTH=60; FOOT_REAR=25; FOOT_FRONT=65; FOOT_THICKNESS=8;
BRIDGE_INSET=38; BRIDGE_OUTSET=50; BRIDGE_HALF_DEPTH=19; PLATE_THICKNESS=4;
ADJUSTER_X=25; ADJUSTER_SHAFT_D=6; ADJUSTER_KNOB_D=14;

assert(cover_front_y_mm+cover_length_mm+cover_travel_mm<=BODY.y-receiver_rear_margin_mm);
assert(grip_depth_mm>0 && grip_depth_mm<cover_thickness_mm);
assert(grip_center_offset_x_mm+grip_width_mm/2<cover_width_mm/2);

module fixed_housing() {
    color(WHITE) {
        component("right_housing",["body"])
            rounded_x([CHAMBER_RIGHT,0,0],BODY,housing_radius_mm);
        component("left_end_housing",["body"])
            rounded_x([0,0,0],[CHAMBER_LEFT,BODY.y,BODY.z],housing_radius_mm);
    }
    color(DARK) {
        component("compartment_shell",["body"]) render(convexity=10) difference() {
            block([CHAMBER_LEFT,cover_front_y_mm,0],[CHAMBER_RIGHT,BODY.y,BODY.z-9]);
            block([CHAMBER_LEFT+chamber_wall_mm,cover_front_y_mm-EPS,chamber_floor_z_mm],
                  [CHAMBER_RIGHT-chamber_wall_mm,CHAMBER_REAR,BODY.z+EPS]);
        }
        component("removable_front_door",["body"])
            rounded_x([CHAMBER_LEFT,0,12],[CHAMBER_RIGHT,22,BODY.z-9],9);
        component("fixed_cover_receiver",["body"])
            translate([CHAMBER_LEFT,cover_front_y_mm+cover_length_mm,
                       LID_ORIGIN.z-receiver_gap_mm-receiver_thickness_mm])
                cube([cover_width_mm,cover_travel_mm,receiver_thickness_mm]);
        component("inner_rails",["body"])
            for(x=[CHAMBER_LEFT+12,CHAMBER_RIGHT-13])
                block([x-2,cover_front_y_mm,chamber_floor_z_mm+56],
                      [x+2,cover_front_y_mm+255,chamber_floor_z_mm+101]);
    }
}

// Lid-local XY: X right, Y rear. IMG_9917 places the grip toward the right edge.
// The two functional corner arcs survive cosmetic simplification.
module handgrip_cut() {
    grip_x=cover_width_mm/2+grip_center_offset_x_mm-grip_width_mm/2;
    translate([grip_x,grip_front_offset_mm,cover_thickness_mm-grip_depth_mm])
        linear_extrude(grip_depth_mm+EPS)
            corner_rectangle([grip_width_mm,grip_length_mm],
                [0,grip_corner_radius_mm,0,grip_corner_radius_mm]);
}
module lid_shell() {
    render(convexity=10) difference() {
        cube(LID_SIZE);
        handgrip_cut();
    }
}
module sliding_cover() {
    translate(LID_ORIGIN+[0,cover_travel_mm*motion_fraction(),0])
        component("lid_shell",["cover"]) color(DARK) lid_shell();
}

module holder_support() {
    color(BLACK) {
        component("holder_foot",["holder"])
            linear_extrude(FOOT_THICKNESS)
                polygon([[-FOOT_HALF_WIDTH,-FOOT_REAR],[-FOOT_HALF_WIDTH,FOOT_REAR],
                         [FOOT_HALF_WIDTH,FOOT_REAR],[FOOT_HALF_WIDTH,-FOOT_REAR],[0,-FOOT_FRONT]]);
        component("holder_bridges",["holder"])
            for(side=[-1,1]) scale([side,1,1])
                block([BRIDGE_INSET,-BRIDGE_HALF_DEPTH,FOOT_THICKNESS],
                      [BRIDGE_OUTSET,BRIDGE_HALF_DEPTH,holder_plate_above_floor_mm-PLATE_THICKNESS]);
    }
    component("holder_plate",["holder"]) color(METAL)
        translate([-holder_plate_width_mm/2,-holder_plate_depth_mm/2,
                   holder_plate_above_floor_mm-PLATE_THICKNESS])
            cube([holder_plate_width_mm,holder_plate_depth_mm,PLATE_THICKNESS]);
}
module sample_socket() {
    // Instrument left, as in brochure cutaway p5; adjusters belong on the right.
    translate([-socket_offset_x_mm,0,holder_plate_above_floor_mm])
        render(convexity=10) difference() {
            translate([-socket_outer_width_mm/2,-socket_outer_depth_mm/2,0])
                cube([socket_outer_width_mm,socket_outer_depth_mm,socket_height_mm]);
            translate([-socket_inner_width_mm/2,-socket_inner_depth_mm/2,-EPS])
                cube([socket_inner_width_mm,socket_inner_depth_mm,socket_height_mm+2*EPS]);
        }
}
module holder_adjusters() {
    color(METAL) {
        component("holder_lower_clamp_knob",["holder"]) rod([0,-25,10.5],20,5);
        component("holder_adjuster",["holder"]) translate([ADJUSTER_X,0,holder_plate_above_floor_mm]) {
            rod([0,0,8],ADJUSTER_SHAFT_D,16);
            rod([0,0,18.5],ADJUSTER_KNOB_D,5);
        }
        component("holder_threaded_post",["holder"]) rod([40,-2,holder_plate_above_floor_mm+10],6,20);
        if(show_details()) component("holder_plate_screws",["holder"])
            for(x=[-47,47]) rod([x,16,holder_plate_above_floor_mm+1.5],6,3);
    }
}
module sample_holder() {
    translate(HOLDER_ORIGIN) {
        holder_support();
        component("socket_shell",["holder","socket"]) color(BLACK) sample_socket();
        holder_adjusters();
    }
}
module cary60() {
    fixed_housing();
    sliding_cover();
    sample_holder();
}
cary60();
