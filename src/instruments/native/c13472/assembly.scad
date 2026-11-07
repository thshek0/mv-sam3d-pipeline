// C13472 orientation system. User approximate envelope; local internals estimated.
include <parameters_native.scad>
include <common.scad>
/* [View and motion] */
pose="closed"; // [closed,open,custom]
opening=0; // [0:0.01:1]
animate_motion=false;
/* [Geometry] */
part="all";
details=true;
fillets=true;
simplified=false;
/* [Hidden] */
$fn=64;
BODY=[body_width_mm,body_depth_mm,body_height_mm];
BASE=BODY.z*base_height_fraction;
SHOULDER=BODY.z*shoulder_height_fraction;
BEVEL=BODY.y*front_bevel_fraction;
REAR=BODY.y*rear_position_fraction;
FLOOR=BASE+wall_mm;
WALL=BODY.x*inner_wall_fraction;
LEFT_WALL=BODY.x*0.032;
HOOD_LEFT=BODY.x*cover_left_fraction;
HOOD_WIDTH=BODY.x*cover_width_fraction;
HOOD_BOTTOM=BASE+10;
RAIL_CLEARANCE=0.1; // modeled clearance, not measured fit
STAGE_ORIGIN=[BODY.x*stage_center_x_fraction,BODY.y*stage_center_y_fraction,FLOOR];
SLIDE_Z=56.6;
MOTOR_LENGTH=40; MOTOR_OFFSET=44;
MOTOR_ORIGIN=[WALL-motor_wall_inset_mm-MOTOR_OFFSET,STAGE_ORIGIN.y+15,FLOOR+motor_above_floor_mm];
assert(cover_left_fraction+cover_width_fraction+cover_travel_fraction<=1);
assert(holder_hole_diameter_mm<holder_collar_diameter_mm && holder_collar_diameter_mm<holder_frame_width_mm);

// Side profiles are explicit manufacturing silhouettes, not stacks of surface patches.
function housing_profile() = [[0,BASE-EPS],[REAR,BASE-EPS],[REAR,BODY.z],[BEVEL,BODY.z],[0,SHOULDER]];
function hood_profile() = [[0,HOOD_BOTTOM],[REAR-4-RAIL_CLEARANCE,HOOD_BOTTOM],
    [REAR-4-RAIL_CLEARANCE,BODY.z],[BEVEL,BODY.z],[0,SHOULDER]];
function hood_void_profile() = [[wall_mm,HOOD_BOTTOM-EPS],[REAR+EPS,HOOD_BOTTOM-EPS],
    [REAR+EPS,BODY.z-wall_mm],[BEVEL,BODY.z-wall_mm],[wall_mm,SHOULDER]];

module fixed_housing() {
    color(WHITE) {
        component("base_housing",["body"]) block([0,0,0],[BODY.x,BODY.y,BASE]);
        component("left_wall",["body"]) yz_prism(housing_profile(),0,LEFT_WALL);
        component("right_inner_housing",["body"])
            yz_prism([[10,BASE-EPS],[REAR,BASE-EPS],[REAR,BODY.z-10],
                      [BEVEL+10,BODY.z-10],[10,SHOULDER-10]],WALL,BODY.x);
    }
    component("back_housing",["body"]) color(DARK)
        block([0,REAR-EPS,BASE-EPS],BODY);
    color(BLACK) {
        component("chamber_floor",["body"]) block([LEFT_WALL,12,BASE-EPS],[WALL,REAR,FLOOR]);
        component("side_black_wall_lining",["body"])
            block([WALL-1,10,FLOOR],[WALL,REAR,FLOOR+181.6]);
    }
    component("slide_rails",["body"]) color(METAL) {
        block([LEFT_WALL,0,BASE-EPS],[BODY.x,10,BASE+8]);
        block([LEFT_WALL,REAR-4,BODY.z-9],[BODY.x,REAR,BODY.z-4]);
    }
}
module hood_shell() {
    grip_x=BODY.x*0.045;
    grip_width=BODY.x*0.026;
    render(convexity=10) difference() {
        yz_prism(hood_profile(),0,HOOD_WIDTH);
        yz_prism(hood_void_profile(),-EPS,HOOD_WIDTH+EPS);
        block([grip_x,-EPS,HOOD_BOTTOM+25],[grip_x+grip_width,wall_mm+EPS,SHOULDER-35]);
    }
}
module sliding_hood() {
    translate([HOOD_LEFT+BODY.x*cover_travel_fraction*motion_fraction(),0,0]) color(DARK) {
        component("cover_shell",["cover"]) hood_shell();
        component("cover_grip_back",["cover"])
            block([BODY.x*0.045-1,wall_mm,HOOD_BOTTOM+24],
                  [BODY.x*0.071+1,2*wall_mm,SHOULDER-34]);
    }
}
module rotary_stage() {
    component("stage_rotary_base",["interior","stage"]) color(DARK) rod([0,0,6],2*stage_radius_mm,12);
    component("stage_rotary_top",["interior","stage"]) color(METAL) rod([0,0,17],1.8*stage_radius_mm,10);
    color(DARK) {
        component("stage_pedestal",["interior","stage"]) block([-40,-40,22],[40,40,SLIDE_Z]);
        component("stage_lower_slide",["interior","stage"]) block([-52,-48,SLIDE_Z],[52,48,SLIDE_Z+14]);
        component("stage_upper_slide",["interior","stage"]) block([-47,-44,SLIDE_Z+18],[47,44,SLIDE_Z+32]);
        component("holder_supports",["interior","holder"])
            for(side=[-1,1]) scale([side,1,1]) block([27,-35,SLIDE_Z+32],[38,35,holder_above_floor_mm-8]);
    }
    component("stage_rails",["interior","stage"]) color(METAL)
        for(y=[-32,32]) rod([0,y,SLIDE_Z+16],8,105,"x");
}
module micrometer(position,axis,direction) {
    rod(position,14,30,axis);
    rod(position-22*direction,18,14,axis);
    rod(position+22*direction,4,25,axis);
}
module holder_frame() {
    half=holder_frame_width_mm/2;
    render(convexity=10) difference() {
        block([-half,-half,holder_above_floor_mm-8],[half,half,holder_above_floor_mm]);
        translate([0,0,holder_above_floor_mm-8-EPS]) cylinder(d=holder_hole_diameter_mm,h=8+2*EPS);
    }
}
module sample_stage() {
    translate(STAGE_ORIGIN) {
        rotary_stage();
        component("holder_frame",["holder","interior"]) color(DARK) holder_frame();
        component("holder_collar",["holder","interior"]) color(METAL)
            ring([0,0,holder_above_floor_mm-EPS],holder_collar_diameter_mm,holder_hole_diameter_mm,4+EPS);
        if(show_details()) color(METAL) {
            component("micrometers",["interior","stage"]) {
                micrometer([-74,-35,SLIDE_Z+25],"x",[1,0,0]);
                micrometer([35,-70,SLIDE_Z+6],"y",[0,1,0]);
            }
            component("holder_frame_screws",["holder","interior"])
                for(x=[-37,37],y=[-37,37]) rod([x,y,holder_above_floor_mm+2],8,4);
        }
    }
}
module side_motor() {
    translate(MOTOR_ORIGIN) {
        component("side_rotary_plate",["side_unit","interior"]) color(DARK) rod([0,0,0],96,9,"x");
        component("side_rotary_hub",["side_unit","interior"]) color(BLACK) rod([-20,0,0],40,32,"x");
        component("side_motor_body",["side_unit","interior"]) color(DARK) rod([24,0,0],2*motor_radius_mm,MOTOR_LENGTH,"x");
        component("side_motor_foot",["side_unit","interior"]) color(DARK)
            block([6,-25,-motor_above_floor_mm],[45,25,-10]);
        if(show_details()) component("side_plate_screws",["side_unit","interior"]) color(METAL)
            for(a=[0:90:270]) rod([-6,40*cos(a),40*sin(a)],6,3,"x");
    }
    component("side_wall_mount",["side_unit","interior"]) color(BLACK)
        block([WALL-motor_mount_thickness_mm,MOTOR_ORIGIN.y-38,MOTOR_ORIGIN.z-40],
              [WALL,MOTOR_ORIGIN.y+38,MOTOR_ORIGIN.z+40]);
}
module c13472() { fixed_housing(); sliding_hood(); sample_stage(); side_motor(); }
c13472();
