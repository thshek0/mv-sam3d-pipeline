// Quantaurus-Tau, low sample box. Brochure p5: 450 x 500 x 253.3 mm.
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
BODY_H=overall_height_mm-lid_height_mm;
FLOOR_Z=BODY_H-chamber_depth_mm;
LEFT=chamber_center_x_mm-clear_width_mm/2-wall_mm;
RIGHT=LEFT+clear_width_mm+2*wall_mm;
FRONT=-front_projection_mm;
REAR=chamber_rear_y_mm;
HOLE_CENTER=[chamber_center_x_mm,(FRONT+wall_mm+REAR)/2];
HINGE=[LEFT,REAR,BODY_H+lid_clearance_mm];
assert(floor_small_hole_diameter_mm<floor_hole_pattern_radius_mm/3);

module fixed_housing() {
    color(DARK) {
        component("rear_housing",["body"]) block([0,REAR,0],[body_width_mm,body_depth_mm,BODY_H]);
        component("front_plinth",["body"]) block([0,0,0],[body_width_mm,REAR,plinth_height_mm]);
        component("side_housings",["body"]) {
            block([0,0,plinth_height_mm],[LEFT,REAR,BODY_H]);
            block([RIGHT,0,plinth_height_mm],[body_width_mm,REAR,BODY_H]);
        }
        component("front_enclosure_panel",["body"])
            block([LEFT-EPS,FRONT,plinth_height_mm],[RIGHT+EPS,FRONT+wall_mm,FLOOR_Z-floor_thickness_mm]);
        component("front_corners",["body"])
            for(side=[-1,1]) translate([side<0 ? LEFT : RIGHT,0,plinth_height_mm])
                linear_extrude(BODY_H-plinth_height_mm)
                    intersection() {
                        // This curve defines the enclosure silhouette, not a fillet.
                        circle(r=front_projection_mm);
                        translate([side<0 ? -front_projection_mm : 0,FRONT])
                            square([front_projection_mm,front_projection_mm]);
                    }
    }
}
module chamber_shell() {
    render(convexity=10) difference() {
        block([LEFT-EPS,FRONT,FLOOR_Z-floor_thickness_mm],[RIGHT+EPS,REAR,BODY_H]);
        block([LEFT+wall_mm,FRONT+wall_mm,FLOOR_Z-floor_thickness_mm-EPS],
              [RIGHT-wall_mm,REAR+EPS,BODY_H+EPS]);
        block([chamber_center_x_mm-grip_width_mm/2,FRONT-EPS,BODY_H-grip_depth_mm],
              [chamber_center_x_mm+grip_width_mm/2,FRONT+wall_mm+EPS,BODY_H+EPS]);
    }
}
module perforated_floor() {
    // User-confirmed topology: eight small through-holes PLUS one centre hole.
    // Pattern radius and small diameter are estimates, independently editable.
    render(convexity=20) difference() {
        block([LEFT+wall_mm,FRONT+wall_mm,FLOOR_Z-floor_thickness_mm],
              [RIGHT-wall_mm,REAR,FLOOR_Z]);
        translate([HOLE_CENTER.x,HOLE_CENTER.y,FLOOR_Z-floor_thickness_mm-EPS]) {
            cylinder(d=hole_diameter_mm,h=floor_thickness_mm+2*EPS);
            for(angle=[0:45:315])
                translate(floor_hole_pattern_radius_mm*[cos(angle),sin(angle),0])
                    cylinder(d=floor_small_hole_diameter_mm,h=floor_thickness_mm+2*EPS);
        }
    }
}
module chamber() {
    component("chamber_shell",["body"]) color(DARK) chamber_shell();
    component("perforated_floor",["floor"]) color(BLACK) perforated_floor();
    component("grip_recess_back",["body"]) color(BLACK)
        block([chamber_center_x_mm-grip_width_mm/2-wall_mm,FRONT+wall_mm,BODY_H-grip_depth_mm-wall_mm],
              [chamber_center_x_mm+grip_width_mm/2+wall_mm,FRONT+2*wall_mm,BODY_H]);
}
module lid_assembly() {
    pivot_at(HINGE,open_angle_deg*motion_fraction(),[-1,0,0])
        component("sample_box_lid",["lid","cover"]) color(DARK)
            rounded_x([LEFT,FRONT,BODY_H+lid_clearance_mm],[RIGHT,REAR,overall_height_mm],lid_corner_radius_mm);
}
module chamber_hardware() {
    if(show_details()) color(METAL) {
        component("rear_adjusters",["hardware"]) for(x=[LEFT+knob_inset_mm,RIGHT-knob_inset_mm]) {
            rod([x,REAR-knob_shaft_offset_mm,BODY_H-knob_below_rim_mm],knob_shaft_diameter_mm,knob_shaft_length_mm,"y");
            rod([x,REAR-knob_offset_mm,BODY_H-knob_below_rim_mm],knob_diameter_mm,knob_length_mm,"y");
        }
        component("front_latch_mounts",["hardware"]) for(x=[LEFT+latch_inset_mm,RIGHT-latch_inset_mm])
            translate([x-latch_width_mm/2,FRONT+wall_mm,FLOOR_Z])
                cube([latch_width_mm,latch_depth_mm,latch_height_mm]);
    }
}
module c16361() { fixed_housing(); chamber(); lid_assembly(); chamber_hardware(); }
c16361();
