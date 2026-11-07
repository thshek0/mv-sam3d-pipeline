// Reusable native solids. Millimetres; X right, Y rear, Z up.
module block(lo,hi) { assert(min(hi-lo)>0,"Non-positive block"); translate(lo) cube(hi-lo); }
module xy_prism(points,z0,z1) { translate([0,0,z0]) linear_extrude(z1-z0) hull() polygon(points); }
module yz_prism(points,x0,x1) {
    multmatrix([[0,0,1,x0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])
        linear_extrude(x1-x0) polygon(points);
}
module rounded_x(lo,hi,r) {
    radius=min(r,(hi.y-lo.y)*.24,(hi.z-lo.z)*.24);
    hull() for(y=[lo.y+radius,hi.y-radius],z=[lo.z+radius,hi.z-radius])
        translate([lo.x,y,z]) rotate([0,90,0]) cylinder(r=radius,h=hi.x-lo.x);
}
module rod(center,d,length,axis="z") {
    translate(center) rotate(axis=="x" ? [0,90,0] : axis=="y" ? [90,0,0] : [0,0,0])
        cylinder(d=d,h=length,center=true);
}
module annulus(center,outer_d,inner_d,height) {
    assert(outer_d>inner_d && inner_d>0);
    translate(center) difference() {
        cylinder(d=outer_d,h=height);
        translate([0,0,-.01]) cylinder(d=inner_d,h=height+.02);
    }
}
module plate_hole(lo,hi,center,d,z0,z1) {
    difference() {
        block([lo.x,lo.y,z0],[hi.x,hi.y,z1]);
        translate([center.x,center.y,z0-.01]) cylinder(d=d,h=z1-z0+.02);
    }
}
module plate_holes(lo,hi,holes,z0,z1) {
    render(convexity=20) difference() {
        block([lo.x,lo.y,z0],[hi.x,hi.y,z1]);
        for(hole=holes)
            translate([hole.x,hole.y,z0-.01]) cylinder(d=hole.z,h=z1-z0+.02);
    }
}
module radial_profile(center,profile,axis="z",segments=16) {
    translate(center) rotate(axis=="x" ? [0,90,0] : axis=="y" ? [-90,0,0] : [0,0,0])
        rotate_extrude() polygon(profile);
}
module quarter_disc(edge,r,z0,z1,side) {
    translate([edge,0,z0]) linear_extrude(z1-z0) intersection() {
        circle(r=r); translate([side<0 ? -r:0,-r]) square([r,r]);
    }
}
module ellipsoid(center,radii) { translate(center) scale(radii) sphere(r=1); }
module capsule_y(center,width,height,depth) {
    hull() for(dz=[-(height-width)/2,(height-width)/2]) rod(center+[0,0,dz],width,depth,"y");
}
