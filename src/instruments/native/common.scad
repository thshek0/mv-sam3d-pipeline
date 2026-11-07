// Native authoring helpers. Millimetres, degrees. X right, Y rear, Z up.
// Cosmetic edge treatments can be disabled; functional circular cuts never are.
EPS = 0.02;
WHITE = "#e0e0d7";
DARK = "#373e46";
BLACK = "#17191c";
METAL = "#aeb5bb";
function edge_radius(r) = fillets && !simplified ? r : 0;
function show_details() = details && !simplified;
function motion_fraction() = animate_motion ? 1-abs(2*$t-1) : pose=="open" ? 1 : pose=="custom" ? opening : 0;
function selected(name, groups=[]) = part=="all" || part==name || len([for(g=groups) if(g==part) g])>0;
module component(name, groups=[]) { if (selected(name,groups)) children(); }
module block(lo,hi) {
    assert(min(hi-lo)>0, "Non-positive block");
    translate(lo) cube(hi-lo);
}
// Corner order: lower-left, lower-right, upper-right, upper-left in XY.
// Polygon arcs avoid tiny fake squares at sharp corners; preserve exact bounds.
module corner_rectangle(size, radii=[0,0,0,0]) {
    assert(min(size)>0 && min(radii)>=0 && 2*max(radii)<=min(size));
    corners=[[0,0],[size.x,0],size,[0,size.y]];
    inward=[[1,1],[-1,1],[-1,-1],[1,-1]];
    polygon([for(i=[0:3]) each
        radii[i]==0 ? [corners[i]] :
        [for(a=[180+i*90:90/12:270+i*90])
            corners[i]+radii[i]*inward[i]+radii[i]*[cos(a),sin(a)]]]);
}
module xy_prism(points,z0,z1) {
    translate([0,0,z0]) linear_extrude(z1-z0) polygon(points);
}
module yz_prism(points,x0,x1) {
    multmatrix([[0,0,1,x0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])
        linear_extrude(x1-x0) polygon(points);
}
module rounded_x(lo,hi,r) {
    radius=edge_radius(min(r,(hi.y-lo.y)/2,(hi.z-lo.z)/2));
    translate(lo) rotate([90,0,90]) linear_extrude(hi.x-lo.x)
        corner_rectangle([hi.y-lo.y,hi.z-lo.z],[radius,radius,radius,radius]);
}
module rod(center,d,length,axis="z") {
    translate(center) rotate(axis=="x" ? [0,90,0] : axis=="y" ? [90,0,0] : [0,0,0])
        cylinder(d=d,h=length,center=true);
}
module ring(center,outer_d,inner_d,height) {
    translate(center) difference() {
        cylinder(d=outer_d,h=height);
        translate([0,0,-EPS]) cylinder(d=inner_d,h=height+2*EPS);
    }
}
module pivot_at(origin,angle,axis) {
    translate(origin) rotate(a=angle,v=axis) translate(-origin) children();
}
module capsule_y(center,width,height,depth) {
    if (fillets && !simplified)
        hull() for(dz=[-(height-width)/2,(height-width)/2])
            rod(center+[0,0,dz],width,depth,"y");
    else translate(center-[width,depth,height]/2) cube([width,depth,height]);
}
