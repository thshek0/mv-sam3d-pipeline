"""Export the current authored HARMYNTA profile as a verified millimetre STEP solid.

Requires pythonocc-core and scipy. Builds native surfaces of revolution directly
from the authoring profile, without reconstructing CAD from mesh triangles.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad

from OCC.Core.gp import gp_Pnt, gp_Ax1, gp_Dir
from OCC.Core.Geom import Geom_BSplineCurve
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger, TColStd_SequenceOfAsciiString
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeRevol
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Check
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_IN, TopAbs_OUT
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_Reader, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

ROOT=Path(__file__).resolve().parents[2]
ASSET=ROOT/'output/objects/hestia-rl/plate_harmynta_white_22cm/codex'
INPUT=ROOT/'src/plates/inputs/plate_harmynta_white_22cm.json'
FIT_REPORT=INPUT.with_name('plate_harmynta_white_22cm_photo_fit.json')
AUDIT=ASSET/'intermediate/solidworks_20261107'
STEP=ASSET/'products/HARMYNTA_220mm.step'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def spline_curve(interpolant):
    """Represent a piecewise Hermite cubic exactly as one native B-spline."""
    xx=interpolant.x;yy=interpolant(xx);derivative=interpolant.derivative()(xx)
    poles=[]
    for i in range(len(xx)-1):
        span=xx[i+1]-xx[i]
        if i==0:poles.append([xx[i],yy[i]])
        poles.extend([[xx[i]+span/3,yy[i]+derivative[i]*span/3],
                      [xx[i+1]-span/3,yy[i+1]-derivative[i+1]*span/3],
                      [xx[i+1],yy[i+1]]])
    points=TColgp_Array1OfPnt(1,len(poles))
    for i,(r,z) in enumerate(poles,1):points.SetValue(i,gp_Pnt(float(r),0,float(z)))
    knots=TColStd_Array1OfReal(1,len(xx));mult=TColStd_Array1OfInteger(1,len(xx))
    for i,r in enumerate(xx,1):knots.SetValue(i,float(r));mult.SetValue(i,4 if i in (1,len(xx)) else 3)
    curve=Geom_BSplineCurve(points,knots,mult,3,False)
    for i in range(curve.NbKnots()-1,1,-1):curve.RemoveKnot(i,2,1e-9)
    error=max(abs(curve.Value(float(r)).Z()-float(interpolant(r))) for r in np.linspace(xx[0],xx[-1],1001))
    assert error<1e-7,error
    return curve


def top_curve(profile):
    source=PchipInterpolator(profile[:,0],profile[:,1])
    samples=np.linspace(40,110,7001)
    knots=[40,43.85,51.25,60,70,80,90,100,106,108,109,109.7,110]
    for _ in range(100):
        fitted=PchipInterpolator(knots,source(knots));error=abs(fitted(samples)-source(samples))
        index=int(np.argmax(error))
        if error[index]<=.005:return fitted,float(error[index])
        knots=sorted(set(knots+[float(samples[index])]))
    raise ValueError('Cannot approximate the inner profile within 0.005 mm')


def properties(shape):
    box=Bnd_Box();brepbndlib.AddOptimal(shape,box,False,False)
    # Integrate spline spans adaptively; default fixed quadrature misses knots.
    props=GProp_GProps()
    integration_error=brepgprop.VolumePropertiesGK(shape,props,1e-10,True,True)
    assert integration_error>=0,integration_error
    counts={}
    for label,kind in [('solids',TopAbs_SOLID),('shells',TopAbs_SHELL),('faces',TopAbs_FACE)]:
        explorer=TopExp_Explorer(shape,kind);count=0
        while explorer.More():count+=1;explorer.Next()
        counts[label]=count
    return {**counts,'bounds_mm':list(box.Get()),'volume_mm3':props.Mass(),
            'valid':bool(BRepCheck_Analyzer(shape,True).IsValid())}


def build():
    AUDIT.mkdir(parents=True,exist_ok=True);STEP.parent.mkdir(parents=True,exist_ok=True)
    p=json.loads(INPUT.read_text());profile=np.array(p['profile_radius_top_bottom_mm'])
    fit=p['profile_fit'];assert 'knots_radius_mm' in fit,'Requires the reviewed photo-fit revision'
    accepted_fit=json.loads(FIT_REPORT.read_text())
    for key in ['knots_radius_mm','knots_bottom_mm']:
        np.testing.assert_allclose(fit[key],accepted_fit[key],rtol=0,atol=1e-9,
                                   err_msg='Update the accepted fit report with the profile')
    bottom=PchipInterpolator(fit['knots_radius_mm'],fit['knots_bottom_mm'])
    top,top_error=top_curve(profile)
    f=p['foot_ring_measurements_mm'];inner=f['inner_diameter']/2;outer=f['outer_diameter']/2
    ring_r=np.array([0,inner,inner+.12,inner+.35,outer-.35,outer-.12,outer])
    ring_z=np.array([f['central_recess_depth_estimate'],f['central_recess_depth_estimate'],.5,0,0,.5,f['height']])
    np.testing.assert_allclose(np.interp(ring_r,profile[:,0],profile[:,2]),ring_z,atol=1e-6)
    wire=BRepBuilderAPI_MakeWire()
    def line(a,b):wire.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(float(a[0]),0,float(a[1])),gp_Pnt(float(b[0]),0,float(b[1]))).Edge())
    for i in range(len(ring_r)-1):line((ring_r[i],ring_z[i]),(ring_r[i+1],ring_z[i+1]))
    wire.Add(BRepBuilderAPI_MakeEdge(spline_curve(bottom)).Edge())
    line((110,float(bottom(110))),(110,float(top(110))))
    edge=BRepBuilderAPI_MakeEdge(spline_curve(top)).Edge();edge.Reverse();wire.Add(edge)
    line((40,float(top(40))),(0,7));line((0,7),(0,ring_z[0]))
    assert wire.IsDone()
    face=BRepBuilderAPI_MakeFace(wire.Wire(),True).Face()
    revol=BRepPrimAPI_MakeRevol(face,gp_Ax1(gp_Pnt(0,0,0),gp_Dir(0,0,1)),2*np.pi,True)
    assert revol.IsDone();solid=revol.Shape()
    if properties(solid)['volume_mm3']<0:solid.Reverse()
    original=properties(solid)
    assert original['valid'] and original['solids']==1 and original['shells']==1,original
    assert BRepAlgoAPI_Check(solid,True,True).IsValid(),'CAD has small-edge or self-interference errors'
    print('Native revolved solid valid:',original,flush=True)
    Interface_Static.SetCVal('write.step.schema','AP214IS')
    Interface_Static.SetCVal('write.step.unit','MM');Interface_Static.SetCVal('xstep.cascade.unit','MM')
    writer=STEPControl_Writer();assert writer.Transfer(solid,STEPControl_AsIs)==IFSelect_RetDone
    assert writer.Write(str(STEP))==IFSelect_RetDone
    reader=STEPControl_Reader();assert reader.ReadFile(str(STEP))==IFSelect_RetDone
    units=[TColStd_SequenceOfAsciiString() for _ in range(3)];reader.FileUnits(*units)
    length_units=[units[0].Value(i).ToCString() for i in range(1,units[0].Length()+1)]
    assert length_units and all('milli' in s.lower() for s in length_units),length_units
    assert reader.TransferRoots()>0;restored=reader.OneShape();after=properties(restored)
    assert after['valid'] and after['solids']==1 and after['shells']==1,after
    assert BRepAlgoAPI_Check(restored,True,True).IsValid(),'Re-imported STEP has topology errors'
    np.testing.assert_allclose(after['bounds_mm'],[-110,-110,0,110,110,40],atol=1e-5)
    np.testing.assert_allclose(after['volume_mm3'],original['volume_mm3'],rtol=1e-8)
    def bfun(r):return float(np.interp(r,ring_r,ring_z)) if r<=outer else float(bottom(r))
    def tfun(r):return 7 if r<=40 else float(top(r))
    # Independent integral of the radial shell, split at every polynomial knot.
    breaks=np.unique(np.r_[ring_r,40,bottom.x,top.x])
    integral=sum(quad(lambda r:2*np.pi*r*(tfun(r)-bfun(r)),a,b,epsabs=1e-7)[0] for a,b in zip(breaks[:-1],breaks[1:]))
    # OCC's adaptive surface quadrature and the polynomial radial integral
    # agree within 10 ppm; geometry is checked independently below.
    assert abs(after['volume_mm3']/integral-1)<1e-5,(after['volume_mm3'],integral)
    gap=min(tfun(r)-bfun(r) for r in np.linspace(0,110,11001));assert gap>.49
    probes={'floor_material':([0,0,4.5],True),'foot_material':([47.5,0,.25],True),
            'dish_cavity':([0,0,20],False),'underside_recess':([0,0,1],False),
            'outer_wall_material':([80,0,(bfun(80)+tfun(80))/2],True),
            'above_inner_wall':([80,0,tfun(80)+.5],False)}
    classifier=BRepClass3d_SolidClassifier(restored)
    for label,(point,inside) in probes.items():
        classifier.Perform(gp_Pnt(*point),1e-7)
        assert classifier.State()==(TopAbs_IN if inside else TopAbs_OUT),label
    boundary=TopExp_Explorer(restored,TopAbs_SHELL).Current()
    surface_errors=[];boundary_points=[]
    section_samples=[(inner,ring_z[0]),(outer,5),(110,38.7),(109,40)]
    section_samples += [(r,bfun(r)) for r in np.linspace(0,110,111)]
    section_samples += [(r,tfun(r)) for r in np.linspace(0,110,111)]
    for r,z in section_samples:
        for angle in [0,17,123,277]:
            a=np.deg2rad(angle);point=[r*np.cos(a),r*np.sin(a),float(z)]
            boundary_points.append(point)
            vertex=BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex()
            distance=BRepExtrema_DistShapeShape(vertex,boundary);distance.Perform();assert distance.IsDone()
            surface_errors.append(distance.Value())
    assert max(surface_errors)<1e-6,max(surface_errors)
    report={'file':str(STEP.relative_to(ROOT)),'sha256':digest(STEP),'format':'STEP AP214; single native B-rep solid',
        'units':length_units,'after_step_reimport':after,'independent_radial_integral_mm3':integral,
        'volume_relative_error':abs(after['volume_mm3']/integral-1),'minimum_vertical_shell_gap_mm':gap,
        'volume_crosscheck_relative_tolerance':1e-5,'boundary_samples_checked':len(surface_errors),
        'top_curve_approximation_error_mm':top_error,'measured_ring':f,
        'max_measured_edge_and_rotational_surface_error_mm':max(surface_errors),
        'solid_classification_checks':list(probes),'self_interference_check':'passed before and after STEP roundtrip',
        'source_hashes':{str(path.relative_to(ROOT)):digest(path) for path in [INPUT,Path(__file__),FIT_REPORT]},
        'limitations':['Not opened in SolidWorks; STEP reopened and checked with Open CASCADE.',
            'Physical fit against the gold print has not been tested.',
            'Outer curvature is photo-fitted; wall thickness, edge rounding and central recess are estimates.',
            'Measured diameters imply 7.40 mm radial width; separate 7.82 mm reading is retained as inconsistent evidence.']}
    (AUDIT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    np.save(AUDIT/'verified_boundary_points_mm.npy',np.array(boundary_points))
    # Keep the standard CAD source registry current after every successful build.
    def reference(path):
        return {'path':os.path.relpath(path,ASSET),'sha256':digest(path)}
    registry={
        'schema_version':1,'source_kind':'radial_profile_cad','units':'mm',
        'frame':'XY centre, Z up; lowest ring contact Z=0',
        'sources':[reference(path) for path in
                   [Path(__file__),INPUT,FIT_REPORT,AUDIT/'verification.json']],
        'rebuild':[sys.executable,'-m','src.plates.solidworks'],
        'variants':{'HARMYNTA_220mm':{'step':reference(STEP)}},
        'native_files':{},'export_directory':'exports',
        'policy':'One current STEP under products; explicit deliveries only.',
        'dependency_policy':{'experimental_method_inputs_allowed':False,
                             'registered_paths_checked_after_resolving_symlinks':True}}
    (ASSET/'solidworks_asset.json').write_text(json.dumps(registry,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();build()
