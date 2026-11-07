"""Build the authored SCAD directly. Python never reconstructs its geometry."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from .paths import ROOT, SOURCE_ROOT, MODELS

NATIVE = SOURCE_ROOT / 'src/instruments/native'
FILES = ('assembly.scad', 'parameters_native.scad', 'common.scad')


def executable():
    path = os.environ.get('OPENSCAD') or shutil.which('openscad')
    if not path:
        raise RuntimeError('Install OpenSCAD or set OPENSCAD to its executable.')
    return path


def source_hashes(folder):
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in FILES}


def stage_source(model, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        source = NATIVE / 'common.scad' if name == 'common.scad' else NATIVE / model / name
        shutil.copy2(source, folder / name)


def run_engine(source, target, definitions=None, options=()):
    command = [executable(), '-o', str(target), *options]
    for key, value in (definitions or {}).items():
        command += ['-D', f'{key}={json.dumps(value)}']
    command.append(str(source.resolve()))
    run = subprocess.run(command, capture_output=True, text=True, timeout=240)
    target.with_suffix(target.suffix + '.log').write_text(run.stdout + run.stderr)
    if run.returncode or not target.exists() or re.search(r'ERROR:|WARNING:', run.stderr):
        raise RuntimeError(f'{target.name}: OpenSCAD failed\n{run.stdout}\n{run.stderr}')
    return target


def read_parameters(folder, work):
    """Ask OpenSCAD to evaluate dimensions; expressions in the source remain valid."""
    text = (folder / 'parameters_native.scad').read_text()
    names = re.findall(r'^([a-zA-Z_]\w*)\s*=', text, re.M)
    probe = work / 'parameter_probe.scad'
    probe.write_text('include <' + str((folder / 'parameters_native.scad').resolve()) + '>\n' +
                     '\n'.join(f'echo(["{name}", {name}]);' for name in names))
    output = work / 'parameters.echo'
    run_engine(probe, output)
    values = dict(json.loads(line.removeprefix('ECHO: ')) for line in output.read_text().splitlines()
                  if line.startswith('ECHO: '))
    if set(values) != set(names):
        raise ValueError('Could not evaluate all native parameters')
    return values


def parameter_evidence(folder):
    notes = {}
    comment = ''
    for line in (folder / 'parameters_native.scad').read_text().splitlines():
        if line.startswith('// '):
            comment = line[3:]
        elif match := re.match(r'^(\w+)\s*=', line):
            notes[match[1]] = comment
            comment = ''
    return notes


def export_mesh(source, work, name, **definitions):
    path = run_engine(source, work / (name + '.stl'), definitions)
    mesh = trimesh.load(path, force='mesh')
    if not mesh.is_watertight or not mesh.is_volume:
        raise AssertionError(f'{name}: native export is not a positive watertight solid')
    return mesh


def check_geometry(model, folder, work, values, extended=True):
    """Engine checks are independent of the retired Python geometry recipes."""
    source = folder / 'assembly.scad'
    cases = {}

    def mesh(name, **definitions):
        result = export_mesh(source, work, name, **definitions)
        cases[name] = {'definitions': definitions, 'bounds_mm': result.bounds.tolist(),
                       'triangles': len(result.faces), 'volume_mm3': float(result.volume),
                       'watertight': True, 'positive_volume': True}
        return result

    for pose in ('closed', 'open'):
        mesh(pose, pose=pose)
    mesh('mid_motion', animate_motion=True, **{'$t': 0.25})
    mesh('simplified', simplified=True)
    if extended:
        v = values
        if model in ('cary60', 'c13472'):
            part = 'lid_shell' if model == 'cary60' else 'cover'
            axis = 1 if model == 'cary60' else 0
            travel = v['cover_travel_mm'] if axis == 1 else v['body_width_mm'] * v['cover_travel_fraction']
            closed = mesh('cover_closed', part=part)
            for phase, fraction in ((0, 0), (0.25, 0.5), (0.5, 1), (1, 0)):
                moving = mesh(f'animation_{phase}', part=part, animate_motion=True, **{'$t': phase})
                expected = np.zeros((2, 3)); expected[:, axis] = fraction * travel
                np.testing.assert_allclose(moving.bounds - closed.bounds, expected, atol=0.02)
            if model == 'cary60':
                socket = mesh('socket', part='socket_shell')
                expected = (v['socket_outer_width_mm']*v['socket_outer_depth_mm'] -
                            v['socket_inner_width_mm']*v['socket_inner_depth_mm'])*v['socket_height_mm']
                np.testing.assert_allclose(socket.volume, expected, rtol=2e-5)
                assert socket.euler_number == 0, 'Socket must have a through opening'
                simple = mesh('grip_simplified', part='lid_shell', simplified=True)
                np.testing.assert_allclose(simple.volume, closed.volume, rtol=1e-6)
                # Independent volume and position checks for the actual grip removal.
                area = v['grip_width_mm']*v['grip_length_mm'] - 2*v['grip_corner_radius_mm']**2*(1-np.pi/4)
                expected = v['cover_width_mm']*v['cover_length_mm']*v['cover_thickness_mm']-area*v['grip_depth_mm']
                np.testing.assert_allclose(closed.volume, expected, rtol=5e-4)
                section = closed.section(plane_origin=[0,0,v['body_height_mm']-v['grip_depth_mm']/2],plane_normal=[0,0,1])
                loops = sorted(section.discrete, key=lambda a: np.ptp(a[:,:2],axis=0).prod())
                assert len(loops) == 2
                pocket = loops[0]
                center_x = v['chamber_center_x_mm'] + v['grip_center_offset_x_mm']
                np.testing.assert_allclose([pocket[:,0].min(),pocket[:,0].max()],
                                           [center_x-v['grip_width_mm']/2,center_x+v['grip_width_mm']/2],atol=.02)
                np.testing.assert_allclose([pocket[:,1].min(),pocket[:,1].max()],
                    [v['cover_front_y_mm']+v['grip_front_offset_mm'],v['cover_front_y_mm']+v['grip_front_offset_mm']+v['grip_length_mm']],atol=.02)
            else:
                holder = mesh('holder_frame', part='holder_frame')
                simple = mesh('holder_simplified', part='holder_frame', simplified=True)
                assert holder.euler_number == 0 and simple.euler_number == 0
                np.testing.assert_allclose(holder.volume, simple.volume, rtol=1e-6)
                motor = mesh('wall_mount', part='side_unit')
                assert len(motor.split()) == 1, 'Motor must join the wall mount'
                edited = mesh('edited_holder', part='holder_frame', holder_hole_diameter_mm=50)
                assert edited.euler_number == 0 and edited.volume < holder.volume
        elif model == 'c16361':
            for simple in (False, True):
                floor = mesh(f'floor_{simple}', part='perforated_floor', simplified=simple)
                assert len(floor.split()) == 1 and floor.euler_number == -16, 'Expected nine through-holes'
                section = floor.section(plane_origin=floor.bounds.mean(0),plane_normal=[0,0,1])
                assert len(section.discrete) == 10, 'Nine holes and an outer perimeter'
            closed = mesh('lid_closed', part='sample_box_lid')
            opened = mesh('lid_open', part='sample_box_lid', pose='open')
            np.testing.assert_allclose(closed.volume, opened.volume, rtol=1e-5)
            assert opened.bounds[1,2] > closed.bounds[1,2]+100
        else:
            infill = mesh('left_infill', part='lower_left_infill')
            np.testing.assert_allclose(infill.bounds[:,2],
                                      [v['lower_floor_z_mm']-2,v['lower_floor_z_mm']+v['lower_left_infill_height_mm']],atol=.02)
            for label in ('lower','upper'):
                for component in ('shell','inner_lining'):
                    name = f'{label}_{component}'
                    closed = mesh(name+'_closed', part=name)
                    for fraction in (.25,.5,.75,1):
                        opened = mesh(name+f'_{fraction}', part=name, pose='custom', **{label+'_door_opening':fraction})
                        transform = trimesh.transformations.rotation_matrix(
                            -np.deg2rad(v['door_open_angle_deg']*fraction),[0,0,1],
                            [-v['hinge_axis_outboard_mm'],v['tower_rear_y_mm']-v['rear_strip_depth_mm'],0])
                        expected = closed.copy(); expected.apply_transform(transform)
                        np.testing.assert_allclose(opened.bounds, expected.bounds, atol=.02)
                        np.testing.assert_allclose(opened.volume, closed.volume, rtol=2e-5)
            floor = mesh('floor_recess', part='lower_recess_wall')
            simple = mesh('floor_recess_simple', part='lower_recess_wall', simplified=True)
            assert floor.euler_number == 0 and simple.euler_number == 0
            np.testing.assert_allclose(floor.volume, simple.volume, rtol=1e-6)
    return {'status': 'passed', 'authoring': 'native OpenSCAD', 'cases': len(cases),
            'checks': cases, 'native_source_sha256': source_hashes(folder),
            'limitations': ['Geometry and selected motion transforms checked; physical fit and continuous collision clearance are not certified.']}


def previews(model, folder, work, values):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    source = folder / 'assembly.scad'
    cameras = {'cary60': '600,-750,950,200,230,80', 'c16361': '700,-950,850,200,190,100',
               'c13472': '-650,-900,850,280,180,150', 'c13534': '500,-1600,850,200,220,250'}
    options = ['--imgsize=1100,850', '--autocenter', '--viewall', '--projection=o', '--colorscheme=Tomorrow']

    def render(name, camera, **definitions):
        path = work / (name+'.png')
        run_engine(source,path,definitions,[*options,'--camera='+camera])
        return Image.open(path).convert('RGB')

    board = Image.new('RGB',(2200,950),'white'); draw=ImageDraw.Draw(board)
    for i,pose in enumerate(('closed','open')):
        board.paste(render(pose,cameras[model],pose=pose),(i*1100,60))
        draw.text((i*1100+25,20),model.upper()+' / '+pose,font=font,fill='black')
    draw.text((25,920),'Native OpenSCAD render. Dimensions in mm. Internal dimensions and motion limits include estimates.',font=small,fill='black')
    board.save(folder.parent/'assembly.png')
    # Orthographic review, dimensions taken from evaluated native parameters.
    h=values.get('body_height_mm',values.get('overall_height_mm'))
    w=values['body_width_mm']; d=values['body_depth_mm']
    views=[('Front',f'{w/2},-1500,{h/2},{w/2},{d/2},{h/2}',f'Width {w:g}; height {h:g}'),
           ('Top',f'{w/2},{d/2},1800,{w/2},{d/2},0',f'Width {w:g}; depth {d:g}'),
           ('Right',f'2000,{d/2},{h/2},{w/2},{d/2},{h/2}',f'Depth {d:g}; height {h:g}'),
           ('Open detail',cameras[model],'Functional interiors and moving assemblies')]
    sheet=Image.new('RGB',(2200,1900),'white');draw=ImageDraw.Draw(sheet)
    for i,(label,camera,caption) in enumerate(views):
        x=i%2*1100;y=i//2*950
        sheet.paste(render('dimension_'+str(i),camera,pose='open' if i==3 else 'closed'),(x,y+60))
        draw.text((x+25,y+15),model.upper()+' / '+label,font=font,fill='black')
        draw.text((x+25,y+915),caption,font=small,fill='black')
    sheet.save(folder.parent/'overview.png')
    # Animation frames are diagnostic only; no extra preview folders in delivery.
    for phase in (.125,.25,.375):
        render('motion_'+str(phase),cameras[model],animate_motion=True,**{'$t':phase})


def build(models=MODELS, from_cad_parameters=False, check_cad=False, keep_validation=False, extras=False):
    if from_cad_parameters:
        raise ValueError('Native SCAD is now authoritative. Copy reviewed edits to src/instruments/native/<model>/ and run build; no Python parameter import is needed.')
    from .publish import publish, overview
    for model in models:
        machine=ROOT/'output/objects/hkqai-instruments'/model/'codex'
        machine.mkdir(parents=True,exist_ok=True)
        work=machine/'intermediate/.native_build'
        if work.exists():
            raise RuntimeError(f'Previous build diagnostics remain at {work}; inspect and move them before retrying.')
        stage=work/'package';stage.mkdir(parents=True)
        checks=work/'checks';checks.mkdir()
        folder=stage/'openscad';stage_source(model,folder)
        values=read_parameters(folder,checks)
        print(model,'checking native SCAD',flush=True)
        report=check_geometry(model,folder,checks,values,extended=check_cad)
        print(model,'rendering native previews',flush=True)
        previews(model,folder,checks,values)
        (stage/'reports').mkdir()
        (stage/'reports/cad_summary.json').write_text(json.dumps(report,indent=2)+'\n')
        defaults=json.loads((SOURCE_ROOT/'src/instruments/inputs'/model/'design_defaults.json').read_text())
        # Preserve evidence references, but retire redundant legacy geometry snapshots.
        previous=machine/'products/parameters.json'
        old=json.loads(previous.read_text()) if previous.exists() else {}
        metadata={key:old[key] for key in ('new_photo_review','body_dimension_evidence','joint_evidence') if key in old}
        metadata.update(model=model,units='mm and degrees',frame='X right, Y rear, Z up; support Z=0',
                        authoring='src/instruments/native/'+model+'/*.scad; this JSON is an evaluated export snapshot',
                        design_parameters=values,design_parameter_evidence=parameter_evidence(folder),
                        limitations=defaults.get('metadata',{}).get('limitations',[])+report['limitations'],
                        native_source_sha256=source_hashes(folder))
        (stage/'parameters.json').write_text(json.dumps(metadata,indent=2)+'\n')
        publish(model,source=stage)
        from ..assets import save_manifest
        native_files = {name: NATIVE/'common.scad' if name == 'common.scad' else NATIVE/model/name
                        for name in FILES}
        save_manifest(machine/'asset.json', kind='openscad', sources=list(native_files.values()),
                      variants={model: {}}, units='mm', frame='X right, Y rear, Z up; support Z=0',
                      native_files=native_files,
                      rebuild=['python', '-m', 'src.instruments', 'build', '--model', model])
        if keep_validation or extras:
            latest=machine/'intermediate/validation/native_latest'
            if latest.exists():shutil.rmtree(latest)
            latest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(checks,latest)
        shutil.rmtree(work)
        print(model,f'complete: {report["cases"]} native checks',flush=True)
    overview()


def print_part(model, part, pose='closed', keep_validation=False):
    """Export the current authored source, including topology edits, in mm."""
    with tempfile.TemporaryDirectory(prefix='native_print_') as temp:
        work=Path(temp);folder=work/'openscad';stage_source(model,folder)
        mesh=export_mesh(folder/'assembly.scad',work,'part',part=part,pose=pose)
        if len(mesh.split())!=1:
            raise ValueError('Choose one manufactured solid; this selection has multiple disconnected components.')
        dest=ROOT/'output/objects/hkqai-instruments'/model/'codex/intermediate/printing'
        dest.mkdir(parents=True,exist_ok=True)
        # Selection may be a user-defined string, never let it escape the destination.
        safe=re.sub(r'[^a-zA-Z0-9_-]','_',part)
        target=dest/f'{safe}_{pose}.stl';shutil.copy2(work/'part.stl',target)
        target.with_suffix('.json').write_text(json.dumps({'model':model,'part':part,'pose':pose,
            'units':'millimetres','watertight':True,'positive_volume':True,'native_source_sha256':source_hashes(folder),
            'bounds_mm':mesh.bounds.tolist()},indent=2)+'\n')
        print(target,flush=True)
        return target
