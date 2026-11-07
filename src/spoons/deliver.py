"""Refresh requested MuJoCo spoon deliveries and ZIP after rebuilding.

python -m src.spoons.deliver
"""
from pathlib import Path
import json
import tempfile
import zipfile
import numpy as np
import mujoco
from src.assets import export_asset
from src.export import sha256
from src.mujoco_check import check
from .build import ROOT, json_write

NAMES=('spoon_daiso_pink','spoon_home_black')


def deliver():
    reports={}
    for name in NAMES:
        root=ROOT/'output/objects/hestia-rl'/name/'codex';products=root/'products';dest=root/'exports/mujoco_rl'
        dest.parent.mkdir(parents=True,exist_ok=True)
        p=json.loads((products/'parameters.json').read_text())
        with tempfile.TemporaryDirectory(prefix='.spoon_delivery_',dir=dest.parent) as temp:
            stage=Path(temp)/'current';export_asset(root/'asset.json',stage,['mujoco'])
            model=stage/'mujoco/collision/model.xml'
            report=check(model,ball_xy=(0,0) if 'pink' in name else (0,p['support_center_y_mm']/1000),cavity_rim=.038 if 'pink' in name else .06)
            scene=model.parent/'scene.xml'
            scene.write_text(f'<mujoco model="{name}_scene"><include file="model.xml"/><option timestep="0.002" gravity="0 0 -9.81"/><worldbody><light pos="0 -0.2 0.6"/><geom name="floor" type="plane" size="1 1 .1" pos="0 0 -.002" rgba=".8 .8 .8 1"/><camera name="overview" pos=".35 -.4 .35" xyaxes=".75 .66 0 -.35 .4 .85"/></worldbody></mujoco>\n')
            m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
            for _ in range(1000):
                mujoco.mj_step(m,d)
                assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
            report['delivered_scene_compiled_and_stepped']=True;reports[name]=report
            json_write(stage/'mujoco/runtime_validation.json',report)
            collision_report=model.parent/'validation.json';v=json.loads(collision_report.read_text())
            v.update(mujoco_runtime_tested=True,runtime_validation='../runtime_validation.json');json_write(collision_report,v)
            (model.parent/'README.txt').write_text(f'{name}: MuJoCo RL asset\nLoad scene.xml for a floor scene, or model.xml to integrate the object.\nMetres; visual contacts disabled (group 2). Separate convex contact hulls (group 3).\nKeep every hull separate to preserve the open cavity.\nContact priority 1, margin 0.5 mm, solref 0.004 1, solimp 0.99 0.999 0.0001; tested at timestep 0.002 s.\nMuJoCo {mujoco.__version__}: compilation, 1000 steps and a 1500-step ball-in-cavity test passed.\nNo floor was present in the cavity test; final spoon contact is required.\nDimensions, material thickness, mass and friction are estimates.\nKeep all relative OBJ/MTL/PNG paths intact. See ../runtime_validation.json.\nNo SAM3D software or experimental folder is required. Black rebuilds from saved accepted mesh sources with explicit provenance.\n')
            manifest=stage/'manifest.json';v=json.loads(manifest.read_text())
            v['checks']['mujoco'].update(mujoco_runtime_tested=True,runtime_validation='mujoco/runtime_validation.json');v['runtime_validation']=report
            v['limitations']=[x for x in v['limitations'] if 'offline checks only' not in x]
            v['limitations'].append('Compile, dynamic and cavity smoke tests passed; RL task success and calibrated physical dynamics are not established.')
            v['files']={str(f.relative_to(stage)):sha256(f) for f in sorted(stage.rglob('*')) if f.is_file() and f!=manifest};json_write(manifest,v)
            if dest.exists():dest.rename(Path(temp)/'previous')
            stage.rename(dest)
        json_write(products/'mujoco_validation.json',report)
        v=json.loads((products/'validation.json').read_text());v['mujoco_runtime_tested']=True;json_write(products/'validation.json',v)
        print(name,json.dumps(report),flush=True)
    family=ROOT/'output/comparisons/hestia_rl_collision'
    for filename in ('inventory.json','runtime_validation.json'):
        path=family/filename
        if not path.exists():continue
        entries=json.loads(path.read_text())
        for entry in entries:
            name=entry['object']
            if name not in reports:continue
            if filename=='runtime_validation.json':entry.update(reports[name])
            else:
                v=json.loads((ROOT/'output/objects/hestia-rl'/name/'codex/products/validation.json').read_text())
                entry.update(parts=v['collision_parts'],triangles=v['collision_triangles'],action='Current accepted spoon and validated convex collision assembly')
        json_write(path,entries)
    package=ROOT/'output/packages/spoons_mujoco_rl.zip';temporary=package.with_suffix('.tmp.zip');hashes={}
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('README.txt','Two authored spoon assets in metres. Load <object>/mujoco/collision/scene.xml or integrate model.xml. Keep hulls separate. Dynamic and cavity tests passed. No SAM3D experimental folder or inference is required. Black uses permanent accepted mesh sources. Mass, dimensions and friction remain estimates.\n')
        for name in NAMES:
            root=ROOT/'output/objects/hestia-rl'/name/'codex'
            files=[(f,f'{name}/{f.relative_to(root/"exports/mujoco_rl")}') for f in sorted((root/'exports/mujoco_rl').rglob('*')) if f.is_file()]
            for f in (root/'products').iterdir():
                if f.suffix in ('.glb','.png','.json','.txt'):files.append((f,f'{name}/products/{f.name}'))
            for source,destination in files:z.write(source,destination);hashes[destination]=sha256(source)
        z.writestr('manifest.json',json.dumps({'units':'m','files':hashes,'mujoco_version':mujoco.__version__,'cavity_tests_passed':True},indent=2)+'\n')
    with zipfile.ZipFile(temporary) as z:assert z.testzip() is None
    temporary.replace(package)
    print('Package:',package,flush=True)
    return reports


if __name__=='__main__':deliver()
