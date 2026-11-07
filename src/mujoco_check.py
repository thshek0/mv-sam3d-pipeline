"""Compile MJCF, exercise contact, and optionally check a hollow bowl with a ball.

python -m src.mujoco_check model.xml --ball-x 0 --ball-y 0 --cavity-rim 0.038
Tests modify an in-memory copy only; original object/XML remain unchanged.
"""
from pathlib import Path
import argparse,json,time,xml.etree.ElementTree as ET
import numpy as np
import mujoco


def check(path,*,ball_xy=None,cavity_rim=None):
    path=Path(path).resolve();root=ET.parse(path).getroot()
    for asset in root.findall('asset/*'):
        if asset.get('file'):asset.set('file',str((path.parent/asset.get('file')).resolve()))
    # Test dynamic assembly on a floor without changing the delivered model.
    option=root.find('option')
    if option is None:option=ET.SubElement(root,'option')
    option.set('timestep','0.002');option.set('gravity','0 0 -9.81')
    world=root.find('worldbody');ET.SubElement(world,'geom',name='test_floor',type='plane',size='2 2 .1',pos='0 0 -.002')
    start=time.perf_counter();m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));compile_time=time.perf_counter()-start
    d=mujoco.MjData(m);max_contacts=0;start=time.perf_counter()
    for i in range(1000):
        mujoco.mj_step(m,d);max_contacts=max(max_contacts,d.ncon)
        if not(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):raise ValueError('Non-finite dynamic state')
    report={'mujoco_version':mujoco.__version__,'compiled':True,'mesh_assets':m.nmesh,'geoms':m.ngeom,'collision_geoms':int(((m.geom_contype!=0)|(m.geom_conaffinity!=0)).sum())-1,'compile_seconds':compile_time,'steps':1000,'simulation_seconds':2,'step_wall_seconds':time.perf_counter()-start,'max_contacts':max_contacts,'finite_state':True,'mass_friction_calibrated':False}
    if ball_xy is not None:
        # Freeze the spoon/bowl so the probe diagnoses its cavity, not its motion.
        for body in world.findall('body'):
            for joint in list(body):
                if joint.tag=='freejoint' or (joint.tag=='joint' and joint.get('type')=='free'):body.remove(joint)
        for geom in list(world.findall('geom')):
            if geom.get('name')=='test_floor':world.remove(geom)
        z=float(cavity_rim)+.025;radius=.004
        probe=ET.SubElement(world,'body',name='cavity_test_ball',pos=f'{ball_xy[0]} {ball_xy[1]} {z}')
        ET.SubElement(probe,'freejoint');ET.SubElement(probe,'geom',name='probe_sphere',type='sphere',size=str(radius),mass='.003',friction='.5 .005 .0001')
        mb=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));db=mujoco.MjData(mb)
        bid=mujoco.mj_name2id(mb,mujoco.mjtObj.mjOBJ_BODY,'cavity_test_ball');gid=mujoco.mj_name2id(mb,mujoco.mjtObj.mjOBJ_GEOM,'probe_sphere')
        contact=False
        for i in range(1500):
            mujoco.mj_step(mb,db)
            if i>500:
                for c in db.contact:
                    if gid in c.geom and all(mujoco.mj_id2name(mb,mujoco.mjtObj.mjOBJ_GEOM,int(g))!='test_floor' for g in c.geom):contact=True
        final=db.xpos[bid].copy();contact=any(gid in c.geom for c in db.contact)
        speed=float(np.linalg.norm(db.qvel[:3]));passed=bool(contact and speed<.05 and final[2]+radius<float(cavity_rim)-.002 and final[2]>0 and np.linalg.norm(final[:2]-ball_xy)<.045)
        report['cavity_ball_test']={'passed':passed,'radius_m':radius,'initial_xyz_m':[ball_xy[0],ball_xy[1],z],'final_xyz_m':final.tolist(),'rim_height_m':float(cavity_rim),'final_object_contact':contact,'final_speed_m_s':speed,'no_floor_in_cavity_test':True,'steps':1500}
        if not passed:raise ValueError('Ball did not settle inside open cavity: '+str(report['cavity_ball_test']))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);p.add_argument('--ball-x',type=float);p.add_argument('--ball-y',type=float);p.add_argument('--cavity-rim',type=float);p.add_argument('--report',type=Path)
    a=p.parse_args();r=check(a.model,ball_xy=None if a.ball_x is None else (a.ball_x,a.ball_y),cavity_rim=a.cavity_rim)
    if a.report:a.report.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
