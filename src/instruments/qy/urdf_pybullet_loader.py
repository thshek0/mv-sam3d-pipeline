"""Load the portable C13534 package with fixed base and mounting filters.

Requires pybullet. From this folder: python load_pybullet.py --pose open
Import load_model/set_pose from an application to use an existing connection.
The joints are independent: this helper does not implement door latch logic.
"""
from pathlib import Path
import argparse
import json

PACKAGE = Path(__file__).resolve().parent


def load_model(p, client_id=0, base_position=(0, 0, 0), base_orientation=(0, 0, 0, 1)):
    flags = (p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT |
             p.URDF_USE_INERTIA_FROM_FILE | p.URDF_INITIALIZE_SAT_FEATURES)
    # Convex polyhedral features avoid padded GJK overlap reports at the exact
    # closed seam. Without this flag, PyBullet 3.2.7 reported ~2 mm overlap.
    # Do not use URDF_MERGE_FIXED_LINKS: narrow mounting filters use these links.
    body = p.loadURDF(str(PACKAGE/'robot.urdf'),basePosition=base_position,
        baseOrientation=base_orientation,useFixedBase=True,flags=flags,physicsClientId=client_id)
    links={'base_link':-1};joints={}
    for index in range(p.getNumJoints(body,physicsClientId=client_id)):
        info=p.getJointInfo(body,index,physicsClientId=client_id)
        links[info[12].decode()]=index
        if info[2]!=p.JOINT_FIXED:
            joints[info[1].decode()]=index
            p.setJointMotorControl2(body,index,p.VELOCITY_CONTROL,force=0,physicsClientId=client_id)
    settings=json.loads((PACKAGE/'export_settings.json').read_text())
    for index in links.values():
        p.changeDynamics(body,index,collisionMargin=settings['pybullet_collision_margin_m'],physicsClientId=client_id)
    filters=json.loads((PACKAGE/'collision_filters.json').read_text())
    disabled=set()
    for name in filters['self_excluded_hardware_links']:
        for other in links:
            if name!=other:disabled.add(tuple(sorted([name,other])))
    for pair in filters['explicit_disabled_pairs']:
        disabled.add(tuple(sorted(pair['links'])))
    for a,b in disabled:
        p.setCollisionFilterPair(body,body,links[a],links[b],0,physicsClientId=client_id)
    return body,links,joints


def set_pose(p, body, joints, name='closed', client_id=0):
    pose=json.loads((PACKAGE/'joint_states.json').read_text())['poses'][name]
    for joint,index in joints.items():
        p.resetJointState(body,index,pose.get(joint,0.),targetVelocity=0.,physicsClientId=client_id)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose',default='closed',choices=['closed','handles_released','upper_open','lower_open','open'])
    parser.add_argument('--gui',action='store_true')
    args=parser.parse_args()
    import pybullet as p
    client=p.connect(p.GUI if args.gui else p.DIRECT)
    body,links,joints=load_model(p,client)
    set_pose(p,body,joints,args.pose,client)
    p.setGravity(0,0,0,physicsClientId=client)
    print(f'Loaded {len(links)} links / {len(joints)} moving joints, fixed base; pose={args.pose}')
    if args.gui:
        import time
        p.resetDebugVisualizerCamera(1.4,40,-25,[.25,.18,.27],physicsClientId=client)
        try:
            while p.isConnected(client):
                p.stepSimulation(physicsClientId=client);time.sleep(1/240)
        except KeyboardInterrupt:
            pass
    if p.isConnected(client):p.disconnect(client)


if __name__=='__main__':main()
