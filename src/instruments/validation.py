"""Verify a standalone copy of the current package without requiring a ZIP."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from .paths import ROOT, MODELS
from .physics import pybullet
from .storage.diagnostics import session, workspace


def validate(models=MODELS, keep_validation=False):
    pb = pybullet()
    reports = {}
    for model in models:
        root = ROOT / 'output/instruments' / model / 'model'
        with session(root.parent, keep=keep_validation) as diagnostics:
            with workspace('standalone_package') as temp:
                folder = Path(temp)
                shutil.copytree(root, folder, dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.zip'))
                manifest = json.loads((folder / 'manifest.json').read_text())
                for name, item in manifest['files'].items():
                    assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == item['sha256'], name
                robot = ET.parse(folder / 'urdf/robot.urdf').getroot()
                for mesh in robot.findall('.//mesh'):
                    assert (folder / 'urdf' / mesh.get('filename')).is_file(), mesh.attrib
                for pattern, directive in (('*.obj', 'mtllib '), ('*.mtl', 'map_Kd ')):
                    for path in folder.rglob(pattern):
                        for line in path.read_text().splitlines():
                            if line.startswith(directive):
                                assert (path.parent / line[len(directive):]).is_file(), path
                client = pb.connect(pb.DIRECT)
                try:
                    body = pb.loadURDF(str(folder / 'urdf/robot.urdf'), useFixedBase=True, physicsClientId=client)
                    assert body >= 0
                    for joint in range(pb.getNumJoints(body, physicsClientId=client)):
                        info = pb.getJointInfo(body, joint, physicsClientId=client)
                        if info[2] != pb.JOINT_FIXED:
                            for value in (info[8], info[9]):
                                pb.resetJointState(body, joint, value, physicsClientId=client)
                                pb.performCollisionDetection(physicsClientId=client)
                    for ref in re.findall(r'(?:href|src)="([^"]+)"', (folder / 'index.html').read_text()):
                        assert (folder / ref).exists(), ref
                finally:
                    pb.disconnect(client)
            reports[model] = dict(package_hashes=True, obj_material_images_resolve=True,
                                  standalone_pybullet_import=True, joint_endpoints_load=True,
                                  index_links_resolve=True,
                                  note='Loading checks do not verify physical contact accuracy or simultaneous motion.')
            (diagnostics / 'package_validation.json').write_text(json.dumps(reports[model], indent=2) + '\n')
        print(model, 'standalone package checked', flush=True)
    return reports
