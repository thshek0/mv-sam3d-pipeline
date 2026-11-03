"""Load optional simulation validation dependency, with an explicit environment override."""
import importlib,os,sys
from pathlib import Path

def pybullet():
 try:return importlib.import_module('pybullet')
 except ImportError:
  location=Path(os.environ.get('INSTRUMENT_PYBULLET_PATH','/tmp/c13534_urdf_validation'))
  if location.is_dir():sys.path.insert(0,str(location))
  try:return importlib.import_module('pybullet')
  except ImportError as exc:raise RuntimeError('PyBullet is required for simulation validation. Install it or set INSTRUMENT_PYBULLET_PATH.') from exc
