from pathlib import Path
import os

SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('INSTRUMENT_WORKSPACE', SOURCE_ROOT)).expanduser().resolve()
MODELS = ("cary60", "c16361", "c13472", "c13534")
