"""Retain requested or failed validation work; clean successful scratch work."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import tempfile
import traceback

_active = ContextVar('instrument_validation', default=None)


@contextmanager
def session(machine_root, keep=False):
    """One diagnostic session per machine; nested checks share its lifecycle."""
    if _active.get() is not None:
        yield _active.get()
        return
    machine_root = Path(machine_root)
    machine_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.validation_', dir=machine_root))
    token = _active.set(work)
    failure = None
    try:
        yield work
    except BaseException:
        failure = traceback.format_exc()
        (work / 'failure.txt').write_text(failure)
        raise
    finally:
        _active.reset(token)
        if keep or failure:
            stamp = datetime.now(timezone.utc).isoformat()
            (work / 'session.json').write_text(json.dumps({
                'created_utc': stamp, 'status': 'failed' if failure else 'passed',
                'scope': 'Only checks invoked in this session; see reports for coverage.',
            }, indent=2) + '\n')
            dest = machine_root / 'validation' / ('failures' if failure else '')
            dest.mkdir(parents=True, exist_ok=True)
            dest = dest / (work.name.removeprefix('.') if failure else 'latest')
            if dest.exists():
                shutil.rmtree(dest)
            work.rename(dest)
            print(f'Validation artifacts: {dest}', flush=True)
        else:
            shutil.rmtree(work)


@contextmanager
def workspace(label):
    root = _active.get()
    if root is None:
        raise RuntimeError('Validation workspace requires a diagnostics.session')
    folder = root / 'workspaces'
    folder.mkdir(exist_ok=True)
    yield tempfile.mkdtemp(prefix=label + '_', dir=folder)


def capture(path, relative):
    """Copy a diagnostic before publication prunes it from the user package."""
    root = _active.get()
    if root is None or not Path(path).exists():
        return
    dest = root / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)


def command_log(path, command, result):
    Path(path).with_suffix('.log.json').write_text(json.dumps({
        'command': command, 'returncode': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr,
    }, indent=2) + '\n')
