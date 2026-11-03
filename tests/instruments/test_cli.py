"""The consolidated CLI remains discoverable without loading the render stack."""
from pathlib import Path
import subprocess
import sys


def test_cli_help_from_repository_root():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, '-m', 'src.instruments', '--help'],
        cwd=root, text=True, capture_output=True, check=True,
    )
    for command in ('build', 'publish', 'validate', 'share', 'cad'):
        assert command in result.stdout


def test_cad_help_exposes_all_instruments():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, '-m', 'src.instruments', 'cad', '--help'],
        cwd=root, text=True, capture_output=True, check=True,
    )
    for model in ('cary60', 'c16361', 'c13472', 'c13534'):
        assert model in result.stdout
    assert '--check' in result.stdout
