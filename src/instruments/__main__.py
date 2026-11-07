"""Command interface for the current instrument models."""
import argparse
import json
from .paths import MODELS


def main():
    parser = argparse.ArgumentParser(description='Build, review and share instrument models.')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in (
        ('build', 'Rebuild geometry, visuals and current package'),
        ('publish', 'Repackage existing assets without changing geometry'),
        ('validate', 'Check portable CAD, evidence hashes and references'),
        ('cad', 'Export native parametric CSG'),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument('--model', choices=MODELS, help='Default: all four instruments')
        sub.add_argument('--keep-validation', action='store_true',
                         help='Keep diagnostics in <machine>/codex/intermediate/validation/latest; failures are always retained')
        if name in ('build', 'publish', 'cad'):
            sub.add_argument('--extras', action='store_true',
                             help='Retain private native render/mesh diagnostics')
        if name == 'build':
            sub.add_argument('--from-cad-parameters', action='store_true',
                             help='Retired: edit src/instruments/native/<model>/ directly')
            sub.add_argument('--check-cad', action='store_true', help='Run native CAD consistency checks')
        if name == 'cad':
            sub.add_argument('--check', action='store_true')
    printing = commands.add_parser('stl', help='Export one native CAD part for printing, in mm')
    printing.add_argument('--model', choices=MODELS, required=True)
    printing.add_argument('--part', required=True, help='Component name from native assembly.scad')
    printing.add_argument('--pose', choices=('closed', 'open'), default='closed')
    printing.add_argument('--keep-validation', action='store_true')
    share = commands.add_parser('share', help='Create a small model ZIP and an image-only source-sheet ZIP')
    share.add_argument('--project', action='store_true', help='Include source and datasets for rebuilding')
    args = parser.parse_args()
    models = [args.model] if getattr(args, 'model', None) else MODELS
    if args.command in ('build', 'cad'):
        from .native_build import build
        build(models, from_cad_parameters=getattr(args, 'from_cad_parameters', False),
              check_cad=getattr(args, 'check_cad', False) or getattr(args, 'check', False),
              keep_validation=args.keep_validation, extras=args.extras)
    elif args.command == 'stl':
        from .native_build import print_part as export
        export(args.model, args.part, pose=args.pose, keep_validation=args.keep_validation)
    elif args.command == 'validate':
        from .validation import validate
        print(json.dumps(validate(models, keep_validation=args.keep_validation), indent=2))
    elif args.command == 'publish':
        from .publish import publish, overview
        from .storage.diagnostics import session
        from .builder import current_model_dir
        for model in models:
            current = current_model_dir(model)
            with session(current.parent / 'intermediate', keep=args.keep_validation):
                publish(model, extras=args.extras)
        overview()
    else:
        from .sharing import main as share, project
        project() if args.project else share()


if __name__ == '__main__':
    main()
