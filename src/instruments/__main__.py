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
        ('validate', 'Check hashes, references and simulator loading'),
        ('cad', 'Export native parametric CSG'),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument('--model', choices=MODELS, help='Default: all four instruments')
        sub.add_argument('--keep-validation', action='store_true',
                         help='Keep diagnostics in <machine>/validation/latest; failures are always retained')
        if name in ('build', 'publish', 'cad'):
            sub.add_argument('--extras', action='store_true',
                             help='Keep extra states, individual views, motion and collision reviews')
        if name == 'build':
            sub.add_argument('--from-cad-parameters', action='store_true',
                             help='Import numeric edits from parameters_native.scad')
            sub.add_argument('--check-cad', action='store_true', help='Run native CAD consistency checks')
        if name == 'cad':
            sub.add_argument('--check', action='store_true')
    printing = commands.add_parser('stl', help='Export one native CAD part for printing, in mm')
    printing.add_argument('--model', choices=MODELS, required=True)
    printing.add_argument('--part', required=True, help='Semantic part name from reports/parts.json or the recipe')
    printing.add_argument('--pose', choices=('closed', 'open'), default='closed')
    printing.add_argument('--keep-validation', action='store_true')
    share = commands.add_parser('share', help='Create a sharing ZIP on demand')
    share.add_argument('--project', action='store_true', help='Include source and datasets for rebuilding')
    args = parser.parse_args()
    models = [args.model] if getattr(args, 'model', None) else MODELS
    if args.command == 'build':
        from .builder import build
        build(models, from_cad_parameters=args.from_cad_parameters,
              check_cad=args.check_cad, keep_validation=args.keep_validation, extras=args.extras)
    elif args.command == 'stl':
        from .printing import export
        export(args.model, args.part, pose=args.pose, keep_validation=args.keep_validation)
    elif args.command == 'validate':
        from .validation import validate
        print(json.dumps(validate(models, keep_validation=args.keep_validation), indent=2))
    elif args.command in ('publish', 'cad'):
        from .publish import publish, overview
        from .storage.diagnostics import session
        from .builder import current_model_dir
        for model in models:
            current = current_model_dir(model)
            with session(current.parent, keep=args.keep_validation):
                if args.command == 'publish':
                    publish(model, extras=args.extras)
                else:
                    _cad(current, model, args)
        overview()
    else:
        from .sharing import main as share, project
        project() if args.project else share()


def _cad(current, model, args):
    from pathlib import Path
    import shutil
    from .storage.diagnostics import workspace
    from .cad import export, check
    from .publish import publish
    from .design import build
    from .geometry import open_value
    with workspace('cad_export') as tmp:
        work = Path(tmp)
        shutil.copytree(current, work, dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.zip'))
        if args.check:
            # Compact packages omit review GLBs; regenerate from the same recipe.
            m = build(json.loads((work / 'parameters.json').read_text()))
            states = m.demo_states if model == 'c13534' else {'closed': 0, 'open': open_value(m.p)}
            for state, value in states.items():
                m.scene(value).export(work / 'meshes' / f'collision_{state}.glb')
        dest = export(work)
        if args.check:
            print(json.dumps(check(dest), indent=2), flush=True)
        publish(model, source=work, extras=args.extras)


if __name__ == '__main__':
    main()
