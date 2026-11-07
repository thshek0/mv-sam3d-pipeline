# Project specification

Scope: the whole mv-sam3d-pipeline repository—capture/import, masks, depth/cameras,
reconstruction, authored CAD/products, simulation, exports, layout, tooling and
documentation. **This file is the single source of truth for project requirements.**
Keep rules concise, define each once, and reference them elsewhere. Code and
supporting documents must conform; existing work is not automatically certified.

## Development contract

**Specify → implement → verify → record.** This applies to every project change,
not just particular objects. [AGENTS.md](AGENTS.md) makes it a standing instruction
for agents. Explicit user instructions take precedence; record their effect on
the specification rather than silently retaining contradictory requirements.

Before implementation, update the affected requirement here or its existing
domain document. Small changes need only a short entry under the relevant section:

```text
Change / requirement IDs:
Intended behavior and scope:
Authoritative inputs, dependencies, units and outputs:
Acceptance checks and tolerances:
Assumptions / unresolved questions:
After implementation: checks actually run, results and remaining limitations.
```

Use existing documents and directories. Do not create a spec per task, a new
framework, or duplicate parameter data already held in an authoritative source.
An object-specific document can supplement this specification; it cannot replace
the project-wide process. A user-directed routine change does not need a separate
approval ceremony. Resolve only ambiguities that prevent safe, correct progress.

If behavior or scope changes during implementation, update the specification
before continuing the dependent work. Record meaningful decisions, failures,
changed tolerances, commands and results in [EXPERIMENTS.md](EXPERIMENTS.md), with
the relevant requirement IDs. Keep detailed generated evidence in the existing
object `intermediate/` or project `output/maintenance/` area. Label retrospective
documentation as retrospective; do not invent a prior spec or verification run.

## Project requirements

| ID | Required behavior | Acceptance / evidence |
| --- | --- | --- |
| P01 | Every change follows the development contract above, including scripts, tests, dependencies and documentation. | A requirement and acceptance check exist before implementation; completion records actual results and unresolved requirements. For prose-only edits, check consistency and links rather than inventing tests. |
| P02 | Preserve original captures, annotations, calibration and accepted source geometry. Keep distinct captures and their poses separate. Use canonical identities and compatibility aliases; never silently substitute a different dataset. | Input/provenance inventory, dependency checks before retirement, and scene/identity tests. Historical relocation records remain traceable. |
| P03 | Reconstruction honors the selected depth/camera and mesh methods. Reject unavailable or incompatible inputs explicitly. TSDF requires real poses; identity placeholders for unposed RealSense/hybrid input are insufficient. | Method, pipeline, depth-resolution and TSDF tests; applicable capture metadata checks. Distinguish inferred camera parameters from calibrated ones. |
| P04 | Use the existing layout below. No application-named output roots, extra `model/` layer, or alternate canonical object directories. Current products, intermediates and requested deliveries have distinct purposes. | Path/identity tests and an inventory of affected outputs. Relocations preserve hashes and update active references. |
| P05 | Units, scale, coordinate frames and transformations are explicit for each asset/workflow. Convert only deliberately and verify the conversion. Unknown scale remains unknown. | Registry/source metadata and numeric dimension/transform checks. CAD millimetres and simulation metres may coexist; there is no global unit rule. Bird print scale remains user-selected in the slicer. |
| P06 | Build from authoritative editable inputs: native CAD, procedural recipes plus parameters, or accepted meshes plus provenance. Generated parameter snapshots are not authoring inputs. | Rebuild source/dependency inventory and hashes. Native instruments use their SCAD sources. Codex assets cannot depend on disposable SAM3D/TSDF/DualBRep output directories; promoted accepted geometry retains its provenance. |
| P07 | Keep appearance, CAD surfaces, collision geometry and physics semantics distinct. Do not imply that mesh-to-STEP conversion recovers native CAD. Do not fill object cavities by merging collision pieces into one hull. Label estimated mass, inertia and friction. | Format-specific geometry/material/collision checks; declared conversion losses; no invented physical measurements or dynamic properties. |
| P08 | Produce only requested delivery formats from registered sources. Preserve existing recipient snapshots; reject stale source hashes. Collection deliveries must be self-contained for use and include the previews/instructions required by the delivery policy. | Export/packaging tests, relocated-package dependency checks and relevant visual inspection. Do not automatically generate every format or duplicate a current asset merely for another application. |
| P09 | A reproducibility claim identifies the source recipe, all required data/weights, versions, command, output and comparison criterion. State whether equality means identical bytes, geometry within tolerance, or statistical agreement. | An actual isolated rebuild and comparison for the claimed scope. Record seeds/settings where relevant; stochastic inference and timestamped exports need explicit criteria. A Git checkout alone may omit ignored datasets, weights and generated provenance. |
| P10 | Verification is tied to requirements and proportionate to the change. Test changed behavior and relevant failures; check actual exported artifacts when their correctness matters. Failed checks require investigation, not silent removal or relaxed tolerances. | Commands, results and artifact references in the change record. Distinguish source checks, export checks, simulation checks, target-application checks and physical tests; one does not prove the others. |
| P11 | Keep commands and dependency setup documented. Preserve public CLI/alias behavior unless the specified change intentionally revises it. Tests must not accidentally start real inference or require network/GPU work. | Relevant CLI regression tests, setup references and isolated test configuration. Explicit integration runs are recorded separately from offline tests. |

## Storage contract (P04)

```text
src/                                      implementation and editable recipes
dataset/<family>/<object-or-capture>/      original evidence and accepted inputs
tests/                                    regression and contract checks
env/                                      dependency/install records
docs/                                     existing detailed domain guidance
output/objects/<family>/<object>/<method>/
  products/                               current canonical assets
  intermediate/                           working data, history and diagnostics
  exports/<delivery-name>/                explicitly requested object snapshots
output/packages/                          collection handoffs and ZIPs
output/comparisons/                       cross-object/method reviews
output/maintenance/                       project audits and relocation records
output/index.html                         overview
```

Only `objects/`, `packages/`, `comparisons/`, `maintenance/` and `index.html` belong
at the output root. Methods are `codex`, `sam3d`, `tsdf`, and retained `dualbrep`
experiments. Existing capture-group arrangements and `examples/gripper/` are
documented exceptions, not a reason to relocate original inputs. Model names and
aliases are defined in [src/output_layout.py](src/output_layout.py). Existing
explicit domain subpaths remain governed by their domain documentation.

For Git sharing, retain the complete README gripper example. Include code, tests,
specifications and small required rebuild inputs; document separately supplied
large data. Keep local environments, tool state and caches ignored; caches may
stay on disk. Remove large redundant temporary files only after checking retained
copies and dependencies, and log their paths and hashes. Shared commands must be
portable; machine-specific installation notes belong in ignored `LOCAL.md`.
Track the existing instrument evidence JSON and procedural plate pattern JSON
in place; raw files remain ignored. Offline tests must pass without ignored data.

## Detailed specifications and existing checks

Referenced documents and machine-readable inputs own their specific details;
they must not duplicate or override this specification. Resolve conflicts here
first, following explicit user instructions. Keep measurements, nominal values
and estimates distinct, and retain conflicting evidence. Listed tests indicate
available checks, not complete coverage or results for the current change.

| Area | Detail / authoritative configuration | Relevant existing checks |
| --- | --- | --- |
| Capture, masks, depth and reconstruction | [README](README.md), [SETUP](SETUP.md), method CLI/input metadata | [pipeline](tests/test_pipeline.py), [methods](tests/test_methods.py), [scene validation](tests/test_validate_scene.py), [TSDF](tests/test_tsdf.py), [unposed depth](tests/test_unposed_rs.py), converter tests under `tests/` |
| Dataset identities and storage | [README layout](README.md#data-format), [layout implementation](src/output_layout.py) | [output layout](tests/test_output_layout.py), [dataset families](tests/test_dataset_families.py) |
| Native instruments | [authoring contract](docs/instruments/prompt.md), [instrument guide](docs/instruments/README.md), `src/instruments/native/` and dataset evidence | [native authoring](tests/instruments/test_native_authoring.py), [motion](tests/instruments/test_native_motion.py), other `tests/instruments/` checks |
| Procedural products, plates and spoons | [README](README.md), editable recipes/JSON in `src/products.py`, `src/product_inputs/`, `src/plates/`, `src/spoons/`; [existing plate record](src/plates/SPEC.md) is supplemental | [products](tests/test_products.py), registered per-model build/export validation |
| Simulation, asset exports and handoffs | [delivery contract](docs/asset-delivery-rules.md), [delivery selection](src/delivery_profiles.json), asset registries | [assets](tests/test_assets.py), [delivery](tests/test_delivery.py), [mesh export](tests/test_mesh_export.py), [MuJoCo](tests/test_products_mujoco.py) |
| Environments, CLI and benchmarking | [SETUP](SETUP.md), `env/`, optional Git-ignored `LOCAL.md` | [dataset CLI](tests/test_run_dataset.py), [timing](tests/test_time_run.py), relevant component tests |

Run the affected offline checks with `.venv/bin/python -m pytest <test-paths> -q`.
Use [SETUP.md](SETUP.md#9-tests) for interpreter-isolation guidance. Native CAD,
GPU inference, export previews and physical checks require their own documented
commands and dependencies; do not report them as covered by offline unit tests.

## Adoption and completion

This is the single project entry point. `README.md` explains use, `SETUP.md`
explains installation, domain documents retain specifics, `LOCAL.md` holds local
environment notes, and `EXPERIMENTS.md` records development evidence. No extra
specification directory or automatic approval system is introduced.

Legacy work is not retroactively certified. When changing an area, first record
its intended behavior and missing acceptance criteria, then implement and verify
that scope. Completion requires linked evidence for the applicable requirements,
updated active documentation, and an explicit statement of unverified limits.
This is a mandatory development/review workflow; a CI check cannot by itself
prove that a prose specification is complete or that implementation began after it.

2026-11-07 adoption: P01–P11 consolidate existing project rules and the user's
project-wide spec-driven requirement. This change updates documentation and agent
instructions only; it does not assert a full repository rebuild or physical audit.
