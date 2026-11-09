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
object work area or project `output/maintenance/` area. Label retrospective
documentation as retrospective; do not invent a prior spec or verification run.

## Project requirements

| ID | Required behavior | Acceptance / evidence |
| --- | --- | --- |
| P01 | Every change follows the development contract above, including scripts, tests, dependencies and documentation. | A requirement and acceptance check exist before implementation; completion records actual results and unresolved requirements. For prose-only edits, check consistency and links rather than inventing tests. |
| P02 | Preserve required image information, annotations, calibration and accepted source geometry. Raw imports may be retired after verifying their accepted replacements; record any deliberate information loss. Keep distinct captures and poses separate. | Input/provenance inventory, replacement checks before retirement, and scene/identity tests. Historical relocation records remain traceable. |
| P03 | Reconstruction honors the selected depth/camera and mesh methods. Reject unavailable or incompatible inputs explicitly. TSDF requires real poses; identity placeholders for unposed RealSense/hybrid input are insufficient. | Method, pipeline, depth-resolution and TSDF tests; applicable capture metadata checks. Distinguish inferred camera parameters from calibrated ones. |
| P04 | Use the existing layout below. No application-named output roots, extra `model/` layer, or alternate canonical object directories. Current products, intermediates and requested deliveries have distinct purposes. | Path/identity tests and an inventory of affected outputs. Relocations preserve hashes and update active references. |
| P05 | Units, scale, coordinate frames and transformations are explicit for each asset/workflow. Convert only deliberately and verify the conversion. Unknown scale remains unknown. | Registry/source metadata and numeric dimension/transform checks. CAD millimetres and simulation metres may coexist; there is no global unit rule. Bird print scale remains user-selected in the slicer. |
| P06 | Build from authoritative editable inputs: native CAD, procedural recipes plus parameters, or accepted meshes plus provenance. Generated parameter snapshots are not authoring inputs. | Rebuild source/dependency inventory and hashes. Native instruments use their SCAD sources. Codex assets cannot depend on disposable SAM3D/TSDF/DualBRep output directories; promoted accepted geometry retains its provenance. |
| P07 | Keep appearance, CAD surfaces, collision geometry and physics semantics distinct. Do not imply that mesh-to-STEP conversion recovers native CAD. Do not fill object cavities by merging collision pieces into one hull. Label estimated mass, inertia and friction. | Format-specific geometry/material/collision checks; declared conversion losses; no invented physical measurements or dynamic properties. |
| P08 | Produce only requested delivery formats from registered sources. Preserve existing recipient snapshots; reject stale source hashes. Collection deliveries must be self-contained for use and include the previews/instructions required by the delivery policy. | Export/packaging tests, relocated-package dependency checks and relevant visual inspection. Do not automatically generate every format or duplicate a current asset merely for another application. |
| P09 | A reproducibility claim identifies the source recipe, all required data/weights, versions, command, output and comparison criterion. State whether equality means identical bytes, geometry within tolerance, or statistical agreement. | An actual isolated rebuild and comparison for the claimed scope. Record seeds/settings where relevant; stochastic inference and timestamped exports need explicit criteria. A Git checkout alone may omit ignored datasets, weights and generated provenance. |
| P10 | Verification is tied to requirements and proportionate to the change. Test changed behavior and relevant failures; check actual exported artifacts when their correctness matters. Failed checks require investigation, not silent removal or relaxed tolerances. | Commands, results and artifact references in the change record. Distinguish source checks, export checks, simulation checks, target-application checks and physical tests; one does not prove the others. |
| P11 | Keep commands and dependency setup documented. Preserve public CLI/alias behavior unless the specified change intentionally revises it. Tests must not accidentally start real inference or require network/GPU work. | Relevant CLI regression tests, setup references and isolated test configuration. Explicit integration runs are recorded separately from offline tests. |
| P12 | Agent methodology must travel with the repository. Accepted decisions, prompts, inputs and checks cannot depend on a chat, API key, personal skill or ignored local file. | Fresh-session instructions lead from AGENTS to this spec and the relevant domain contract. Verify a source-only checkout without API credentials; disclose missing data and untested agent/environment differences. |

## Portable agent workflow (P01, P09, P12)

`AGENTS.md` is the agent entry point; this specification owns requirements.
Task prompts supply the object/scope, evidence, requested outputs and acceptance
criteria, then reference the relevant domain contract. `LOCAL.md` may locate local
tools but cannot supply otherwise undocumented requirements or accepted decisions.
Persist decisions in the authoritative inputs and development log before ending
a session. A new session must inspect the current files and diff, not assume that
earlier chat context, generated assets or uncommitted changes exist in a clone.

No custom skill is required for this workflow. Optional personal skills do not
replace it. If a reusable project skill becomes necessary, track its source under
`.agents/skills/`, reference this spec and existing domain instructions, and avoid
duplicating their rules. Do not share credentials or personal Codex configuration.

Changing credentials does not preserve a model, settings, tools or permissions.
For agent comparisons, record the task, source revision/diff, model and available
settings, tool versions, inputs and acceptance results; never record the key.
Consistency means satisfying the same checks and preserving accepted sources;
identical new model-generated designs are not promised. Normal geometry builders
must work independently of the authoring agent's API credentials.

2026-11-09 review: check prompt/spec conflicts, skill discovery/ignore policy,
source-only tests without API credentials and contributor setup. Correct concise
instructions in place; record remaining public-release gaps without claiming a
fresh dependency installation or a cross-account agent evaluation.

2026-11-09 accepted follow-up (P06, P09–P12): reuse existing files only. Promote
the three older plates' parameters, dimension evidence and radial profiles into
`src/product_inputs/designs.json`; builders must no longer read editable inputs
from output folders. Verify identical visual geometry/appearance and valid collision
exports before removing superseded input copies. Test a source-only copy in a new
temporary Python environment installed from the existing dependency lists; record
native-tool dependencies and skips. Keep verification as one documented command,
with no CI file, extra skill, wrapper or folder redesign. Review, commit and push
the verified changes, excluding unrelated edits.
Compare old/new input readers under the same environment byte-for-byte. Retain
existing accepted GLBs: older exports use legacy material labels and UV rounding;
verify their vertices, faces, normals and texture pixels exactly, and report any
float32 UV difference separately rather than claiming identical export bytes.

## Storage contract (P04)

Use the same family grouping for inputs, current outputs and working files:

```text
dataset/<family>/<object>/      accepted inputs, evidence and unreplaced originals
output/<family>/<object>/       current authored model, preview/review and asset.json
output/<family>/<object>/<method>/  separate reconstruction results when needed
output/<family>/<object>/simulation/  separate metre assets when CAD uses millimetres
work/<family>/<object>/<method>/     diagnostics, experiments and preserved history
output/packages/                requested collection snapshots
output/comparisons/             cross-object reviews
output/maintenance/             migration and verification records
output/index.html               direct links to current assets
```

Keep `hestia-rl`, `hkqai-instruments`, `hkqai-trial` and fallback `misc` families.
Remove redundant `objects/`, `codex/` and `products/` layers from current outputs.
Authored products live directly under the object; SAM3D, TSDF and retained DualBRep
results keep method folders to avoid conflating different reconstructions. Work
keeps method names, including `codex`, for provenance. Preserve required instrument
states and trial capture groups. Existing dataset subdivisions (images, raw,
appearance, accepted geometry, online sources and captures) retain their meaning;
do not shuffle unique evidence merely to force one capture format. Editable
recipes remain in `src/`; the complete `examples/gripper/` stays in place.

2026-11-07 correction (P02, P04, P06, P08–P11): apply this across all existing
object outputs, restore HARMYNTA to its family in dataset/output/work, and update
builders, CLI routing, registries, deliveries and active documentation. Preserve
asset bytes and unique inputs; rebase registered references without blessing
previously stale sources. Verify relocation hashes, registered outputs, offline
source-only tests, representative exports, and isolated HARMYNTA CAD reproduction.
Historical reports/snapshots remain historical and traceable through the relocation
ledger; no compatibility symlinks or duplicate canonical roots. Existing shared
scene/depth links remain valid and do not duplicate arrays. Geometry and physical accuracy
are outside this path-only change. No new layout framework is required.
The legacy flat-layout migration leaves canonical work directories untouched and
returns its usual empty change list; cover that no-op in the existing migration test.

Model names and aliases are defined in [src/output_layout.py](src/output_layout.py).
Domain documents retain format-specific details; this contract owns storage rules.

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
