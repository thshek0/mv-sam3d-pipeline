# Retiring the local archive and preparing commits

Status: implemented, 2026-11-03. The root archive folder has been removed.
The local record is `output/maintenance/archive_retirement/`: `preflight.json`
contains the original inventory, `migration.json` records every disposition,
`path_map.json` resolves historical paths, and `summary.json` gives the outcome.

The 5.21 GiB archive contained 1,501 files. Retained sources/results occupy about
1.61 GiB of new regular files; estimated net recovery is 3.60 GiB, excluding small
navigation/audit files and filesystem overhead. Five full RealSense records were
recovered into their matching instrument captures using lossless NPZ compression.
All numeric arrays were checked. One timing baseline depth NPZ is retained.

Current instrument package files were verified unchanged. Checks covered 97
annotation/image pairs and 18 NPZ path updates. Twenty-five preexisting missing
image references in historical camera caches remain explicitly documented in
`reference_updates.json`; they were not replaced using guessed correspondences.
These retained hypotheses are not ready-to-run reconstruction inputs.

## Retention decisions

| Contents | Implemented disposition |
| --- | --- |
| Original six-view gripper photos and annotations | Recovered to `dataset/gripper/captures/legacy_6view/`; keep separate from the existing twelve-view set. |
| Unique heater captures, masks, scene ordering, RealSense depth and calibration | Merged into the corresponding dataset capture/version; deduplicate by hash, never by filename alone. |
| Coffee and PC processed image/mask pairs | Moved to `dataset/coffee/captures/legacy/scene/` and `dataset/pc/captures/legacy/scene/`; these are unique processed evidence, not original raw frames. |
| Manufacturer PDFs, extracted reference images and measurements | Place under the corresponding `dataset/instruments/<machine>/references/`; preserve source attribution. |
| Timing summaries, comparison PNGs and one baseline run | Moved to `output/gripper/benchmarks/gripper_timed/`; preserve per-run timings and one canonical depth/mesh pair. |
| Useful heater variants, single-view comparisons, CoACD, simplification and symmetry comparisons | Keep clearly named results under each family's `output/<family>/experiments/`; cross-object comparison pages belong under `output/comparisons/`. Do not overwrite current models. |
| DualBRep experiment | Kept outcome logs, preview, input mesh and final STEP results; removed intermediate PLY/OBJ/NPZ files. |
| Old inferred DA3 arrays and dense intermediate meshes | Removed inferred depth/pointmap intermediates while preserving camera metadata and comparison results. Retained mixed/metric PnP and hybrid hypotheses. Stochastic outputs may not reproduce exactly. |
| Instrument scratch/rejected-model/metadata tarballs | Recovered source references, all five measured-depth NPZs, selected meshes and metadata; removed containers and superseded dense scratch results. |
| Python bytecode and verified duplicate artifacts | Deleted after recording hashes and resolving references. Source aliases use relative file links where needed for pairing. |
| One-time migration/prototype scripts | Retired with purposes and hashes recorded. Scene-specific QY bundle-adjustment/SIFT details remain in method_notes.txt; those pilots were not promoted as generic tested functionality. |
| Detailed organization and cleanup ledgers | Moved to ignored `output/maintenance/`; keep this concise policy and development conclusions in versioned docs. |

This organization separates sources from results instead of renaming archive to
another catch-all folder. Detailed experiment artifacts remain local and ignored.
Existing accepted instrument models and the checked-in gripper example remain the
stable references.

## Migration acceptance checks

1. Record every old path, destination/deletion reason and file hash before edits.
2. Recover unique capture/annotation/reference data, including backup members.
3. Move selected results; update active links, manifests and NPZ path arrays without
   changing numeric arrays. Keep a relocation map for historical log references.
4. Remove verified duplicates, approved generated caches and obsolete scripts;
   retain a concise outcome record for retired experiments.
5. Verify hashes of moved files, annotation/image pairing and retained depth arrays;
   check current HTML links and pipeline tests. Remove empty archive directories.
6. Check source/docs/tests for active archive dependencies. Historical chronology
   may name old paths, but should point readers to the relocation ledger.

The timing CLI has already been redirected to
`output/<family>/benchmarks/<scene>_timed/`, or `output/benchmarks/` for unknown
families. It will not recreate archive. The retained timing baseline and summaries now use that location.

## Focused Git commits

Prepare three coherent commits, testing each before moving to the next:

1. **Organize dataset paths and benchmark outputs** — helpers, pipeline/run_dataset
   routing, time_run defaults, corresponding family/timing tests.
2. **Add shared parametric instrument authoring** — src/instruments (including
   storage, exporters, recipes and bootstrap inputs), tests/instruments, pinned
   instrument requirements, instrument workflow guide and prompt.
3. **Document layout, retention and reproduction** — root README/SETUP, experiment
   conclusions, maintenance policy and ignore rules.

The local file migration itself does not appear in Git: dataset, output and archive
are ignored. Avoid `git add .`; stage these explicit code/doc groups and inspect
`git diff --cached --stat` and `git diff --cached --check`. Keep runtime session
state, local environments, full datasets, generated ZIPs and obsolete scripts out
of commits. Use the separate model/project ZIPs for large colleague handoffs.

After migration and review, commit the groups above and push the reviewed branch
to origin. Do not rewrite
existing Git history to remove local files that were never tracked.
