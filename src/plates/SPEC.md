# HARMYNTA plate specification

Supplemental object record under the [project-wide specification](../../SPEC.md).
This file does not define the development process for the rest of the repository.

Updated 2026-11-07. User preference: simplify the plate directory and keep a short,
reproducible specification. Earlier geometry work is recorded retrospectively.
Follow the spec-before-implementation workflow in [AGENTS.md](../../AGENTS.md).

## Sources and layout

The shape is defined by [the profile JSON](inputs/plate_harmynta_white_22cm.json)
and [the CAD builder](solidworks.py), not by the dimension table alone.
Original evidence remains in `dataset/hestia-rl/plate_harmynta_white_22cm/`.
The object root is `output/objects/hestia-rl/plate_harmynta_white_22cm/codex/`:

- `products/HARMYNTA_220mm.step`: the **single current SolidWorks file**.
- `solidworks_asset.json`: CAD source/output hashes, millimetres and rebuild command.
- `asset.json` and existing GLBs: separate simulation assets in metres.
- `intermediate/`: existing evidence, previous revisions and verification records.

Do not keep an identical STEP in an extra delivery directory. On request, portable
collection handoffs still follow [the delivery rules](../../docs/asset-delivery-rules.md).

## Requirements

| ID | Requirement and provenance | Acceptance |
| --- | --- | --- |
| H01 | Nominal diameter 220 mm, height 40 mm; existing source records manufacturer's 22 cm / 4 cm specification. Initial chat used 22 mm / 4 mm, interpreted using that product specification. | Re-imported STEP bounds. |
| H02 | User-measured ring: outer diameter 102.5 mm, inner diameter 87.7 mm, external ring height 5 mm. Lowest contact is Z=0; outer wall attachment is Z=5. | Re-imported radial boundary checks. |
| H03 | Separate width reading 7.82 mm conflicts with the diameters, which imply 7.40 mm. | Retain both. Current modeling decision prioritizes diameters; resolution is not user-confirmed. |
| H04 | Rotational symmetry and simple geometry. | CAD: full revolution, one valid solid/shell, currently 10 faces; simulation: 64 sectors, at most 5,000 triangles (current 4,864). |
| H05 | Improve underside curvature from added photos and compare cross-sections. | Existing `intermediate/curvature_refit_20261107/` contains old/new sections and held-out image checks. |
| H06 | Reproduce one millimetre STEP directly from saved profile/code. | Isolated rebuild, valid STEP re-import, identical geometry DATA section to current STEP under the recorded environment. Timestamps may differ. |
| H07 | Keep one current CAD file within the existing object hierarchy. | No application-specific output root or duplicate current STEP delivery. Verify hashes before removing duplicates; record relocation under existing `intermediate/`. |
| H08 | Do not claim verified physical fit. | Gold-support fit test and actual SolidWorks session remain pending. |

## Rebuild

Environment used: Python 3.12.14, pythonocc-core/Open CASCADE 7.9.0,
NumPy 2.1.2, SciPy 1.16.2. From the repository root:

```bash
python -m src.plates.solidworks
```

Use an interpreter with the dependencies above; no DualBRep model output is
used. The builder must write the canonical STEP and verification records and
refresh `solidworks_asset.json` so a rebuild does not leave stale hashes.
The source-controlled `inputs/plate_harmynta_white_22cm_photo_fit.json` is also
required: the builder checks it against the profile and hashes it as provenance.
Update this accepted snapshot when accepting a new fit. Historical paths in the
profile describe evidence; the CAD rebuild does not require an existing output tree.
A fresh environment installation and a fresh photo refit are outside this check.

## Assumptions, results and decisions

- Estimated: central recess depth 1.99 mm, inner wall thickness, camera calibration
  and edge rounding. Ring edges transition 0.35 mm within the nominal envelope;
  flat contact-land diameters are 88.4 / 101.8 mm, not additional measurements.
- Existing photo check: held-out median silhouette errors improved 3.88→1.28 and
  7.74→3.67 pixels. This does not establish dimensional accuracy or physical fit.
- Existing CAD checks: 904 boundary samples; native/re-imported topology passes.
  Independent volume difference is 4.6151e-6 relative, below 1e-5. Fixed quadrature
  initially failed; adaptive integration and the documented threshold resolved
  the numerical check without changing geometry. Sampled CAD-to-GLB maximum
  distance was 0.11128 mm, a representation check only.
- Prior `output/SolidWorks/` creation was an assistant layout error. Its relocation
  was logged. The user's subsequent preference is one canonical CAD file, so
  remove the verified duplicate and retain its manifest as historical evidence.
- This simplification must preserve geometry. Rebuild comparison and layout/hash
  checks are recorded in `intermediate/solidworks_20261107/reproduction.json`.
- **Verified 2026-11-07:** An isolated workspace containing only the three listed
  inputs rebuilt exactly the same STEP geometry DATA section as the previous
  delivery. The canonical rebuild matches too; only export metadata differs.
  Refreshed registry hashes, a temporary standard export, single-current-STEP
  storage and output-root layout checks passed. No new physical-fit claim.
