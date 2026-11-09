# Instrument modeling brief

Read [the project specification](../../SPEC.md) and [the instrument guide](README.md).
Apply the shared workflow before editing. `LOCAL.md` is optional tool setup only.
Build or refine the requested geometry, render and validate it; re-exporting an
unchanged recipe does not verify a geometry correction.

- Instrument and installed accessory: <model / variant>
- Evidence: <existing dataset, PDFs, captures and measurements>
- Task: <viewing / opening cover / sample insertion / simulation>
- Corrections: <component, observation, source image>
- Motion: <moving assemblies, direction, measured or estimated limits>
- Measurements: <value, units, endpoints, uncertainty; leave unknowns explicit>
- Requested outputs: <current model / explicitly requested delivery formats>
- Acceptance: <affected views/sections, dimensions, topology, motion and tolerances>

## Evidence and geometry

Use dataset/hkqai-instruments/<model>/ for authoritative evidence. Separate user_captures
from downloaded online_sources. Preserve capture dates and annotations beside
their originals. Use all relevant earlier evidence, not only the latest photo.
Do not add history, preview or duplicate source folders to the dataset.

Read the manufacturer's original specification diagrams first, for the exact
variant. Dimensioned PDFs establish envelope and datums. User measurements fill
undocumented dimensions; photographs establish profiles, installed accessories
and movement. Online photos support appearance. Perspective is not a measurement.
Record sources and uncertainty in evidence.json / measurements.json. Never imply
more precision than the evidence supports or silently promote estimates to facts.

## Authoring contract

Native OpenSCAD is authoritative: src/instruments/native/<model>/assembly.scad
and parameters_native.scad, with native/common.scad copied into each delivery.
Edit these files directly. Python orchestrates engine exports, validation and
packaging; it must not generate expanded geometry expressions from a second recipe.
Retired Python recipes are historical implementation, not current authoring inputs.

Structure every model as instrument -> fixed housing / moving assemblies ->
functional components -> optional hardware. Use named modules, short expressions,
local component datums, named dimensions and native loops. Keep profile samples in
small labeled tables, with meaningful precision. Explain geometry choices and
source observations. Do not dump sampled trig values or repeat expanded transforms.
Derive joint children from their parent's coordinate system; one transform moves a
door, lining, hinges and handles together. Never maintain separate closed/open solids.

Establish plan, side and end profiles where the actual housing requires them.
Use real Boolean subtraction for cavities, pockets, sockets, holes and clearances.
Dark surfaces cannot stand in for removed material. Use positive overlap inside
joined solids and explicit gaps between moving solids. Inspect sections and final
rendered meshes, not only F5 preview. Evaluate moving Boolean shells before their
transforms when needed to prevent OpenCSG phantom faces.

Provide details and fillets controls plus a simplified preset. Cosmetic rounding
and fasteners must be removable without deleting functional holes, recesses,
chambers or motion datums. Functional curves (bores, grips, bearings) retain their
geometry. Do not assume optional libraries such as BOSL2 are installed. If needed,
document and package the dependency. Keep the delivery easy to edit.

## Review and completion

Render closed, intermediate and open poses from the native source. Compare the
relevant views with the cited evidence. Check positive watertight solids, intended
hole counts, motion transforms and simplified geometry. Check affected components
individually; a whole-assembly manifold result does not prove clearance or fidelity.
Report unmeasured clearances and unresolved issues honestly. No physical-fit claim
from software checks alone. Any simulation mesh must be exported from this same
native source, with explicit mm-to-m conversion and documented simplification.

Run `.venv/bin/python -m src.instruments build --model <model> --check-cad`,
then `.venv/bin/python -m src.instruments validate --model <model>`.
Current model contents and optional sharing commands are defined in
[the instrument guide](README.md#validation-and-outputs); delivery requirements
are in [the delivery contract](../asset-delivery-rules.md).
Refresh affected current previews, evidence metadata and manifests. Generate
archives only when requested, preserving existing recipient snapshots (P08).
Record changes, checks and unresolved measurements in the existing development log.
