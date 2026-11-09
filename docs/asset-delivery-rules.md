# Asset storage and delivery rules

Agreed on 2026-11-05. This is the standing reference for future asset/export work.
The executable collection policy is [`src/delivery_profiles.json`](../src/delivery_profiles.json).

## What to send

Copy or ZIP **one complete object/variant folder** from
`output/packages/delivery/`. The recipient must not need the author's workspace,
absolute paths, reconstruction environments, or files outside that folder.

| Collection | Required delivery | Open this |
| --- | --- | --- |
| QAI instruments (`hkqai-instruments`) | Native OpenSCAD, required SCAD dependencies, previews and instructions | `openscad/assembly.scad` |
| Hestia (`hestia-rl`), except birds | MuJoCo MJCF, visual mesh, individual convex collision meshes, required textures and previews | `model.xml` |
| Both birds (`bird_swallow`, `bird_mahjong`) | STL and geometry previews | `<object>.stl` in a slicer |
| HKQAI trial | Exclude from collection deliveries; preserve capture identities | Existing products |

The birds have uncalibrated physical scale. The user explicitly chose to preserve
their existing coordinates and **set the size in the slicer**. Do not silently
convert them to millimetres, choose a print size, repair/simplify their accepted
geometry, or substitute their collision approximations for the printable surface.
STL has no units field; make this explicit in the recipient's instructions.

The weighing instrument is named **electronic balance**. Use `electronic_balance`
for the canonical dataset/object folder, and `electronic_balance_round` /
`electronic_balance_square` for the two delivery variants. `scale` remains a
compatibility alias for commands and the historical product recipe/physics key.
Do not recreate a second canonical `scale/` dataset or object folder.

## Minimum authoritative storage

Preserve the best original source and information that cannot be reliably regenerated:

- Native SCAD with parameters and included modules for instruments.
- Recipe code, parameters, profiles and necessary appearance inputs for procedural objects.
- Accepted reconstructed meshes and calibration/provenance for mesh-derived objects.
- Reviewed collision geometry, or its reproducible construction recipe.
- Explicit mass, inertia, friction and joint information when available; label estimates.
- A common `asset.json` registry with file hashes, units, coordinates, variants and rebuild information.

An accepted imported mesh is a source dependency, even when a Python recipe uses
it. In particular, preserve the black spoon's accepted geometry and provenance.
Keep captures/evidence if remeasurement or reconstruction must remain repeatable.

Generate delivery formats and previews from these sources. Do not keep mandatory
STL + OBJ + GLB + STEP copies for every object. Do not replace editable CAD with
a mesh, or imply that converting a mesh to STEP restores precise CAD features.
Existing exports are snapshots, not competing sources of truth. Retirement of
old working outputs is a separate operation requiring dependency checks.

## Uniform MuJoCo packaging

```text
object_or_variant/
  model.xml
  meshes/
    visual/                  # appearance mesh
    collision/               # one OBJ per registered convex piece
  textures/                  # only images referenced by MJCF
  thumbnails/                # individual rendered views
  preview.jpg                # directly openable contact sheet
  index.html                 # instructions and embedded previews
  README.txt
  asset.json                 # units, provenance, checks and delivered file hashes
```

Enumerate registered scene nodes and preserve their transforms. Select variants
through metadata/configuration. Do not maintain object-specific OBJ filename lists
or fixed part counts in the delivery exporter. Construct geometry in its source
recipe; exporting does not guess a new decomposition from an object's name.

Keep visual and collision geometry separate. Merging collision pieces into one
convex hull fills cavities such as dishes, bowls, spoons and containers. XML defines
materials directly, so an unused MTL or duplicate GLB/STL is not needed in the
MuJoCo package. Share assets only when their geometry, coordinates and purpose match.

Current MuJoCo deliveries use metres and preserve existing physics. Three plates
without recorded mass/inertia are **static**, labelled as such. Do not invent mass
to make an asset dynamic. Other masses/friction values remain estimates; neither
compilation nor a brief simulation establishes physical calibration.

## Previews are required

Every object needs visible previews plus a root `preview.jpg`, so a colleague can
inspect it without an HTML viewer or the target application. The collection also
has `output/packages/delivery/preview.jpg` showing all deliveries. HTML embeds its
preview images to avoid broken image references in local-file viewers; direct
image files remain available for normal image viewers.

Render MuJoCo previews from the exported model, and bird previews from the exported
STL. For native instruments, use published previews only when source and preview
hashes match. Check images visually: successful image decoding and MuJoCo compilation
do not establish correct colours, visibility or camera framing.

The 2026-11-05 electronic balance check found an extra texture flip producing black
silhouettes. MuJoCo's OBJ import already converts V coordinates; these exported PNG
textures use `vflip="false"`. Keep the compiled texture/UV regression check.

## Generate and validate

```bash
.venv/bin/python -m src.delivery --output output/packages/new_delivery
.venv/bin/python -m src.delivery --only electronic_balance cary60 --output /tmp/selected_delivery
```

Use a new or empty destination. The exporter stages the collection and publishes
it only when all selected objects succeed. A rebuild must not silently overwrite
an existing recipient's snapshot.

Before handing off:

1. Verify registered source hashes, units and required dependencies.
2. Copy each package to an unrelated directory and check all runtime references.
3. Compile MuJoCo models and run the finite-state/warning smoke test.
4. Check bird STLs for closed, consistently wound positive-volume surfaces.
5. Check SCAD includes and the published source/preview snapshot.
6. Inspect the preview images and verify the gallery/file hashes.

These checks do not certify printing wall thickness/supports, native CAD rebuilding,
task-specific contacts, or physical accuracy. Record exactly what was checked.

The handoff is portable for **use**. Reproducing it additionally requires the
source inputs and suitable software dependencies; the Git repository alone does
not include the Git-ignored dataset, generated deliveries or model weights.
