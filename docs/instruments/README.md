# Instrument modeling

Native OpenSCAD is the authoritative geometry source. Start with [prompt.md](prompt.md).
The current four models are `cary60`, `c16361`, `c13472`, and `c13534`.
For the installed OpenSCAD GUI and remote rendering see [the setup guide](../openscad/README.md).

For the compact colleague handoff, open the
[delivery gallery](../../output/packages/delivery/index.html) and copy or ZIP
`output/packages/delivery/hkqai-instruments/<model>/`. It contains the native SCAD
files, two previews and opening instructions. No meshes, simulation files or
evidence datasets are required. Generate a new collection with
`.venv/bin/python -m src.delivery --output <new-folder>`; see the main README's
[delivery instructions](../../README.md#ready-to-share-folders).

## Edit and rebuild

Edit `src/instruments/native/<model>/assembly.scad` for topology and assembly
relationships, and `parameters_native.scad` for dimensions. `native/common.scad`
provides small shared helpers; each model directory links to it. Published copies
contain three ordinary SCAD files, with no external libraries or meshes required.

The modules describe fixed housings, moving assemblies and functional components.
Door linings and handles inherit their door transform; nested handle motions are
local to that door. Native loops express repeated holes and hardware. Dimensions
are in millimetres and angles in degrees. Estimated values are labeled in the
parameter source and remain estimates after export.

```bash
.venv/bin/python -m src.instruments build --model cary60 --check-cad
.venv/bin/python -m src.instruments validate
.venv/bin/python -m src.instruments share
```

Omit `--model` to rebuild all four sequentially. `cad --check` uses the same native
build path. `publish` refreshes packaging/evidence from existing model assets;
it does not regenerate geometry. OpenSCAD is required for every build. Set
`OPENSCAD` if it is not on PATH. Python needs NumPy, trimesh and Pillow; pytest is
used for repository checks. Poppler is needed for PDF source-sheet generation.
No GPU, reconstruction weights, PyBullet or URDF are involved in a native build.

For a colleague's SCAD edits, copy the reviewed files back into the corresponding
`src/instruments/native/<model>/` directory before rebuilding. Shared helper edits
belong in `native/common.scad`. The former `--from-cad-parameters` import is retired
and gives an explicit migration message. A build never reads geometry back from
the current output `parameters.json`; that file is an evaluated snapshot of native parameters.

## Viewing and simplification

Open `openscad/assembly.scad` in the delivered folder. The top-level Customizer
controls `pose`, motion, `part`, `details`, `fillets`, and `simplified`.

- `pose="closed"` / `"open"` displays the two endpoints.
- `pose="custom"` enables the `opening` slider for Cary, Tau and C13472.
- QY has separate door, handle turn/pop and selector sliders, plus `upper_open`,
  `lower_open` and `handles_released` poses.
- Set `animate_motion=true`, View > Animate, FPS=20 and Steps=100 for an
  approximately five-second open/close cycle, subject to rendering speed.
- `simplified=true` disables cosmetic fillets and small hardware. `fillets=false`
  disables edge rounding alone; `details=false` disables cosmetic hardware alone.
  Functional bores, chamber openings, grip pockets and kinematic datums remain.
  Structural bevels and curved enclosure profiles are not cosmetic fillets.
- `part` selects a named component or group (`body`, `cover`, `holder`, `interior`,
  or QY `upper_door` / `lower_door`). Read the short component calls in the source
  for available names. Unsupported selections produce empty geometry.

The models demonstrate motion transforms, not forces, interlocks or contact dynamics.
For one solid STL in mm:

```bash
.venv/bin/python -m src.instruments stl --model cary60 --part socket_shell
```

Explicit STLs go to `work/hkqai-instruments/<model>/codex/printing/`, keeping the normal delivery slim.
Simulation geometry must be derived from native part exports; any later conversion
to metres, convex decomposition or articulation requires its own validation.
The older Python recipes/exporters remain for historical regression tests and are
not current authoring inputs or an alternative route for rebuilding these models.

## Current corrections and limits

- Cary recessed handgrip: user clarified that the problem was the pocket, not the
  whole sliding cover. IMG_9917 supports a right-offset, longer pocket. The native
  candidate uses a 58 x 90 mm footprint, centre offset +35 mm, front offset 15 mm,
  and two diagonal 16 mm corner arcs. These are photo estimates for review.
  The cutaway p5 holder handedness remains instrument-left socket/right adjusters.
- QY lower-left infill: IMG_9866, IMG_9867 and IMG_9869 show the small plinth at floor
  level. Its top is now floor +18 mm (estimated), independently editable. It no
  longer fills the room height. Black linings move with both doors, as in IMG_9845.
- Tau retains eight small through-holes plus the centre hole. Count/type are user
  confirmed; small-hole diameter and pitch-circle radius remain estimates.

These are review candidates. Software checks do not establish physical fit.
The dataset evidence.json records correction sources and pending user review.

## Validation and outputs

The builder exports actual OpenSCAD STL solids for closed, open, intermediate and
simplified configurations and rejects warnings, errors and invalid meshes. With
`--check-cad`, checks also exercise selected part transforms, animation times,
real recess/through-hole topology, pocket position, and simplified functional
openings. QY shell and lining transforms are sampled separately; this does not
certify a continuous collision-free sweep. Moving Boolean shells are evaluated
before transforms to avoid F5 phantom faces.

`--keep-validation` retains meshes, engine logs and intermediate motion PNGs under
`work/hkqai-instruments/<model>/codex/validation/native_latest/`. `--extras` also retains these
private diagnostics. Failed work remains in `.native_build/` for inspection; a
failed build does not replace the published package.

```text
src/instruments/native/
  common.scad
  <model>/assembly.scad, parameters_native.scad, common.scad -> ../common.scad

dataset/hkqai-instruments/<model>/
  evidence.json, measurements.json, states.json
  user_captures/                         dated originals and annotations
  online_sources/                       PDFs, downloads, retained PDF images

output/hkqai-instruments/<model>/
  openscad/assembly.scad, parameters_native.scad, common.scad
  assembly.png                          native closed/open views
  overview.png                          native orthographic views and dimensions
  parameters.json                       evaluated snapshot; not authoring input
  reports/cad_summary.json               engine checks and source hashes
  evidence/                             complete current dataset snapshot
  README.txt, index.html, manifest.json
```

Previews and dimension sheets come directly from the same SCAD as the model.
`validate` checks a standalone copy's hashes, dependencies and links. The shared
review is `output/comparisons/hkqai-instruments/current_review.png`; navigation starts
at `output/index.html`. No URDF, GLB or duplicate preview/dimension folder is
published. Only `share` makes ZIPs:

- `output/packages/instruments_models.zip`: three SCAD files per model, root
  preview/dimension images and concise viewing instructions; no evidence or JSON.
- `output/packages/instruments_sources.zip`: raw-image sheets grouped by model
  and user/online origin. Native photo pixels, JPEG quality 94; cited PDF pages
  rendered at 300 dpi into lossless PNG sheets. No annotation layers, depth,
  video or unselected catalog images. Selection audit is outside the archive at
  `output/maintenance/instrument_source_sheets.json`.

Optional `share --project` includes authoring source, docs, tests and datasets for
rebuilding. It is unnecessary for viewing/editing the native model handoff.

## Evidence handling

`user_captures/` groups your dated photographs and RGB-D captures with their annotations.
Annotated batches end in `_annotated`; distinct physical states remain separate.
Full-resolution PNGs replace HEICs after pixel, EXIF and color-profile verification.
The dataset has no separate review/preview folder. Builds generate review images
in output from canonical sources at each material record's `review_size`, preserving
the coordinate frame for `crop_pixels`. The user confirmed that the older exterior
images are online downloads: these live under `online_sources/exterior_annotated/`.
QY's `user_captures/undated_annotated/` contains three processed image/mask pairs
taken by the user, with capture date and correspondence to originals still unknown.
Image format does not determine source origin; annotations stay with their source batch.

`online_sources/` holds verified online material, including one copy of each source PDF.
`online_sources/pdf_images/` keeps selected embedded images/masks at native resolution and
selected diagram crops. `online_sources/index.json`
maps retained embedded images to their source PDF, page, dimensions and hash;
identical occurrences share a file. Omitted image counts document curation without
keeping stale paths. Instruments without a PDF need no extraction index. Complete pages and vector drawings remain
in the PDFs, so full-page PNG caches are omitted. Small embedded photographs
remain limited by their source resolution. C13472 has no local source PDF.

Geometry follows the matching manufacturer's specification diagrams and dimension
tables wherever available. `evidence.json` records `geometry_policy` and
`specification_sources`, with 1-based PDF page numbers: QY page 8, Tau page 5,
Cary brochure pages 15–16 and user guide page 16. Use original PDF vector detail
for drawings; raster crops are navigation aids. Establish dimensioned profiles,
datums and feature positions before interpreting perspective photos. User captures
confirm the installed variant, detail and motion; measurements fill undocumented
dimensions. Record and resolve conflicts explicitly, checking variant and endpoints
before adopting a deviation. Missing specifications and photo estimates stay labeled.

To refresh the index after curating images, without restoring removed images, run:

```bash
.venv/bin/python -m src.instruments.pdf_evidence
```

Only when full native extraction is wanted, install Poppler's `pdfinfo` and
`pdfimages` and add `--extract`. That explicit mode recreates missing embedded images.

Use `--model cary60` to select one instrument. The model package's `evidence/`
folder is a complete local snapshot; publishing never writes it back into the dataset. The share command creates separate slim deliveries and does not copy this entire folder into either ZIP.
The package manifest maps each evidence file to its canonical dataset source and hash.
Original photos, videos, annotations, PDFs and retained extracted images are included.
Dataset-prefixed paths in the copied metadata resolve under the package evidence folder.


## Implementation

`native_build.py` copies native source, calls OpenSCAD, validates exported meshes,
and generates review images. It never emits geometry expressions from Python.
`publish.py` stages the compact package and copies the current evidence dataset.
`sharing.py` and `source_sheets.py` produce the two deliveries. Dataset observations
and native SCAD remain authoritative; generated output can be recreated.

Run `.venv/bin/python -m pytest tests/instruments -q` for repository regressions.
Some tests cover retired geometry exporters; current geometry checks are in
`native_build.py` and are run by `build --check-cad`.
