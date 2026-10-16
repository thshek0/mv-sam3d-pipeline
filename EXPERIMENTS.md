# Experiments

What we kept, measured, and dropped. Product is the mesh (`*_mesh.glb`). `--seal` / `--coacd` are optional post-steps.

**Terms.** Generation = `--mesh sam3d` (MV-SAM3D). Reconstruction = `--mesh tsdf` (Open3D fusion). TSDF needs real extrinsics; unposed RS/hybrid (identity `w2c`) is refused — use SAM3D or `da3` / `da3_posed` TSDF.

---

## Datasets

Counts are from `dataset/<name>/` and the staged `output/<name>/scene/`. **Frames** is one number when every captured view is in the scene. `used/captured` means some views were dropped. **Tags** is whether AprilTag poses were stored. **No** means the RealSense npz is unposed, not that a tag is absent from the photo. Masks are LabelMe. The JSON does not say whether SAM 2 or a person drew the outline. No current folder uses rembg.

| Dataset | Frames | Resolution | Camera | Tags | Masks |
|---|---|---|---|---|---|
| gripper (`examples/gripper`) | 12 | 5712×4284 | phone | no | LabelMe |
| gripper_v2 | 12 | 4032×3024 | phone | no | LabelMe |
| heater_v2 | 21 | 5712×4284 | phone | no | LabelMe |
| heater_v3 | 14 | 640×480 | RGB-D; model and preset not in the file | yes | LabelMe |
| heater_v4 | 17 | 640×480 | D415, High Accuracy | yes | LabelMe |
| heater_v5 | 12 | 1280×720 | D415, High Accuracy | yes, 12 posed | LabelMe |
| device1_all | 7 | 1280×720 | D415, High Accuracy | yes, 7 posed | LabelMe |
| device1_lower | 7/13 | 1280×720 | D415, High Accuracy | yes, 7 posed | LabelMe |
| device1_upper | 11 | 1280×720 | D415, High Accuracy | yes, 11 posed | LabelMe |
| device2 | 12 | 1280×720 | D415, High Accuracy | yes, 12 posed | LabelMe |
| device3 | 9/16 | 1280×720 | D415, High Accuracy | yes, 9 posed | LabelMe brush; 7 other views are a rectangle or polygon |
| bird1_v1 | 12 | 1280×720 | D415, High Accuracy | no | LabelMe |
| bird1_v2 | 12 | 640×480 | D415, High Accuracy | no | LabelMe |
| bird2 | 12 | 1280×720 | D415, High Accuracy | no | LabelMe |
| bowl | 9/12 | 1280×720 | D415, High Accuracy | no | LabelMe |
| structure_before | 26/28 | 640×480 | D415, High Density | no | LabelMe |
| structure_after_v1 | 30 | 640×480 | D415, High Density | yes, 30 posed | LabelMe |
| structure_after_v2 | 28 | 640×480 | D435I, High Density | yes, 28 posed | LabelMe |

device1_lower drops 6 views that did not pose. bowl and structure_before drop views with no LabelMe file. device3 drops 6 rectangles and 1 polygon that cover almost the whole frame; the scene keeps the 9 brush masks. heater_v3 has intrinsics and 640×480, and v4 on the same setup names the D415.

---

## Pipeline lessons

| Topic | Finding |
|---|---|
| **Orientation** | Phone stills often store rotated pixels; LabelMe is upright. Ingest applies EXIF before writing `images/` (else masks stretch — gripper became a flat T). |
| **Crops** | DA3’s image-center crop cuts long objects. We crop a square on the mask bbox with 1.15× margin. |
| **Masks** | One folder is all LabelMe or all rembg. Mixing pulls table/people into the object. |
| **CoACD** | Default `t=0.05`. Coffee → 2 hulls; gripper ~50 with fingers separate. Higher `t` fuses fingers; second pass on crumbs explodes parts. |

---

## Dropped approaches

- **VGGT + Poisson** — snapped to the table; replaced by MV-SAM3D.
- **DualBrep → STEP** — 0 SOLID on our scans (ABC demos were mostly SOLID).
- **One convex hull / capped 500-face hull** — too fat for grasping.
- **Quadric collapse to 10k faces** — melted detail; reconstructed mesh is the look mesh.
- **Vertex clustering** — silhouette OK, not watertight.
- **Heater from video + rembg** — fake legs; capture problem, not a missing algorithm.
- **Well-vs-wall fuse rule** — heater-only; dropped. Public `--method hybrid` = keep valid RS, fill holes with scaled DA3. Through-grid wells stay plugged — use `--method rs` + TSDF to keep holes.
- **Occupancy / silhouette IoU** — never recovered tag poses into SAM3D’s box. Judge with `*_mesh.png` and TSDF vs SAM3D.

---

## Heater case study

**Goal.** Dry-bath block: honest **5×5 through-grid** plus chassis for collision. Each version kills one confound from the previous. What remains is brushed-silver RealSense failure.

### Depth codes (on a given capture)

| Code | Depth | Pose |
|---|---|---|
| **A** | DA3 (dense plate) | DA3 |
| **B** | D415 (`Z=0` invalid) | AprilTag PnP |
| **C** | RS where valid, DA3 in holes | PnP |

D415 Z is IR speckle stereo; RGB is not used for Z.

### Capture ladder

| Ver | Dataset | Method | Eliminated | Result | Next |
|---|---|---|---|---|---|
| **v1** `heater` | phone video | rembg + A | — | fake legs | stills + LabelMe |
| **v2** `heater_v2` | 21 iPhone stills, LabelMe | A | rembg / video | best cups, **closed lid** | RealSense |
| **v3** `heater_v3` | 640 RGB-D, tags on lid, 8/14 posed | A/B/C | “DA3 only” | A≈C closed; B sparse; C plugs wells | table tags + High Accuracy |
| **v4** `heater_v4` | 640, HA, lid+table tags, 16/17 | A/B | weak poses | **RS shows 5×5** overhead; SAM3D still fills | 1280, darker, overhead |
| **v5** `heater_v5` | 1280×720, HA, 12/12 posed | A/B | res / blown RGB / poses | A pretty grid; B honest holes; walls still `Z=0` | freeze capture; vary methods |

Paths for early versions used legacy `output/heater_v*/{da3,rs}/`; current layout is `output/<name>/{scene,depth,meshes}/`.

### Methods on frozen v5

Same 12 views. SAM 2 could not cut wells — TSDF first (no hole mask).

| Method | Eliminates | Result |
|---|---|---|
| **RS TSDF** | SAM3D generative fill | Through-grid lid, **no chassis**. Red walls in visibility preview = no RS Z on silver. `meshes/rs_tsdf_mesh.glb` |
| **1-view SAM3D** | multi-view voting | Regular cups, closed top, invented sides (`archive/heater_v5_da3_v0/`) |
| **da3_posed** | pose-free DA3 cameras | Fills walls RS misses; wells stay a dense plate. `meshes/da3_posed_sam3d_mesh.glb` |
| **SAM3D on da3_posed** | pose-free pointmaps | Cleaner closed box; **lid still closed** |

More orbit video was not run: same grazing silver → still `Z=0`.

### Ruled out

| Suspect | Status |
|---|---|
| rembg / video invents legs | killed (v2) |
| no metric depth | killed (v3 D415) |
| too few poses | killed (v4–v5 tags) |
| RS never sees the 5×5 | **false** — lid holes are in overhead RS |
| 640 / blown RGB | killed (v5) |
| SAM3D is the only problem | **false** — TSDF keeps holes, loses walls |
| TSDF ignored side photos | **false** — those pixels have no RS Z |
| DA3 cameras cause closed wells | **no for wells**; known pose helps metric **sides** |
| One clear photo fixes SAM3D | **false** (1-view still a plate) |

**Still true:** D415 returns no Z on brushed silver walls (and well bores). DA3/SAM3D paint those pixels (RGB models). Hybrid without a hole-size rule closes the grid like v3 C.

**Next on heater_v5.** Prefer `--method hybrid` meshes already on disk. Optional: RS-only TSDF if holes must stay. New capture only for metric walls: matte chassis, face-on, table tags — not more grazing silver.

Do not default the public CLI to `da3_posed` (needs tags). Keep pose-free `da3`.

---

## Lab devices — TSDF vs SAM3D

Same five sets. TSDF: voxel 0.002, sdf 0.006, depth-trunc 2, `--keep-largest`. Compare `meshes/{da3_posed,hybrid}_sam3d_mesh.png` to `meshes/{rs,da3_posed,hybrid}_tsdf_mesh.png`.

| Capture | RS TSDF | da3_posed TSDF | hybrid TSDF | Note |
|---|---|---|---|---|
| device1_lower | ~10k, almost empty Z | 106k | 1.3M | Dark; few tags |
| device1_all | 368k | 431k | 324k | RS = flat lid plate |
| device1_upper | 82k | 201k | 410k | RS sparse |
| device2 | 203k | 442k | 308k | RS has shape, open |
| device3 | 302k | 951k | 1.6M | **RS keeps the L**; posed/hybrid smear |

**device3 RS TSDF** is the honest reconstruction next to SAM3D’s closed L-box: measured shell, no fake fill. Posed/hybrid TSDF = DA3 plate + fusion (same failure mode as hybrid fill).

`--method rs` is the best **depth map** when stereo+tags and holes must stay. Hybrid fills holes. SAM3D on hybrid/posed DA3 still closed the heater lid.

---

## Best capture so far

Gripper LabelMe stills (`examples/gripper/raw/`), orientation-aware ingest + mask-center crop → `examples/gripper/output/mesh.glb`. Public default: `da3` → SAM3D.

Heater / shiny lab gear: no through-grid CAD + chassis yet. RS TSDF = honest lid. Posed SAM3D = prettier closed box. Collision = `--seal` then `--coacd t=0.05` on the mesh you pick.
