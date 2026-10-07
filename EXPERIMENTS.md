# Experiments

The default product is the mesh (`mesh.glb`). `--seal` and `--coacd` are optional. Scripts are in `src/`. Below is what we kept, what we measured, and what we dropped.

**Generation** (`--mesh sam3d`) is the official name for MV-SAM3D. **Reconstruction** (`--mesh tsdf`) is TSDF fusion of a depth npz. Do not say “constructive.”

## Orientation

Many camera stills store pixels rotated and tag the upright view in metadata. LabelMe is drawn on the upright image. If ingest ignores that tag, the mask is stretched onto the wrong axes. On the gripper that produced a **flat T**. The ingest step applies the tagged orientation before writing `images/`.

## Crops

DA3’s default batch crop is image-center. On a long gripper that cut off the forearm. We crop a square around the mask bounding box with a 1.15× margin. If the mask is empty, the square is image-center on the short side.

## Segmentation mask generation

One folder is either fully annotated (JSON beside every still) or fully automatic (rembg). Mixing them in one folder pulled table, face, or a person into the “object” mask.

## Convex decomposition

`--coacd` uses **t=0.05**. Coffee split into 2 hulls (cap + body). The gripper stayed about 50 hulls with fingers separate.

- t=0.08 / 0.10 fused the fingers.
- t=0.02 with a hull cap of 8 or 16 made fat blobs.
- A second CoACD pass on small hulls exploded the coffee cap into **1461** crumbs.
- Smoothing then hulling a bottle neck filled the grasp with a cone.

You still pick `t` per object if 0.05 is wrong; it is a concavity split, not a semantic part split.

## What we dropped

- **VGGT + Poisson.** Snapped to the table. Replaced by MV-SAM3D.
- **DualBrep autoencoder → STEP.** ABC demos were mostly SOLID. Our scans produced **0 SOLID**. A SAM-3D gripper feed (~191 faces) sewed 0/24.
- **One convex hull** of the whole object (including capped 500-face hulls). Too fat for grasping.
- **Open3D quadric collapse to 10k faces** as a “look” mesh. Melted flanges and jaws. The reconstructed mesh is the look mesh; we do not decimate it for appearance.
- **Vertex clustering.** Silhouette looked fine but the mesh was not watertight.
- **Heater from video + rembg.** Result was a box with fake legs. That is a capture problem (few views, weak texture), not a missing algorithm. Better photos, or marks on the object, before changing the pipeline.
- **Well vs wall fuse.** Enclosed-hole vs outline-miss was a heater-only rule. Dropped. Public `--method hybrid` is only: keep valid RS, fill remaining object pixels with scaled DA3. On a through-grid that still plugs wells — use `--method rs` + TSDF if holes must stay empty.
- **Occupancy / silhouette IoU.** SAM3D lives in a canonical box; tag cameras are camera-0 world. Silhouette search never recovered a real pose. Dropped `src/occupancy.py`. Judge meshes with `mesh.png` and TSDF vs SAM3D, not a 2-D score.

## Heater process

One object (dry-bath block). Goal stayed the same: a **5×5 through-grid** plus a chassis for collision. Each version changes **dataset** or **method** to kill one confound from the version before. The shiny-metal RealSense failure is what remains after those jumps.

A/B/C on a given capture:

| Code | Depth | Pose |
|---|---|---|
| **A** | DA3 (dense, always a plate) | DA3 |
| **B** | D415, `Z=0` invalid | AprilTag PnP |
| **C** | RS where valid, DA3 in every hole | PnP |

D415 Z is **IR speckle stereo**. RGB is not used for Z.

### v1 — phone video (`heater`)

- **Dataset:** short phone clip. **Method:** rembg + DA3 + SAM3D.
- **Try:** first Real2Sim-style mesh, no labels.
- **Eliminate:** nothing (baseline).
- **Result:** box with **fake legs**. `archive/heater/`.
- **Jump because:** rembg and few views invent geometry. Next: stills + a real outer mask.

### v2 — phone stills (`heater2`)

- **Dataset:** 21 iPhone stills, LabelMe outer mask. **Method:** A only.
- **Eliminate:** rembg, video, missing mask.
- **Try:** SAM3D with a correct silhouette and more well texture.
- **Result:** **best-looking cups** so far, still a **closed lid**. DA3 is a plate; wells are RGB prior. `output/heater2/da3/`.
- **Jump because:** cups look good but are not measured. Next: RealSense so holes in Z can be real.

### v3 — first D415 (`heater3`)

- **Dataset:** 640×480 RGB-D, tags **on the lid**, 14 LabelMe; B kept **8**. **Method:** A / B / C.
- **Eliminate:** “we only have DA3 depth.”
- **Try:** metric stereo vs DA3; C as “use RS and fill gaps.”
- **Result:** A≈C closed lid. B sparse / ~3×3. C **plugs wells** (same as A). PnP fails when the lid tag is grazing. `output/heater3/{da3,rs}/`. Hybrid C is archived.
- **Jump because:** missing poses (lid-only tags) and a weak stereo preset. Cannot yet blame shiny metal — too many views never posed. Next: tags on the **table**, High Accuracy.

### v4 — High Accuracy + table tags (`heater4`)

- **Dataset:** 640×480, High Accuracy, laser 150, tags **lid + table**, 17 stills, B **16/17**. Color often blown white. **Method:** A / B.
- **Eliminate:** dropped PnP; “maybe RS never sees the 5×5.”
- **Try:** hold pose, ask whether **lid** stereo works.
- **Result:** **RS depth does show 5×5** on overhead frames. SAM3D still irregular pits + cable. Grazing/side views have dead Z and vote a solid box. `output/heater4/{da3,rs}/`.
- **Jump because:** lid holes are in RS; the mesh fill is SAM3D, not “RS is useless everywhere.” Remaining dataset confounds: 640, clipped RGB, too many side shots. Next: 1280, darker color, mostly overhead.

### v5 — 1280 overhead (`heater5`)

- **Dataset:** 1280×720, High Accuracy, laser 150, color exp **80**, tags lid+table, **12/12 posed**, paper scraps in some wells. **Method:** A / B on the same LabelMe.
- **Eliminate:** resolution, blown white, missing poses.
- **Try:** cleaner lid RGB and Z, same object.
- **Result:** A nicer regular grid (RGB again). B honest 5×5 in depth, paper lumps in the mesh. Views 4 and 6 almost empty Z. RGB of v7/v11 **shows the chassis**; RS on those walls is still `Z=0`. `output/heater5/{da3,rs}/`.
- **Jump because:** capture is now “good enough” to freeze. Further jumps are **methods on this same 12-view set**, so we do not mix in a new camera day.

### v5 methods (same dataset)

SAM 2 could not cut wells (holes not detected; too slow). That is why TSDF is first: no hole mask required.

**TSDF** — `src/tsdf.py` on native RS+PnP, outer LabelMe only (`--voxel-length 0.002 --sdf-trunc 0.006 --depth-trunc 2 --keep-largest`).

- **Eliminate:** SAM3D’s generative fill.
- **Try:** if RS has holes, the mesh should have holes.
- **Result:** **through-grid lid**, no chassis. `output/heater5/tsdf/`. `depth_visibility_preview.png` is photo | overlay for all 12 views (cyan = object and Z>0, red = object and Z=0).
- **Isolated:** missing walls are not “TSDF forgot the photos.” Those wall pixels have no RealSense depth. **IR stereo does not work on the silver sides.** Extra RGB of the box does not change that.

**1-view SAM3D** — heater5 `0.png` + pose-free DA3.

- **Eliminate:** multi-view voting for a closed brick (grazing frames).
- **Try:** one lid photo you can “see” the 5×5 in.
- **Result:** regular cups, closed top, invented sides. Archived (`archive/heater5_da3_v0/`). RGB prior, not RS.

**DA3 with PnP** — `src/da3_posed.py` (`w2c` + factory `K`, `align_to_input_ext_scale`).

- **Eliminate:** DA3 guessing cameras (pose-free DA3 on shiny views).
- **Try:** better **depth image** on metal, metric to tags.
- **Result:** `depth_vs_rs.png` — DA3 **fills walls** RS leaves empty; wells stay a **100% valid plate**. `output/heater5/da3_posed_sam3d/`.
- **Isolated:** known pose is not why wells are filled (DA3 is dense by design). Known pose **does** give side depth where RS is blind. That is RGB depth, not D415.

**SAM3D on posed DA3** — `src/pipeline.py --da3-npz work/heater5_da3_posed/da3_output.npz`.

- **Eliminate:** pose-free DA3 as the SAM3D pointmap.
- **Try:** cleaner box from the better cameras.
- **Result:** cleaner closed box than v5 A; cups clearer from the side; **lid still closed**. Preview is `mesh.png`; download `mesh.glb`. `output/heater5/da3_posed_sam3d/`.

Video around the object was considered and **not** run: more frames at the same grazing silver angle would still be `Z=0`. That would not isolate shiny-surface stereo.

### What the jumps ruled out

| Suspect | Killed by | Still true? |
|---|---|---|
| rembg / video invents legs | v1 → v2 LabelMe stills | no |
| no metric depth | v2 → v3 D415 | no |
| too few poses | v3 → v4 table tags (16/17, then 12/12) | no |
| RS never sees the 5×5 | v4/v5 overhead RS | **no — lid 5×5 is in RS** |
| 640 / blown RGB | v4 → v5 1280 + exp 80 | no |
| SAM3D fill is the only problem | v5 TSDF | **no — TSDF keeps holes, loses walls** |
| TSDF ignored the side photos | `depth_visibility_preview.png` red walls | no |
| DA3 cameras were the problem | v5 posed DA3 | **no for wells**; **yes for metric sides** |
| SAM3D needs one clear photo | v5 1-view | no (still a plate) |

**Left:** D415 does not return Z on **brushed silver walls** (and well bores). DA3/SAM3D will paint those pixels because they are RGB models. Fusion of the two without a hole-size rule repeats v3 C and closes the grid.

### Next dataset / method

Same heater5 stills: leftover `output/heater5/rs_da3/` is the old wall-only preview dump. Current fuse is `--method hybrid` (folder `hybrid_sam3d` / `hybrid_tsdf`). **Still not run on heater5:** TSDF on RS-only depth if you want holes kept. New capture only if walls must be metric: matte the chassis, face-on, table tags still in view. Do not recapture “more of the same silver at a grazing angle.”

Do not default the public CLI to `da3_posed` (needs tags). Do not archive the `da3` method.

## Lab captures — TSDF vs SAM3D

Same five sets. TSDF at voxel 0.002, sdf 0.006, depth-trunc 2, `--keep-largest`. Compare `output/<name>/{da3_posed,hybrid}_sam3d/mesh.png` (SAM3D) to `output/<name>/{rs,da3_posed,hybrid}_tsdf/mesh.png`.

| Capture | RS TSDF | da3_posed TSDF | hybrid TSDF | Note |
|---|---|---|---|---|
| 1547 | 10k faces, almost empty Z | 106k | 1.3M | Dark; few tags; RS is a scrap |
| 1556 | 368k | 431k | 324k | RS is a flat lid plate |
| 1558 | 82k | 201k | 410k | RS sparse |
| 1601 | 203k | 442k | 308k | RS has shape, still open |
| 1606 | 302k | 951k | 1.6M | **RS keeps the L**; posed/hybrid TSDF smear |

**1606 RS TSDF** is the reconstruction to put next to SAM3D’s closed L-box: measured shell, no fake fill, jagged, chassis incomplete. Posed/hybrid TSDF is DA3 painting a plate, then fusion — same failure as fill-all, without SAM3D.

`--method rs` is the best **depth map** when you have stereo and tags and want holes kept. Hybrid fills those holes. SAM3D on hybrid/posed DA3 still closed the heater lid.

## Best capture so far

Gripper LabelMe stills (`examples/gripper/raw/`), after orientation-aware ingest and mask-center crop. Mesh in `examples/gripper/output/mesh.glb`. Public default: `da3` → SAM3D.

Heater / shiny lab gear: no through-grid CAD plus chassis yet. TSDF on RS is the honest lid. Posed SAM3D is the prettier closed box. Collision is `--seal` then `--coacd t=0.05` on whichever mesh you pick — not the raw TSDF.
