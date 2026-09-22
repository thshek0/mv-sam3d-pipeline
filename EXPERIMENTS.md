# Experiments

The default product is the reconstructed mesh (`mesh.glb`). `--seal` and `--coacd` are optional. Below is what we kept, what we measured, and what we dropped.

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

## Best capture so far

Gripper LabelMe stills (the files in `examples/gripper/raw/`), after orientation-aware ingest and mask-center crop. Mesh in `examples/gripper/output/mesh.glb`.
