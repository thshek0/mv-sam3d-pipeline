# Setup

Converters run in this repo’s `.venv` (Pillow, tests). rembg, DA3, and MV-SAM3D are **optional** other envs. Point them with env vars; there are no machine-specific paths in this repo.

Driver scripts are in `src/`. Tests are in `tests/` (`pytest.ini` sets `pythonpath = src`). `env/` is **committed** (pip freezes + `camera_lock.json`). `patches/mv-sam3d.patch` is still required. Weights, clones, and `.venv` are gitignored. Leave existing checkpoints on disk; do not delete them.

```text
src/pipeline.py           CLI (depth method + mesh method)
src/methods.py            depth_da3 / da3_posed / hybrid, mesh_sam3d / mesh_tsdf
src/helpers.py            ingest, masks, crop, subprocesses
src/capture_realsense.py  D415 SPACE dump
src/tags.py src/tsdf.py src/da3_posed.py src/post.py src/vis.py src/time_run.py
env/                      pipeline-venv.txt, GPU/rembg freezes, camera_lock.json
patches/mv-sam3d.patch    CUDA_HOME + HF snapshots/<rev>
MV-SAM3D/ Depth-Anything-3/   clones at repo root, not under src/
```

## 1. Prereqs

- NVIDIA driver (this machine: 580.173.02, RTX PRO 6000 Blackwell, compute 12.0 / sm_120)
- `git`, `ffmpeg`
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) only if you run DA3 / SAM 3D / rembg
- [uv](https://docs.astral.sh/uv/) only for this repo’s small driver venv (Open3D + CoACD). Do not install torch+cu128 via uv.
- Hugging Face login / `HF_TOKEN` if `facebook/sam-3d-objects` is gated

Other GPUs: try step 3’s `default.yml` CUDA 12.1 stack first. If PyTorch has no kernel for your card, follow the cu128 swap below.

## 2. Clone

```bash
git clone https://github.com/thshek0/mv-sam3d-pipeline.git
cd mv-sam3d-pipeline

git clone https://github.com/devinli123/MV-SAM3D.git
git -C MV-SAM3D checkout abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd

git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
git -C Depth-Anything-3 checkout 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4

git -C MV-SAM3D apply ../patches/mv-sam3d.patch
```

Keep `MV-SAM3D/` and `Depth-Anything-3/` as siblings of `src/` (repo root). `run_da3.py` imports DA3 from `../Depth-Anything-3`. Do **not** clone `facebookresearch/sam-3d-objects`.

The patch sets `CUDA_HOME` from the env prefix (never `/usr/local/cuda*`) and resolves DA3 Hugging Face weights under `snapshots/<rev>/`.

## 3. GPU env (`sam3d-objects`) — optional (DA3 / MV-SAM3D)

```bash
micromamba create -n sam3d-objects -f MV-SAM3D/environments/default.yml
```

Default yml is CUDA **12.1**. Blackwell sm_120 needs **cu128** wheels. Replace torch / vision / audio / xformers / kaolin, then rebuild pytorch3d:

```bash
micromamba run -n sam3d-objects python -m pip uninstall -y torch torchvision torchaudio xformers pytorch3d kaolin

micromamba run -n sam3d-objects python -m pip install \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

micromamba run -n sam3d-objects python -m pip install \
  xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128

micromamba run -n sam3d-objects python -m pip install kaolin==0.18.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

micromamba run -n sam3d-objects python -m pip install numpy==1.26.4
```

Rebuild pytorch3d against that torch. `CUDA_HOME` must be the **env prefix**, not a system CUDA install. Include NVIDIA pip headers so `cusparse.h` is found:

```bash
PREFIX="$(micromamba run -n sam3d-objects python -c 'import sys; from pathlib import Path; print(Path(sys.executable).resolve().parents[1])')"
export CUDA_HOME="$PREFIX"
export FORCE_CUDA=1
export CPATH="$(echo "$PREFIX"/lib/python3.11/site-packages/nvidia/*/include | tr ' ' ':')"
micromamba run -n sam3d-objects python -m pip install \
  "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
```

Do **not** import `flash_attn`. A 2.8.3 wheel is a torch 2.5 binary (`undefined symbol`). Inference uses SDPA.

[`env/sam3d-objects-pip.txt`](env/sam3d-objects-pip.txt) is a freeze of what ran. It is **not** `pip install -r`. Use it to match versions if an import is missing after the steps above.

## 4. rembg env — optional (no LabelMe)

Separate env. Do not pip rembg, DA3, or SAM into `.venv`.

```bash
micromamba create -n rembg python=3.11 -c conda-forge
micromamba run -n rembg python -m pip install rembg==2.0.85
```

Full freeze: [`env/rembg-pip.txt`](env/rembg-pip.txt). Python that ran: 3.11.16 ([`env/rembg-python.txt`](env/rembg-python.txt)).

## 5. Driver `.venv`

Only for this repo’s CLI, `--seal` / `--coacd`, and tests (Open3D 0.19.0, CoACD 1.0.14, trimesh). Import Open3D **before** CoACD (`preprocess_mode="on"`; `auto` segfaulted).

```bash
uv venv .venv
uv pip install -r env/pipeline-venv.txt
```

That file is the **install list** (Open3D, CoACD, trimesh, Pillow, numpy, matplotlib, pytest), not a dump of a dirty venv. Do not pip the GPU freeze into `.venv`.

## 6. Weights

Gitignored. Skip this if `MV-SAM3D/checkpoints/hf/pipeline.yaml` already exists.

```bash
huggingface-cli download facebook/sam-3d-objects --local-dir MV-SAM3D/checkpoints
```

Need `MV-SAM3D/checkpoints/hf/pipeline.yaml` (`--model_tag hf`). Skip this section if you only convert LabelMe stills and pass `--da3-npz`.

DA3 downloads on first run, or:

```bash
hf download depth-anything/DA3NESTED-GIANT-LARGE
```

The cache **repo root** has no weights. The files live in `snapshots/<rev>/` and must include `config.json` + `model.safetensors`. A revision that ran: `8615eefb62f2db4f8d6ebaa59160086981672829`.

## 7. Point interpreters

```bash
export SAM3D_PYTHON="$(micromamba run -n sam3d-objects python -c 'import sys; print(sys.executable)')"
export REMBG_PYTHON="$(micromamba run -n rembg python -c 'import sys; print(sys.executable)')"
```

Or pass `--da3-python` / `--mvsam-python`. `DA3_PYTHON` / `MVSAM_PYTHON` override `SAM3D_PYTHON` for those steps. `REMBG_PYTHON` is only for `convert_rembg_to_masks`.

## 8. Smoke test

Processed scene in `examples/gripper/processed` (`images/` + `object/`). Use repo-local dirs:

```bash
source .venv/bin/activate
python src/pipeline.py -i examples/gripper/processed --scene gripper \
  --work-dir ./work --output-dir ./output
```

Needs `SAM3D_PYTHON` (or `--da3-npz` plus `--mvsam-python`). Success: `output/gripper/da3_sam3d/mesh.glb` and `mesh.png`. Missing DA3/npz raises; it does not skip.

Optional closed solid + convex parts (CoACD `t=0.05`):

```bash
python src/pipeline.py -i examples/gripper/processed --scene gripper \
  --work-dir ./work --output-dir ./output --seal --coacd
```

`--seal` writes `sealed.stl`. `--coacd` writes `convex_parts/` hulls. Video, stills, LabelMe, rembg, and crop are functions in `helpers`. Depth/mesh recipes are functions in `methods` (`depth_da3`, `depth_da3_posed`, `depth_rs_da3` / hybrid, `mesh_sam3d`, `mesh_tsdf`). `src/pipeline.py --method` is `da3` (default), `da3_posed`, `rs`, or `hybrid` (`rs_da3` / `fill_all_holes` aliases); `--mesh` is `sam3d` (default) or `tsdf`. `src/time_run.py` times `da3` → SAM3D (`--keep-runs` / `--no-keep-runs`, `--preview` / `--no-preview`). `src/tags.py`, `src/tsdf.py`, `src/da3_posed.py`, and `src/capture_realsense.py` remain thin CLIs. Live tag detect needs OpenCV in `SAM3D_PYTHON`; tests inject detections.

D415 capture needs `pyrealsense2` on the camera machine (not the repo `.venv`):

```bash
python src/capture_realsense.py
```

SPACE writes `{stem}_rgb.png` / `{stem}_depth.png` / `{stem}.json` under `input/captures_*`. Lock file is `env/camera_lock.json` (High Accuracy, laser 150). `1/2` exposure, `3/4` brightness, `9/0` gain, `Q` quit. No camera: `python src/capture_realsense.py --self-test --out /tmp/rs_self_test` writes a synthetic dump and ingests it.

## 9. Tests

```bash
python -m pytest tests
```

`pythonpath = src` is set in `pytest.ini`. Unset `SAM3D_PYTHON` / `DA3_PYTHON` / `MVSAM_PYTHON` if those point at a real reconstruct (tests must stay offline).

## 10. Versions that ran

| Piece | Version |
|---|---|
| GPU env Python | 3.11.0 |
| torch / torchaudio | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| xformers | 0.0.32.post2 |
| kaolin | 0.18.0 |
| pytorch3d | git `75ebeeae` |
| numpy | 1.26.4 |
| rembg | 2.0.85 |
| Open3D / CoACD | 0.19.0 / 1.0.14 |
| pytest | 9.1.1 |
