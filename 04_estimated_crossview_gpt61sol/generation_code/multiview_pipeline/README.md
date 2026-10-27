# MultiView camera and rendering pipeline

This package turns two or three numbered RGB images into a smooth camera-path render. It runs VGGT Omega for cameras and scene depth, plans a 21-frame path, renders with Stable Virtual Camera (SEVA) v1.1, and can optionally estimate depth on the final RGB with MoGe-2-L.

## Input

One directory containing `image_1.png`, `image_2.png`, and optionally `image_3.png`. All input images must have the same dimensions. The pipeline reads these numbered PNG files; other files in the directory are ignored.

## Models and environments

- **VGGT Omega:** install the [official repository](https://github.com/facebookresearch/vggt-omega) in a Python environment with CUDA PyTorch, NumPy, SciPy and Pillow. Download the `VGGT-Omega-1B-512` checkpoint and pass the checkout and checkpoint paths to the command below.
- **SEVA:** install the [official Stable Virtual Camera repository](https://github.com/Stability-AI/stable-virtual-camera) and its demo dependencies in a separate Python environment. Place the v1.1 weights in a directory as `modelv1.1.safetensors`. Provide that environment's Python path. The upstream model also loads its VAE and CLIP dependencies; follow its model access and setup instructions.
- **Optional depth:** install [MoGe](https://github.com/microsoft/MoGe) in a Python environment and provide that environment's Python path with `--depth-python`. The script loads `Ruicheng/moge-2-vitl-normal`.

Known compatible source revisions are VGGT Omega `39a0cb8af88554f15ddcb5354cd52bde588fa014` and SEVA `fe19948e9b7bea261ab2db780a59656131404a83`. The model weights are not bundled. Follow the upstream access and license terms for each model.

For a fully local SEVA setup, `enable_local_seva_weights.py --seva-root /path/to/stable-virtual-camera` makes the pinned SEVA checkout accept `--seva-vae-dir` and `--seva-clip-weights`. This modification is optional when the upstream dependency downloads work.

## Run one case

Run with the VGGT environment's Python:

```bash
python run_case.py \
  --case-dir /data/MVC3I-002-Q1 \
  --output-root /scratch/multiview_run \
  --vggt-root /models/vggt-omega \
  --vggt-checkpoint /models/vggt_omega_1b_512.pt \
  --seva-python /envs/seva/bin/python \
  --seva-repo /models/stable-virtual-camera \
  --seva-model-dir /models/seva-v1.1
```

Use the same command for a two-image case. Add `--depth-python /envs/moge/bin/python` to save a depth PNG, a raw depth/validity NPZ, and a JSON sidecar. Set `--depth-near` and `--depth-far` to the desired display range in metres; the PNG uses reversed Turbo colours and black for invalid pixels.

`--plan-only` runs VGGT and saves the camera path without starting SEVA. Each stage can also be called on its own: `estimate_pose.py`, `plan_path.py`, `render_seva.py`, and `estimate_depth.py` expose `--help`.

## Camera paths

- **Two images:** their estimated camera centres are combined into one centre. The path starts at Image 1 and smoothly rotates to the shortest-arc orientation midpoint between Images 1 and 2. Camera position stays fixed. There are two context frames and 19 target frames.
- **Three images:** the default start view is the angular medoid of the three estimated orientations; `--start-view 1`, `2`, or `3` overrides it. The camera moves horizontally backward from that view, with optional lateral motion (`--lateral-ratio`) and a small look-at correction. There are three context frames and 18 target frames. This is a configurable view-planning heuristic: input images alone do not identify a unique translated destination.

`--three-baseline` controls the endpoint displacement after SEVA's camera normalization. Its default is `0.55` in SEVA's normalized camera units. Increase it for more motion; decrease it if the result looks too zoomed out or unstable. `--depth-fraction` sets the raw planned step as a fraction of VGGT's median scene depth; SEVA normalizes its magnitude, so `--three-baseline` is the main rendered-motion control. The two-image path has zero translation and is unaffected by this displacement setting.

## Outputs

For an input directory named `CASE`, the output root contains:

```text
poses/CASE.json                         estimated camera poses and intrinsics
scenes/CASE/transforms.json             21 SEVA camera frames
scenes/CASE/train_test_split_2.json     or train_test_split_3.json
scenes/CASE/path.json                   path parameters and camera_scale
work_dirs/demo/img2trajvid/multiview_camera_paths/CASE/samples-rgb.mp4
work_dirs/demo/img2trajvid/multiview_camera_paths/CASE/samples-rgb/018.png  # 2 images
work_dirs/demo/img2trajvid/multiview_camera_paths/CASE/samples-rgb/017.png  # 3 images
depth/CASE.png, depth/CASE.npz, depth/CASE.json  # when depth is enabled
runs/CASE.json
```

The path planner uses OpenCV camera coordinates internally and writes SEVA's OpenGL-style camera matrices in `transforms.json`. `path.json` records the chosen source view and the SEVA `camera_scale` value for repeatable rendering.

The camera-path and renderer-interface checks can be run with `python -m unittest discover -s tests -v` in an environment with NumPy and SciPy.
