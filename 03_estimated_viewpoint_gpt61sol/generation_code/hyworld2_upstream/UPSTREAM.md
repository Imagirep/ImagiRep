# HY-World 2.0 (third-party world model)

The Viewpoint Imagination estimates use HY-World 2.0 at a fixed upstream commit, with three small local changes.
The upstream code is not redistributed here; fetch it from the official repository.

- Repository: https://github.com/Tencent-Hunyuan/HY-World-2.0
- Commit: `df9988efb87bfc0f4947eb3889411cf957478b06` (2026-08-13, "Fix citation")
- Weights: https://huggingface.co/tencent/HY-World-2.0 (the exact files are listed in `../hyworld2_deployment/metadata/weight_manifest.json`)
- License: Tencent HY-World 2.0 Community License Agreement (see the upstream repository)

## Local changes

`local_changes.patch` (5 lines added, 2 removed) touches:

- `hyworld2/panogen/hunyuan_image_3/siglip2.py`
- `hyworld2/worldgen/models/worldstereo_wrapper.py`
- `hyworld2/worldgen/traj_generate.py`

Apply it from the repository root:

```bash
git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git
cd HY-World-2.0
git checkout df9988efb87bfc0f4947eb3889411cf957478b06
git submodule update --init --recursive
git apply /path/to/local_changes.patch
```

Trajectory planning, scale calibration and rendering of the representations use the code in `../trajectory_planner/`.
