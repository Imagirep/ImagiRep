#!/usr/bin/env python3
"""Render a prepared two-view or three-view camera path with SEVA v1.1."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def render(scenes_root: Path, case_id: str, output_root: Path, seva_python: Path,
           seva_repo: Path, model_dir: Path, vae_dir: Path | None = None,
           clip_weights: Path | None = None,
           short_side: int = 384) -> dict:
    scene = scenes_root / case_id
    path_file = scene / "path.json"
    transforms_file = scene / "transforms.json"
    if not path_file.is_file() or not transforms_file.is_file():
        raise FileNotFoundError("Run plan_path.py before rendering")
    if not (model_dir / "modelv1.1.safetensors").is_file():
        raise FileNotFoundError(model_dir / "modelv1.1.safetensors")
    if clip_weights is not None and not clip_weights.is_file():
        raise FileNotFoundError(clip_weights)
    if not (seva_repo / "demo.py").is_file():
        raise FileNotFoundError(seva_repo / "demo.py")
    path = json.loads(path_file.read_text())
    n_inputs = len(json.loads((scene / f"train_test_split_{2 if path['mode'] == 'rotation_midpoint' else 3}.json").read_text())["train_ids"])
    save_subdir = "multiview_camera_paths"
    command = [
        str(seva_python), str(seva_repo / "demo.py"),
        "--data_path", str(scenes_root.resolve()),
        "--data_items", case_id,
        "--task", "img2trajvid",
        "--num_inputs", str(n_inputs),
        "--pretrained_model_name_or_path", str(model_dir.resolve()),
        "--weight_name", "modelv1.1.safetensors",
        "--version", "1.1",
        "--cfg", "3.0,2.0",
        "--use_traj_prior", "True",
        "--chunk_strategy", "interp-gt",
        "--camera_scale", str(path["camera_scale"]),
        "--L_short", str(short_side),
        "--video_save_fps", "10",
        "--save_subdir", save_subdir,
    ]
    env = os.environ.copy()
    if vae_dir is not None:
        env["SEVA_VAE_ROOT"] = str(vae_dir.resolve())
    if clip_weights is not None:
        env["SEVA_OPEN_CLIP_WEIGHTS"] = str(clip_weights.resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=output_root, env=env, check=True)
    rendered = output_root / "work_dirs/demo/img2trajvid" / save_subdir / case_id
    video = rendered / "samples-rgb.mp4"
    final_rgb = rendered / "samples-rgb" / f"{path['num_targets'] - 1:03}.png"
    if not video.is_file() or not final_rgb.is_file():
        raise RuntimeError(f"SEVA output not found in {rendered}")
    result = {"video": str(video), "final_rgb": str(final_rgb),
              "camera_scale": path["camera_scale"], "num_inputs": n_inputs}
    (rendered / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seva-python", type=Path, required=True)
    parser.add_argument("--seva-repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vae-dir", type=Path)
    parser.add_argument("--clip-weights", type=Path)
    parser.add_argument("--short-side", type=int, default=384)
    args = parser.parse_args()
    result = render(args.scenes_root, args.case_id, args.output_root,
                    args.seva_python, args.seva_repo, args.model_dir,
                    args.vae_dir, args.clip_weights, args.short_side)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
