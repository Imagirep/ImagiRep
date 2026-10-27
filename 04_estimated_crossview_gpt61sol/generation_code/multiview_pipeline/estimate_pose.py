#!/usr/bin/env python3
"""Estimate cameras and scene depth from two or three numbered RGB images."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def numbered_images(case_dir: Path) -> list[Path]:
    paths = sorted(case_dir.glob("image_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    if len(paths) not in (2, 3) or [p.name for p in paths] != [
        f"image_{i}.png" for i in range(1, len(paths) + 1)
    ]:
        raise ValueError("Expected image_1.png through image_2.png or image_3.png")
    return paths


def estimate(case_dir: Path, checkpoint: Path, vggt_root: Path, device: str) -> dict:
    sys.path.insert(0, str(vggt_root.resolve()))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    paths = numbered_images(case_dir)
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append(image.size)
    if len(set(sizes)) != 1:
        raise ValueError("All input images must have the same dimensions")
    width, height = sizes[0]
    if not 0.5 <= height / width <= 2.0:
        raise ValueError("Input aspect ratio needs a preprocessing crop-aware K mapping")

    images = load_and_preprocess_images([str(p) for p in paths], image_resolution=512).to(device)
    model = VGGTOmega().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    del state
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()
    with torch.inference_mode(), autocast:
        prediction = model(images)
        extrinsics, intrinsics = encoding_to_camera(
            prediction["pose_enc"], prediction["images"].shape[-2:]
        )

    extrinsics = extrinsics[0].float().cpu().numpy()
    intrinsics = intrinsics[0].float().cpu().numpy()
    depths = prediction["depth"][0, :, :, :, 0].float().cpu().numpy()
    pre_h, pre_w = prediction["images"].shape[-2:]
    if extrinsics.shape != (len(paths), 3, 4) or intrinsics.shape != (len(paths), 3, 3):
        raise ValueError("Unexpected VGGT camera output shape")

    poses = []
    for path, w2c34, k, depth in zip(paths, extrinsics, intrinsics, depths):
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :] = w2c34
        c2w = np.linalg.inv(w2c)
        k = k.astype(np.float64)
        k[0, :] *= width / pre_w
        k[1, :] *= height / pre_h
        valid_depth = depth[np.isfinite(depth) & (depth > 0)]
        if not len(valid_depth):
            raise ValueError(f"No positive depth predicted for {path.name}")
        poses.append({
            "image": path.name,
            "c2w_opencv": c2w.tolist(),
            "K_original": k.tolist(),
            "median_scene_depth": float(np.median(valid_depth)),
        })
    return {
        "model": "VGGT-Omega-1B-512",
        "case_dir": str(case_dir.resolve()),
        "image_hw": [height, width],
        "poses": poses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vggt-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = estimate(args.case_dir, args.checkpoint, args.vggt_root, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
