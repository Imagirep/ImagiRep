#!/usr/bin/env python3
"""Estimate depth on the final rendered RGB with MoGe-2-L."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image


MODEL_ID = "Ruicheng/moge-2-vitl-normal"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--pose-json", type=Path, required=True)
    parser.add_argument("--view-index", type=int, default=1)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--near", type=float, default=0.5)
    parser.add_argument("--far", type=float, default=8.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not (math.isfinite(args.near) and math.isfinite(args.far) and
            0 < args.near < args.far):
        raise ValueError("Expected 0 < near < far")
    pose_data = json.loads(args.pose_json.read_text())
    if not 1 <= args.view_index <= len(pose_data["poses"]):
        raise ValueError("view-index is outside the input image range")
    k = np.asarray(pose_data["poses"][args.view_index - 1]["K_original"])
    original_width = pose_data["image_hw"][1]
    fov_x = math.degrees(2 * math.atan(original_width / (2 * float(k[0, 0]))))

    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(MODEL_ID).to(args.device).eval()
    image = np.asarray(Image.open(args.rgb).convert("RGB"))
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).to(args.device).float() / 255
    with torch.inference_mode():
        prediction = model.infer(tensor, fov_x=fov_x)
    depth = prediction["depth"].float().cpu().numpy()
    valid = prediction["mask"].bool().cpu().numpy() & np.isfinite(depth) & (depth > 0)
    if depth.shape != image.shape[:2] or not valid.any():
        raise RuntimeError("Invalid depth prediction")
    fraction = np.clip((depth - args.near) / (args.far - args.near), 0, 1)
    color = np.round(matplotlib.colormaps["turbo_r"](fraction)[..., :3] * 255).astype(np.uint8)
    color[~valid] = 0
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color).save(args.output_prefix.with_suffix(".png"))
    np.savez_compressed(args.output_prefix.with_suffix(".npz"),
                        depth_m=depth.astype(np.float32), valid=valid)
    info = {"model": MODEL_ID, "rgb": str(args.rgb), "fov_x_deg": fov_x,
            "near_m": args.near, "far_m": args.far,
            "color_map": "turbo_r; invalid pixels black",
            "valid_fraction": float(valid.mean())}
    args.output_prefix.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(args.output_prefix)


if __name__ == "__main__":
    main()
