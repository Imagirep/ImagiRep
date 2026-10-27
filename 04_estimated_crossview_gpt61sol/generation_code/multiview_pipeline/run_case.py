#!/usr/bin/env python3
"""Run VGGT Omega, camera planning, SEVA, and optional MoGe depth for one case."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from estimate_pose import estimate
from plan_path import plan
from render_seva import render


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vggt-root", type=Path, required=True)
    parser.add_argument("--vggt-checkpoint", type=Path, required=True)
    parser.add_argument("--seva-python", type=Path, required=True)
    parser.add_argument("--seva-repo", type=Path, required=True)
    parser.add_argument("--seva-model-dir", type=Path, required=True)
    parser.add_argument("--seva-vae-dir", type=Path)
    parser.add_argument("--seva-clip-weights", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--short-side", type=int, default=384)
    parser.add_argument("--depth-fraction", type=float, default=0.15)
    parser.add_argument("--lateral-ratio", type=float, default=0.0)
    parser.add_argument("--three-baseline", type=float, default=0.55)
    parser.add_argument("--start-view", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--depth-python", type=Path)
    parser.add_argument("--depth-near", type=float, default=0.5)
    parser.add_argument("--depth-far", type=float, default=8.0)
    args = parser.parse_args()
    if args.plan_only and args.depth_python:
        parser.error("--depth-python requires rendering")
    case_id = args.case_dir.name
    pose_path = args.output_root / "poses" / f"{case_id}.json"
    scene_dir = args.output_root / "scenes" / case_id

    poses = estimate(args.case_dir, args.vggt_checkpoint, args.vggt_root, args.device)
    save_json(pose_path, poses)
    transforms, split, path = plan(
        poses, args.depth_fraction, args.lateral_ratio,
        args.three_baseline, args.start_view,
    )
    save_json(scene_dir / "transforms.json", transforms)
    save_json(scene_dir / f"train_test_split_{len(split['train_ids'])}.json", split)
    save_json(scene_dir / "path.json", path)
    result = {"case_id": case_id, "pose_json": str(pose_path),
              "scene_dir": str(scene_dir), "path": path}
    if not args.plan_only:
        rendered = render(args.output_root / "scenes", case_id, args.output_root,
                          args.seva_python, args.seva_repo, args.seva_model_dir,
                          args.seva_vae_dir, args.seva_clip_weights, args.short_side)
        result["render"] = rendered
        if args.depth_python:
            depth_prefix = args.output_root / "depth" / case_id
            subprocess.run([
                str(args.depth_python), str(Path(__file__).with_name("estimate_depth.py")),
                "--rgb", rendered["final_rgb"], "--pose-json", str(pose_path),
                "--view-index", str(path["start_view"]),
                "--output-prefix", str(depth_prefix),
                "--near", str(args.depth_near), "--far", str(args.depth_far),
                "--device", args.device,
            ], check=True)
            result["depth_prefix"] = str(depth_prefix)
    save_json(args.output_root / "runs" / f"{case_id}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
