#!/usr/bin/env python3
"""Create a smooth SEVA camera path from estimated input cameras."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Degenerate direction vector")
    return vector / norm


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def horizontal(vector: np.ndarray, up: np.ndarray) -> np.ndarray:
    return unit(vector - up * np.dot(vector, up))


def estimate_up(poses: list[np.ndarray]) -> np.ndarray:
    rights = np.stack([pose[:3, 0] for pose in poses])
    up = np.linalg.svd(rights, full_matrices=False)[2][-1]
    image_ups = np.mean([-pose[:3, 1] for pose in poses], axis=0)
    return unit(up if np.dot(up, image_ups) >= 0 else -up)


def select_center_view(poses: list[np.ndarray]) -> int:
    rotations = [Rotation.from_matrix(pose[:3, :3]) for pose in poses]
    gaps = np.array([
        [float((a.inv() * b).magnitude()) for b in rotations] for a in rotations
    ])
    return int(np.argmin(gaps.sum(axis=1)))


def frame(image: Path | None, pose_cv: np.ndarray, k: np.ndarray, width: int, height: int) -> dict:
    return {
        "file_path": str(image) if image is not None else None,
        "transform_matrix": (pose_cv @ CV_TO_GL).tolist(),
        "fl_x": float(k[0, 0]),
        "fl_y": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
        "w": width,
        "h": height,
    }


def normalized_scale(frames: list[dict], desired_baseline: float) -> float:
    """Choose SEVA camera_scale for a desired post-centering endpoint displacement."""
    centers = np.stack([np.asarray(f["transform_matrix"])[:3, 3] for f in frames])
    first_radius = float(np.linalg.norm(centers[0] - centers.mean(axis=0)))
    endpoint_baseline = float(np.linalg.norm(centers[-1] - centers[0]))
    if first_radius < 1e-8 or endpoint_baseline < 1e-8:
        raise ValueError("Three-view path has no usable camera displacement")
    return desired_baseline * first_radius / endpoint_baseline


def plan(data: dict, depth_fraction: float = 0.15, lateral_ratio: float = 0.0,
         three_baseline: float = 0.55, start_view: int | None = None) -> tuple[dict, dict, dict]:
    observed = data["poses"]
    n = len(observed)
    if n not in (2, 3):
        raise ValueError("Expected two or three input cameras")
    if not (np.isfinite(depth_fraction) and depth_fraction > 0):
        raise ValueError("depth_fraction must be positive and finite")
    if not (np.isfinite(three_baseline) and three_baseline > 0):
        raise ValueError("three_baseline must be positive and finite")
    if not np.isfinite(lateral_ratio):
        raise ValueError("lateral_ratio must be finite")
    if start_view is not None and not 1 <= start_view <= n:
        raise ValueError("start_view is outside the input image range")

    height, width = map(int, data["image_hw"])
    case_dir = Path(data["case_dir"])
    poses = [np.asarray(item["c2w_opencv"], dtype=np.float64).copy() for item in observed]
    intrinsics = [np.asarray(item["K_original"], dtype=np.float64) for item in observed]
    if any(p.shape != (4, 4) or not np.isfinite(p).all() for p in poses):
        raise ValueError("Invalid input camera matrix")
    if any(k.shape != (3, 3) or not np.isfinite(k).all() for k in intrinsics):
        raise ValueError("Invalid input intrinsics")
    center = np.mean([pose[:3, 3] for pose in poses], axis=0)
    for pose in poses:
        pose[:3, 3] = center
    target_k = np.mean(intrinsics, axis=0)
    frames = [frame(case_dir / item["image"], pose, k, width, height)
              for item, pose, k in zip(observed, poses, intrinsics)]

    if n == 2:
        if start_view not in (None, 1):
            raise ValueError("Two-view path always starts at image_1.png")
        rotations = Rotation.from_matrix(np.stack([p[:3, :3] for p in poses]))
        slerp = Slerp([0.0, 1.0], rotations)
        num_targets = 19
        for i in range(num_targets):
            t = smoothstep((i + 1) / num_targets) * 0.5
            pose = np.eye(4)
            pose[:3, :3] = slerp([t]).as_matrix()[0]
            pose[:3, 3] = center
            frames.append(frame(None, pose, target_k, width, height))
        camera_scale = 2.0
        path = {"mode": "rotation_midpoint", "start_view": 1,
                "num_targets": num_targets, "camera_scale": camera_scale}
    else:
        source_index = (start_view - 1) if start_view is not None else select_center_view(poses)
        source = poses[source_index]
        up = estimate_up(poses)
        forward = horizontal(source[:3, 2], up)
        right = horizontal(np.cross(forward, up), up)
        median_depth = float(observed[source_index]["median_scene_depth"])
        if not np.isfinite(median_depth) or median_depth <= 0:
            raise ValueError("Source depth must be positive and finite")
        step = median_depth * depth_fraction
        destination = center + step * (-forward + lateral_ratio * right)
        anchor = center + median_depth * forward
        destination_forward = horizontal(anchor - destination, up)
        yaw = float(np.arctan2(
            np.dot(np.cross(forward, destination_forward), up),
            np.dot(forward, destination_forward),
        ))
        target_rotation = Rotation.from_rotvec(up * yaw).as_matrix() @ source[:3, :3]
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(
            np.stack([source[:3, :3], target_rotation])
        ))
        num_targets = 18
        for i in range(num_targets):
            t = smoothstep((i + 1) / num_targets)
            pose = np.eye(4)
            pose[:3, :3] = slerp([t]).as_matrix()[0]
            pose[:3, 3] = center + t * (destination - center)
            frames.append(frame(None, pose, target_k, width, height))
        camera_scale = normalized_scale(frames, three_baseline)
        path = {"mode": "horizontal_dolly_back", "start_view": source_index + 1,
                "num_targets": num_targets, "depth_fraction": depth_fraction,
                "lateral_ratio": lateral_ratio, "three_baseline": three_baseline,
                "camera_scale": camera_scale, "estimated_up": up.tolist(),
                "destination_c2w_opencv": pose.tolist(),
                "motion_scene_units": float(np.linalg.norm(destination - center))}

    transforms = {"orientation_override": "none", "frames": frames}
    split = {"train_ids": list(range(n)), "test_ids": list(range(n, len(frames)))}
    return transforms, split, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-fraction", type=float, default=0.15)
    parser.add_argument("--lateral-ratio", type=float, default=0.0)
    parser.add_argument("--three-baseline", type=float, default=0.55)
    parser.add_argument("--start-view", type=int)
    args = parser.parse_args()
    transforms, split, path = plan(json.loads(args.pose_json.read_text()),
                                   args.depth_fraction, args.lateral_ratio,
                                   args.three_baseline, args.start_view)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("transforms.json", transforms),
                       (f"train_test_split_{len(split['train_ids'])}.json", split),
                       ("path.json", path)):
        (args.output_dir / name).write_text(json.dumps(data, indent=2) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
