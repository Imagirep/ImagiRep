"""Offline only: audit final-pose modalities and publish input/prompt previews."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import run_endpoint_geometry as run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RELEASE = ROOT / "benchmark_release"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_pose(actual, expected):
    return (actual["viewpoint_id"] == expected["viewpoint_id"]
            and abs(math.remainder(actual["heading_rad"] - expected["heading_rad"], 2 * math.pi)) < 1e-6
            and abs(actual["elevation_rad"] - expected["elevation_rad"]) < 1e-6
            and abs(actual["vfov_degrees"] - expected["vfov_degrees"]) < 1e-6
            and actual["width"] == expected["width"] and actual["height"] == expected["height"])


def main():
    rows = run.base.read_jsonl(HERE / "source_manifest.jsonl")
    index = {r["question_id"]: r for r in run.base.read_jsonl(RELEASE / "fo_selected171_modalities.jsonl")}
    published = json.loads((ROOT / "docs/experiments/fo_selected171_modality_completion_20260916/asset_sha256.json").read_text())
    hashes, final_records, errors = {}, {}, []
    for row in rows:
        qid = row["question_id"]
        canonical = index[qid]
        final = canonical["final_view"]
        pose = final["pose"]
        if (row["question"] != canonical["question"] or row["original_answer_choices"] != canonical["answer_choices"]
                or row["original_gold_answer"] != canonical["oracle_answer"]):
            errors.append(f"{qid}: frozen content differs from canonical")
        expected_kind = "lookback" if row["show_lookback"] else "endpoint"
        if canonical["final_image_kind"] != expected_kind:
            errors.append(f"{qid}: canonical endpoint kind mismatch")
        pair = run.rgb_pair(row)
        row["rgb_pair_sha256"] = [digest(i["path"]) for i in pair]
        if row["rgb_pair_sha256"] != [digest(RELEASE / canonical["frames"][0]["rgb"]), digest(RELEASE / final["rgb"])]:
            errors.append(f"{qid}: wrong start/final RGB")
        for image in pair:
            with Image.open(image["path"]) as im:
                im.verify()
            hashes[image["path"]] = digest(image["path"])
        metadata_path = RELEASE / final["manifest"]
        metadata = json.loads(metadata_path.read_text())
        hashes[str(metadata_path)] = digest(metadata_path)
        if metadata.get("rgb_sha256") != row["rgb_pair_sha256"][-1]:
            errors.append(f"{qid}: final aligned metadata refers to different RGB bytes")
        if hashes[str(metadata_path)] != published[final["manifest"]]:
            errors.append(f"{qid}: final metadata differs from verified publication")
        visual = metadata["visual"]
        visual_pose = visual.get("pose")
        if visual_pose is None:
            # Preserved full-resolution legacy frames store pose under render
            # and projection dimensions in the normals record.
            visual_pose = {**visual["render"], "width": visual["surface_normals"]["resolution"][0],
                           "height": visual["surface_normals"]["resolution"][1]}
        for label, actual in (("frame", metadata["pose"]), ("depth", metadata["depth"]["pose"]), ("visual", visual_pose)):
            if not check_pose(actual, pose):
                errors.append(f"{qid}: {label} not aligned to actual final camera")
        stats = metadata["depth"]["depth"]
        near, far = stats["color_scale_p02_m"], stats["color_scale_p98_m"]
        if not (stats["available"] and 0 <= near < far):
            errors.append(f"{qid}: unusable depth/color scale")
        row["endpoint_modalities"] = {}
        for kind in ("depth", "surface_normals", "panorama"):
            relative = final[kind]
            path = RELEASE / relative
            checksum = digest(path)
            hashes[str(path)] = checksum
            if checksum != published[relative]:
                errors.append(f"{qid}: {kind} differs from verified publication")
            with Image.open(path) as im:
                expected_size = (1024, 512) if kind == "panorama" else (pose["width"], pose["height"])
                if im.size != expected_size:
                    errors.append(f"{qid}: wrong {kind} dimensions")
                im.verify()
            row["endpoint_modalities"][kind] = {"kind": kind, "path": str(path), "sha256": checksum,
                                               "final_image_kind": expected_kind, "pose": pose,
                                               "aligned_manifest": str(metadata_path)}
        row["endpoint_modalities"]["depth"]["color_scale"] = {
            "near_m": near, "far_m": far, "colormap": "reversed OpenCV TURBO",
            "normalization": "2nd to 98th percentile per image; clipped", "black": "invalid depth",
            "measurement": "camera-forward z-depth, not radial range"}
        row["endpoint_modalities"]["surface_normals"]["encoding"] = {
            "rgb_axes": ["camera-right", "camera-up", "camera-forward"],
            "channel_formula": "uint8(clip((component + 1) * 127.5, 0, 255))", "black": "missing geometry"}
        row["endpoint_modalities"]["panorama"]["orientation"] = {
            "center": "front", "quarter_width": "left", "three_quarter_width": "right",
            "edges": "behind; horizontal wrap", "pose_alignment": "source RGB camera-centered spherical rotation"}
        if str(metadata_path) not in final_records:
            raw_path = RELEASE / final["depth_raw"]
            raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
            hashes[str(raw_path)] = digest(raw_path)
            if hashes[str(raw_path)] != published[final["depth_raw"]] or raw.dtype != np.uint16 or raw.shape != (pose["height"], pose["width"]):
                errors.append(f"{qid}: raw depth encoding/dimensions/hash invalid")
            normalized = np.clip((raw.astype(np.float32) / 4000.0 - near) / (far - near), 0, 1)
            values = (255 - normalized * 255).astype(np.uint8)
            values[raw == 0] = 0
            expected_color = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
            expected_color[raw == 0] = 0
            actual_color = cv2.imread(row["endpoint_modalities"]["depth"]["path"])
            if not np.array_equal(expected_color, actual_color):
                errors.append(f"{qid}: colorized depth does not match raw depth and documented scale")
            panorama = metadata["visual"]["panorama"]
            # Legacy panorama metadata omits source_rgb; its published hash,
            # aligned visual pose and explicit aligned RGB hash remain checked.
            if panorama.get("source_rgb") and digest(Path(panorama["source_rgb"])) != row["rgb_pair_sha256"][-1]:
                errors.append(f"{qid}: panorama centered using wrong final RGB")
            final_records[str(metadata_path)] = {"pose": pose, "color_scale": [near, far],
                                                "panorama_center_match_score": panorama["center_match_score"],
                                                "panorama_source_rgb_explicit": bool(panorama.get("source_rgb"))}
    if len(rows) != 171 or len({r["trajectory_id"] for r in rows}) != 50:
        errors.append("Expected frozen171 and 50 trajectories")
    report = {"status": "failed" if errors else "passed", "execution_status": "prepared_not_run", "model_calls": 0,
              "questions": len(rows), "trajectories": len({r["trajectory_id"] for r in rows}),
              "planned_cases": len(rows) * 3, "conditions": list(run.CONDITIONS),
              "final_image_kinds": dict(Counter(run.rgb_pair(r)[-1]["kind"] for r in rows)),
              "verified_final_camera_records": final_records, "verified_input_sha256": hashes,
              "source_manifest_sha256": digest(HERE / "source_manifest.jsonl"), "errors": errors}
    if errors:
        run.base.atomic_json(HERE / "preflight.json", report)
        print({k: v for k, v in report.items() if k not in {"verified_input_sha256", "verified_final_camera_records"}})
        raise SystemExit("Offline preparation failed; no model calls made")
    (HERE / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    report["frozen_config_sha256"] = {str(p): digest(p) for p in
                                      (HERE / "manifest.jsonl", HERE / "runner_snapshot.py", HERE / "run_endpoint_geometry.py")}
    run.base.atomic_json(HERE / "preflight.json", report)
    previews = [{"question_id": r["question_id"], "final_image_kind": run.rgb_pair(r)[-1]["kind"],
                 "system_prompt": run.base.SYSTEM_PROMPT,
                 "settings": [{"condition": c, "images": run.selected_images(r, c),
                               "image_prompts": run.image_prompts(r, c),
                               "question_prompt": run.base.question_text(r)} for c in run.CONDITIONS]} for r in rows]
    run.base.atomic_json(HERE / "prompt_previews.json", previews)
    examples = [next(r for r in rows if r["show_lookback"] == flag) for flag in (False, True)]
    lines = ["# Offline prompt examples", "", "Prepared only. No inference/API calls.", ""]
    for row in examples:
        lines += [f"## {row['question_id']} ({run.rgb_pair(row)[-1]['kind']})", ""]
        for condition in run.CONDITIONS:
            lines += [f"### {condition}", "", "```text", "SYSTEM: " + run.base.SYSTEM_PROMPT]
            for image, caption in zip(run.selected_images(row, condition), run.image_prompts(row, condition)):
                lines += [caption, f"[PNG: {image['path']}]"]
            lines += [run.base.question_text(row), "```", ""]
    (HERE / "PROMPT_EXAMPLES.md").write_text("\n".join(lines), encoding="utf-8")
    print({k: v for k, v in report.items() if k not in {"verified_input_sha256", "verified_final_camera_records", "frozen_config_sha256"}})
    print("Unique final camera records:", len(final_records))


if __name__ == "__main__":
    main()
