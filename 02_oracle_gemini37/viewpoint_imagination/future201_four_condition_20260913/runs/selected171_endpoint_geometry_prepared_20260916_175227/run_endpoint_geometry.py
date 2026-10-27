"""Prepare/preview endpoint geometry inputs; model calls require --execute."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

from google.genai import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
spec = importlib.util.spec_from_file_location("rgb_runner", HERE / "runner_snapshot.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

CONDITIONS = ("start_final_depth", "start_final_surface_normals", "start_final_panorama")
CONDITION_MODALITY = dict(zip(CONDITIONS, ("depth", "surface_normals", "panorama")))
PROMPT_VERSION = "final_state_balanced_v3_plus_endpoint_geometry_v1"
MODALITY_PROMPTS = {
    "depth": (
        "ENDPOINT DEPTH MAP (same camera pose and field of view as FINAL VIEW):\n"
        "Colors encode camera-forward distance (z-depth), not object color. "
        "Red/orange is nearer, green is intermediate, and blue/purple is farther; "
        "black means missing depth, not a nearby surface. "
        "This image's color scale runs from {near_m:.3f} m (near/red) to {far_m:.3f} m "
        "(far/purple), with values outside this range clipped. "
        "Combine this map with START and FINAL VIEW RGB to reason about the final state."
    ),
    "surface_normals": (
        "ENDPOINT SURFACE NORMAL MAP (same camera pose and field of view as FINAL VIEW):\n"
        "Colors encode surface orientation, not object color or distance. "
        "RGB channels encode normal components along camera-right (R), camera-up (G), "
        "and camera-forward (B): 0 = -1, about 128 = 0, and 255 = +1. "
        "Higher R/G/B means a stronger rightward/upward/forward component; "
        "a surface facing the camera has low B. Black means missing geometry. "
        "Combine this map with START and FINAL VIEW RGB to reason about the final state."
    ),
    "panorama": (
        "ENDPOINT PANORAMA (360-degree RGB at the FINAL VIEW position):\n"
        "The panorama is centered on the final camera's facing direction. "
        "At mid-height, the center is forward, one-quarter of the width is left, "
        "three-quarters is right, and the two joined edges are behind. "
        "Top/bottom look up/down; colors are natural RGB, not geometric labels. "
        "This is one position, not a sequence of movement frames. "
        "Combine it with START and FINAL VIEW RGB to reason about the final state."
    ),
}


def modality(condition):
    try:
        return CONDITION_MODALITY[condition]
    except KeyError:
        raise ValueError(f"Unsupported condition: {condition}") from None


def rgb_pair(row):
    final_kind = "lookback" if row["show_lookback"] else "endpoint"
    start = [i for i in row["evidence_images"] if i["kind"] == "start"]
    final = [i for i in row["evidence_images"] if i["kind"] == final_kind]
    if len(start) != 1 or len(final) != 1:
        raise ValueError(f"Ambiguous RGB pair: {row['question_id']}")
    return start + final


def modality_prompt(row, condition):
    kind = modality(condition)
    if kind == "depth":
        scale = row["endpoint_modalities"][kind]["color_scale"]
        return MODALITY_PROMPTS[kind].format(near_m=scale["near_m"], far_m=scale["far_m"])
    return MODALITY_PROMPTS[kind]


def selected_images(row, condition):
    kind = modality(condition)
    return rgb_pair(row) + [{"kind": kind, "label": f"Endpoint {kind}",
                            "path": row["endpoint_modalities"][kind]["path"]}]


def image_prompts(row, condition):
    return ["Start:", base.FINAL_VIEW_PROMPT, modality_prompt(row, condition)]


def gemini_parts(row, condition):
    parts = []
    for image, caption in zip(selected_images(row, condition), image_prompts(row, condition), strict=True):
        parts.extend([types.Part.from_text(text=caption + "\n"),
                      types.Part.from_bytes(data=Path(image["path"]).read_bytes(), mime_type="image/png")])
    parts.append(types.Part.from_text(text=base.question_text(row)))
    return parts


def input_signature(row, condition, model):
    kind = modality(condition)
    signature = {"question": row["question"], "choices": row["answer_choices"],
                 "gold": row["gold_answer"], "show_lookback": row["show_lookback"],
                 "rgb_sha256": row["rgb_pair_sha256"], "modality": kind,
                 "asset_sha256": row["endpoint_modalities"][kind]["sha256"],
                 "system_prompt": base.SYSTEM_PROMPT, "image_prompts": image_prompts(row, condition),
                 "model": model, "temperature": 0, "thinking": "HIGH", "max_output_tokens": 8192,
                 "prompt_version": PROMPT_VERSION}
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def verify_prepared_inputs():
    report = json.loads((HERE / "preflight.json").read_text(encoding="utf-8"))
    if report["status"] != "passed":
        raise ValueError("Prepared input preflight did not pass")
    expected = {**report["verified_input_sha256"], **report["frozen_config_sha256"]}
    for path, checksum in expected.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Prepared input or configuration changed: {path}; re-audit before testing")


def run_one(row, condition, args):
    if args.provider != "gemini":
        raise ValueError("This prepared protocol supports Gemini only")
    output = args.output_dir / "per_sample" / condition / f"{row['question_id']}.json"
    signature = input_signature(row, condition, args.model)
    if args.resume and output.is_file():
        old = json.loads(output.read_text(encoding="utf-8"))
        if (old.get("input_signature") != signature or old.get("prediction") not in row["answer_choices"]
                or old.get("correct") != (old.get("prediction") == row["gold_answer"])):
            raise ValueError(f"Stale/invalid resumed result: {row['question_id']} {condition}")
        return {"status": "skipped"}
    prediction, usage, raw = base.generate(args, row, condition)
    images = selected_images(row, condition)
    result = {**row, "condition": condition, "provider": args.provider, "model": args.model,
              "system_prompt": base.SYSTEM_PROMPT, "prompt_version": PROMPT_VERSION,
              "image_prompts": image_prompts(row, condition), "reasoning_effort": "high", "temperature": 0,
              "prediction": prediction, "correct": prediction == row["gold_answer"], "image_count": 3,
              "evidence_kinds": [i["kind"] for i in images],
              "endpoint_modality": row["endpoint_modalities"][modality(condition)],
              "input_signature": signature, "usage_metadata": usage, "raw_response": raw,
              "completed_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    base.atomic_json(output, result)
    return {"status": "ok"}


base.CONDITIONS = CONDITIONS
base.PROMPT_VERSION = PROMPT_VERSION
base.selected_images = selected_images
base.image_prompts = image_prompts
base.gemini_parts = gemini_parts
base.run_one = run_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Explicitly allow paid model calls; default is offline preview")
    parser.add_argument("--preview-id", help="Question ID for offline preview (does not run inference)")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--model", default="gemini-3.7-flash")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()
    verify_prepared_inputs()
    rows = base.read_jsonl(HERE / "manifest.jsonl")
    if len(rows) != 171:
        raise ValueError("Expected frozen171")
    if not args.execute:
        selected = [r for r in rows if not args.preview_id or r["question_id"] == args.preview_id]
        if not selected:
            raise ValueError(f"Unknown preview ID: {args.preview_id}")
        print(json.dumps({"status": "prepared_not_run", "model_calls": 0, "questions": len(rows),
                          "conditions": args.conditions, "planned_cases": len(rows) * len(args.conditions),
                          "question_id": selected[0]["question_id"], "system_prompt": base.SYSTEM_PROMPT,
                          "previews": [{"condition": c, "images": selected_images(selected[0], c),
                                        "image_prompts": image_prompts(selected[0], c),
                                        "question_prompt": base.question_text(selected[0])} for c in args.conditions]},
                         ensure_ascii=False, indent=2))
        return
    if args.preview_id:
        parser.error("--preview-id is for offline preview only; do not combine with --execute")
    sys.argv = [str(HERE / "runner_snapshot.py"), "--provider", "gemini", "--model", args.model,
                "--env-file", str(args.env_file), "--manifest", str(HERE / "manifest.jsonl"),
                "--expected-count", "171", "--conditions", *args.conditions,
                "--output-dir", str(args.output_dir), "--workers", str(args.workers),
                "--retries", str(args.retries), "--resume"]
    base.main()


if __name__ == "__main__":
    main()
