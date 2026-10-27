"""Only the requested start + endpoint + one topdown/text/video experiments."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

from google.genai import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
spec = importlib.util.spec_from_file_location("rgb_runner", HERE / "runner_snapshot.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

CONDITIONS = ("start_final_topdown", "start_final_text", "start_final_video")
VIDEO_FPS = 12
MODALITY_PROMPTS = {
    "topdown": (
        "ADDITIONAL TOP-DOWN MAP:\n"
        "Numbered markers show the start (0) and successive movement positions. "
        "Lines show the path; the final arrow shows the final facing direction. "
        "Use this map together with the START and FINAL VIEW RGB images."
    ),
    "text": (
        "ADDITIONAL TEXT TRAJECTORY:\n"
        "The description gives the movement and camera poses from start to final state. "
        "Headings are in degrees: 0 faces +Y; positive rotation is clockwise toward +X. "
        "Use it together with the START and FINAL VIEW RGB images."
    ),
    "video": (
        "ADDITIONAL TRAJECTORY VIDEO:\n"
        "The video shows the movement from start to the final state, including camera turns. "
        "Translation uses discrete connectivity hops. "
        "Use it together with the START and FINAL VIEW RGB images."
    ),
}


def modality(condition):
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")
    return condition.rsplit("_", 1)[-1]


def rgb_pair(row):
    start = [i for i in row["evidence_images"] if i["kind"] == "start"]
    final_kind = "lookback" if row["show_lookback"] else "endpoint"
    final = [i for i in row["evidence_images"] if i["kind"] == final_kind]
    if len(start) != 1 or len(final) != 1:
        raise ValueError(f"Ambiguous RGB pair: {row['question_id']}")
    return start + final


def selected_images(row, condition):
    images = rgb_pair(row)
    if modality(condition) == "topdown":
        images += [{"kind": "topdown", "label": "Additional top-down map",
                    "path": row["additional_modalities"]["topdown"]["path"]}]
    return images


def image_prompts(row, condition):
    captions = ["Start:", base.FINAL_VIEW_PROMPT]
    if modality(condition) == "topdown":
        captions.append(MODALITY_PROMPTS["topdown"])
    return captions


def gemini_parts(row, condition):
    parts = []
    for image, caption in zip(selected_images(row, condition), image_prompts(row, condition), strict=True):
        parts += [types.Part.from_text(text=caption + "\n"),
                  types.Part.from_bytes(data=Path(image["path"]).read_bytes(), mime_type="image/png")]
    kind = modality(condition)
    asset = row["additional_modalities"][kind]
    if kind == "text":
        parts.append(types.Part.from_text(text=MODALITY_PROMPTS[kind] + "\n" +
                                         Path(asset["path"]).read_text(encoding="utf-8")))
    elif kind == "video":
        parts.append(types.Part.from_text(text=MODALITY_PROMPTS[kind] + "\n"))
        parts.append(types.Part(inline_data=types.Blob(mime_type="video/mp4", data=Path(asset["path"]).read_bytes()),
                               video_metadata=types.VideoMetadata(fps=VIDEO_FPS)))
    parts.append(types.Part.from_text(text=base.question_text(row)))
    return parts


def input_signature(row, condition, model):
    kind = modality(condition)
    signature = {"question": row["question"], "choices": row["answer_choices"],
                 "gold": row["gold_answer"], "rgb_sha256": row["rgb_pair_sha256"],
                 "final_kind": rgb_pair(row)[-1]["kind"], "modality": kind,
                 "asset_sha256": row["additional_modalities"][kind]["sha256"],
                 "system": base.SYSTEM_PROMPT, "captions": image_prompts(row, condition),
                 "modality_prompt": MODALITY_PROMPTS[kind], "model": model,
                 "temperature": 0, "thinking": "HIGH", "max_output_tokens": 8192,
                 "video_sampling_fps": VIDEO_FPS if kind == "video" else None}
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def run_one(row, condition, args):
    if args.provider != "gemini":
        raise ValueError("This narrowly scoped run supports Gemini only")
    output = args.output_dir / "per_sample" / condition / f"{row['question_id']}.json"
    signature = input_signature(row, condition, args.model)
    if args.resume and output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("input_signature") != signature or existing.get("prediction") not in row["answer_choices"]:
            raise ValueError(f"Stale or invalid resumed result: {row['question_id']} {condition}")
        return {"status": "skipped"}
    prediction, usage, raw = base.generate(args, row, condition)
    images = selected_images(row, condition)
    kind = modality(condition)
    result = {**row, "condition": condition, "provider": args.provider, "model": args.model,
              "system_prompt": base.SYSTEM_PROMPT, "prompt_version": base.PROMPT_VERSION,
              "image_prompts": image_prompts(row, condition), "modality_prompt": MODALITY_PROMPTS[kind],
              "reasoning_effort": "high", "temperature": 0, "prediction": prediction,
              "correct": prediction == row["gold_answer"], "image_count": len(images),
              "evidence_kinds": [i["kind"] for i in images],
              "additional_modality": row["additional_modalities"][kind], "input_signature": signature,
              "video_sampling_fps": VIDEO_FPS if kind == "video" else None,
              "usage_metadata": usage, "raw_response": raw,
              "completed_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    base.atomic_json(output, result)
    return {"status": "ok"}


base.CONDITIONS = CONDITIONS
base.PROMPT_VERSION = "final_state_balanced_v3_plus_single_modality_v1"
base.selected_images = selected_images
base.image_prompts = image_prompts
base.gemini_parts = gemini_parts
base.run_one = run_one

if __name__ == "__main__":
    base.main()
