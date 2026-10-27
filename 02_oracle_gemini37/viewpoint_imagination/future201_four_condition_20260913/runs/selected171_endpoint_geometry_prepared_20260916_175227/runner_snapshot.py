#!/usr/bin/env python3
"""Run matched Future201 text/start/final/trajectory evaluations."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from openai import OpenAI


CONDITIONS = ("text_only", "start_view", "start_final", "start_full_trajectory")
SYSTEM_PROMPT = (
    "Solve the multiple-choice embodied spatial reasoning question using only the provided "
    "evidence. Images are labeled and chronological. Return only A, B, C, or D."
)
PROMPT_VERSION = "final_state_balanced_v3"
FINAL_VIEW_PROMPT = (
    "FINAL VIEW (after completing the movement):\n"
    "This frame shows your final position and facing direction after completing the "
    "movement in the question. Use this frame to ground your reasoning about the final "
    "state, and combine it with earlier frames to understand the movement and spatial "
    "relationships."
)
_thread_local = threading.local()
ENDPOINT_IS_LOOKBACK_PATTERN = re.compile(r"\bturn\s+around\b", re.IGNORECASE)
SEPARATE_LOOKBACK_PATTERN = re.compile(
    r"\bthen\s+turn\s+back\s+to\s+face\s+where\s+i\s+come\s+from\b",
    re.IGNORECASE,
)


def endpoint_is_lookback(row: dict[str, Any]) -> bool:
    return bool(
        row.get("endpoint_is_lookback")
        or ENDPOINT_IS_LOOKBACK_PATTERN.search(str(row.get("question", "")))
    )


def show_separate_lookback(row: dict[str, Any]) -> bool:
    # A frozen manifest already contains the confirmed per-question image choice.
    if "show_lookback" in row:
        return bool(row["show_lookback"])
    if endpoint_is_lookback(row):
        return False
    if row.get("requires_separate_lookback") or SEPARATE_LOOKBACK_PATTERN.search(
        str(row.get("question", ""))
    ):
        return True
    return row["question_set"] == "lookback"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_env(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def selected_images(row: dict[str, Any], condition: str) -> list[dict[str, str]]:
    images = row["evidence_images"]
    start = [image for image in images if image["kind"] == "start"]
    endpoint = [image for image in images if image["kind"] == "endpoint"]
    lookback = [image for image in images if image["kind"] == "lookback"]
    steps = [image for image in images if image["kind"] == "step"]
    if condition == "text_only":
        return []
    if condition == "start_view":
        return start
    if condition == "start_final":
        forced_final_kind = row.get("_forced_final_kind")
        if forced_final_kind == "lookback":
            final = lookback
        elif forced_final_kind == "endpoint":
            final = endpoint
        else:
            final = lookback if show_separate_lookback(row) else endpoint
        return start + final
    if condition == "start_full_trajectory":
        return start + steps + endpoint + (lookback if show_separate_lookback(row) else [])
    raise ValueError(condition)


def question_text(row: dict[str, Any]) -> str:
    choices = "\n".join(f"{letter}. {row['answer_choices'][letter]}" for letter in sorted(row["answer_choices"]))
    return f"Question: {row['question']}\n{choices}\nAnswer:"


def image_prompts(row: dict[str, Any], condition: str) -> list[str]:
    """Use identical final-state wording for both providers and final-image kinds."""
    images = selected_images(row, condition)
    return [
        FINAL_VIEW_PROMPT
        if condition in {"start_final", "start_full_trajectory"}
        and index == len(images) - 1
        and image["kind"] in {"endpoint", "lookback"}
        else f"{image['label']}:"
        for index, image in enumerate(images)
    ]


def gemini_client() -> genai.Client:
    value = getattr(_thread_local, "gemini_client", None)
    if value is None:
        value = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        _thread_local.gemini_client = value
    return value


def openai_client() -> OpenAI:
    value = getattr(_thread_local, "openai_client", None)
    if value is None:
        value = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=180.0)
        _thread_local.openai_client = value
    return value


def gemini_parts(row: dict[str, Any], condition: str) -> list[types.Part]:
    parts: list[types.Part] = []
    for image, caption in zip(selected_images(row, condition), image_prompts(row, condition), strict=True):
        path = Path(image["path"])
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        parts.extend([types.Part.from_text(text=f"{caption}\n"), types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)])
    parts.append(types.Part.from_text(text=question_text(row)))
    return parts


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def openai_content(row: dict[str, Any], condition: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image, caption in zip(selected_images(row, condition), image_prompts(row, condition), strict=True):
        content.append({"type": "input_text", "text": caption})
        content.append({"type": "input_image", "image_url": data_url(Path(image["path"])), "detail": "auto"})
    content.append({"type": "input_text", "text": question_text(row)})
    return content


def generate_gemini(model: str, row: dict[str, Any], condition: str) -> tuple[str, dict[str, Any], str]:
    valid = sorted(row["answer_choices"])
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0,
        max_output_tokens=8192,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
        response_mime_type="text/x.enum",
        response_schema={"type": "STRING", "enum": valid},
    )
    response = gemini_client().models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=gemini_parts(row, condition))],
        config=config,
    )
    prediction = (response.text or "").strip().upper()
    usage = getattr(response, "usage_metadata", None)
    return prediction, usage.model_dump(mode="json") if usage else {}, response.text or ""


def generate_openai(model: str, row: dict[str, Any], condition: str) -> tuple[str, dict[str, Any], str]:
    valid = sorted(row["answer_choices"])
    response = openai_client().responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": openai_content(row, condition)}],
        reasoning={"effort": "high"},
        max_output_tokens=2048,
        text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "answer_letter",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string", "enum": valid}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
        store=False,
    )
    raw = response.output_text or ""
    try:
        parsed = json.loads(raw)
        prediction = str(parsed["answer"]).strip().upper()
    except json.JSONDecodeError:
        prediction = raw.strip().strip('"').upper()
    usage = response.usage.model_dump(mode="json") if response.usage else {}
    return prediction, usage, raw


def generate(args: argparse.Namespace, row: dict[str, Any], condition: str) -> tuple[str, dict[str, Any], str]:
    last_error = ""
    for attempt in range(1, args.retries + 1):
        try:
            if args.provider == "gemini":
                result = generate_gemini(args.model, row, condition)
            else:
                result = generate_openai(args.model, row, condition)
            if result[0] not in sorted(row["answer_choices"]):
                raise ValueError(f"Invalid response {result[0]!r}; raw={result[2]!r}")
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            retryable = any(
                code.lower() in last_error.lower()
                for code in (
                    "429", "500", "502", "503", "504", "UNAVAILABLE",
                    "RESOURCE_EXHAUSTED", "INTERNAL", "timeout", "connection",
                    "ReadError", "WinError 10054", "Invalid response",
                )
            )
            if attempt >= args.retries or not retryable:
                break
            time.sleep(min(45.0, 2**attempt + random.uniform(0.0, 1.0)))
    raise RuntimeError(last_error or "request failed")


def run_one(row: dict[str, Any], condition: str, args: argparse.Namespace) -> dict[str, str]:
    output = args.output_dir / "per_sample" / condition / f"{row['question_id']}.json"
    if args.resume and output.is_file():
        return {"status": "skipped"}
    prediction, usage, raw = generate(args, row, condition)
    valid = sorted(row["answer_choices"])
    if prediction not in valid:
        raise ValueError(f"Invalid response {prediction!r}; raw={raw!r}")
    evidence = selected_images(row, condition)
    result = {
        **row,
        "condition": condition,
        "provider": args.provider,
        "model": args.model,
        "system_prompt": SYSTEM_PROMPT,
        "prompt_version": PROMPT_VERSION,
        "image_prompts": image_prompts(row, condition),
        "reasoning_effort": "high",
        "temperature": 0 if args.provider == "gemini" else None,
        "prediction": prediction,
        "correct": prediction == row["gold_answer"],
        "image_count": len(evidence),
        "evidence_kinds": [image["kind"] for image in evidence],
        "usage_metadata": usage,
        "raw_response": raw,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    atomic_json(output, result)
    return {"status": "ok"}


def aggregate(rows: list[dict[str, Any]], output_dir: Path, conditions: list[str]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        for condition in conditions:
            path = output_dir / "per_sample" / condition / f"{row['question_id']}.json"
            if path.is_file():
                results.append(json.loads(path.read_text(encoding="utf-8")))
    (output_dir / "results_by_sample.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8"
    )
    summary: dict[str, Any] = {
        "provider": results[0]["provider"] if results else None,
        "model": results[0]["model"] if results else None,
        "question_count": len(rows),
        "by_condition": {},
    }
    for condition in conditions:
        selected = [row for row in results if row["condition"] == condition]
        correct = sum(bool(row["correct"]) for row in selected)
        by_set = {}
        for question_set in ("lookback", "non-lookback"):
            subset = [row for row in selected if row["question_set"] == question_set]
            set_correct = sum(bool(row["correct"]) for row in subset)
            by_set[question_set] = {"n": len(subset), "correct": set_correct, "accuracy": set_correct / len(subset) if subset else None}
        summary["by_condition"][condition] = {
            "n": len(selected), "correct": correct, "accuracy": correct / len(selected) if selected else None, "by_set": by_set
        }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("gemini", "openai"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "data" / "manifest.jsonl")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=201,
        help="Expected manifest size before optional trajectory/limit filtering.",
    )
    parser.add_argument(
        "--trajectory-id",
        help="Optionally run only the questions belonging to one trajectory (for targeted reruns).",
    )
    parser.add_argument(
        "--question-ids",
        nargs="+",
        help="Optionally run only these question IDs.",
    )
    parser.add_argument(
        "--force-lookback-ids",
        nargs="*",
        default=(),
        help="For start_final, force these questions to use the oracle lookback image.",
    )
    parser.add_argument(
        "--force-endpoint-ids",
        nargs="*",
        default=(),
        help="For start_final, force these questions to use the normal trajectory endpoint.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    load_env(args.env_file.resolve())
    rows = read_jsonl(args.manifest.resolve())
    if len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} questions, found {len(rows)}")
    if args.trajectory_id:
        rows = [row for row in rows if row.get("trajectory_id") == args.trajectory_id]
        if not rows:
            raise ValueError(f"No questions found for trajectory {args.trajectory_id!r}")
    if args.question_ids:
        requested_ids = set(args.question_ids)
        rows = [row for row in rows if row.get("question_id") in requested_ids]
        found_ids = {row.get("question_id") for row in rows}
        missing_ids = requested_ids - found_ids
        if missing_ids:
            raise ValueError(f"Question IDs not found: {sorted(missing_ids)}")
    force_lookback_ids = set(args.force_lookback_ids)
    force_endpoint_ids = set(args.force_endpoint_ids)
    overlap = force_lookback_ids & force_endpoint_ids
    if overlap:
        raise ValueError(f"Question IDs forced to both endpoint kinds: {sorted(overlap)}")
    for row in rows:
        question_id = row.get("question_id")
        if question_id in force_lookback_ids:
            row["_forced_final_kind"] = "lookback"
            if not any(image["kind"] == "lookback" for image in row["evidence_images"]):
                endpoint_image = next(
                    image for image in row["evidence_images"] if image["kind"] == "endpoint"
                )
                lookback_path = Path(endpoint_image["path"]).with_name("oracle_lookback.png")
                if not lookback_path.is_file():
                    raise FileNotFoundError(
                        f"Missing oracle lookback for {question_id}: {lookback_path}"
                    )
                row["evidence_images"].append(
                    {
                        "label": "Oracle lookback",
                        "kind": "lookback",
                        "path": str(lookback_path),
                    }
                )
        elif question_id in force_endpoint_ids:
            row["_forced_final_kind"] = "endpoint"
    if args.limit:
        rows = rows[: args.limit]
    args.output_dir = args.output_dir.resolve()
    jobs = [(row, condition) for row in rows for condition in args.conditions]
    counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, row, condition, args): (row["question_id"], condition) for row, condition in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            question_id, condition = futures[future]
            try:
                status = future.result()["status"]
                counts[status] += 1
                print(f"[{index}/{len(jobs)}] {status} {question_id} {condition}", flush=True)
            except Exception as exc:  # noqa: BLE001
                counts["failed"] += 1
                safe_error = str(exc).encode("ascii", errors="backslashreplace").decode("ascii")
                print(f"[FAILED] {question_id} {condition}: {safe_error}", flush=True)
    summary = aggregate(rows, args.output_dir, args.conditions)
    complete = not counts["failed"] and all(value["n"] == len(rows) for value in summary["by_condition"].values())
    atomic_json(args.output_dir / "EXPERIMENT_COMPLETE.json", {"status": "complete" if complete else "incomplete", "counts": counts, **summary})
    print(json.dumps(summary, indent=2), flush=True)
    if not complete:
        raise SystemExit(f"Incomplete run: {dict(counts)}")


if __name__ == "__main__":
    main()
