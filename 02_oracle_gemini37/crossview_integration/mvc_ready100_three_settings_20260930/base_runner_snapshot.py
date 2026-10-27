#!/usr/bin/env python3
"""Run the three matched Gemini 3.7 MultiView-200 RGB conditions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


CONDITIONS = ("original_only", "oracle_rgb", "wrong_rgb")
_thread_local = threading.local()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_env(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip("'\"")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def client() -> genai.Client:
    value = getattr(_thread_local, "client", None)
    if value is None:
        value = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        _thread_local.client = value
    return value


def text_part(text: str) -> types.Part:
    return types.Part.from_text(text=text)


def image_part(path: Path) -> types.Part:
    if not path.is_file():
        raise FileNotFoundError(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


def option_text(choices: dict[str, str]) -> str:
    return "\n".join(f"{key}. {value}" for key, value in sorted(choices.items()))


def prompt_parts(
    row: dict[str, Any], condition: str, root: Path, prompts: dict[str, str]
) -> tuple[list[types.Part], list[str], str]:
    choices = {str(key): str(value) for key, value in row["answer_choices"].items()}
    shown_paths = [root / asset["path"] for asset in row["shown_rgb"]]
    parts = []
    image_paths = []
    for index, shown in enumerate(shown_paths, start=1):
        parts.extend([text_part(f"Image {index}:\n"), image_part(shown)])
        image_paths.append(str(shown))

    if condition in {"oracle_rgb", "wrong_rgb"}:
        key = "oracle_rgb" if condition == "oracle_rgb" else "wrong_rgb"
        additional = root / row[key]["path"]
        parts.extend(
            [
                text_part("\nAdditional simulated RGB view rendered at the same fixed viewpoint:\n"),
                image_part(additional),
            ]
        )
        image_paths.append(str(additional))
        condition_prompt = prompts["additional_rgb"]
    else:
        condition_prompt = prompts["original_only"]

    parts.append(
        text_part(
            f"\n{condition_prompt}\n\nQuestion: {row['question']}\n"
            f"{option_text(choices)}\nAnswer:"
        )
    )
    return parts, image_paths, condition_prompt


def prompt_hash(system: str, condition_prompt: str, row: dict[str, Any], condition: str) -> str:
    family = "additional_rgb" if condition in {"oracle_rgb", "wrong_rgb"} else condition
    value = {
        "system": system,
        "family": family,
        "condition_prompt": condition_prompt,
        "question": row["question"],
        "answer_choices": row["answer_choices"],
        "image_labels": [
            *[f"image_{index}" for index in range(1, len(row["shown_rgb"]) + 1)],
            "additional_rgb",
        ]
        if family == "additional_rgb"
        else [f"image_{index}" for index in range(1, len(row["shown_rgb"]) + 1)],
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def generate(
    model: str,
    system: str,
    parts: list[types.Part],
    valid_options: list[str],
    retries: int,
) -> tuple[str, dict[str, Any]]:
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0,
        max_output_tokens=8192,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
        response_mime_type="text/x.enum",
        response_schema={"type": "STRING", "enum": valid_options},
    )
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            response = client().models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
                config=config,
            )
            answer = (response.text or "").strip().upper()
            usage = getattr(response, "usage_metadata", None)
            usage_dict = usage.model_dump(mode="json") if usage is not None else {}
            if answer not in valid_options:
                raise ValueError(f"Invalid enum response {answer!r}; valid={valid_options}")
            return answer, usage_dict
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            retryable = any(
                marker in last_error
                for marker in ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "INTERNAL")
            )
            if attempt >= retries or not retryable:
                break
            wait_seconds = min(120.0, 2**attempt + random.uniform(0.0, 1.5))
            print(f"[retry {attempt}/{retries}] {last_error}; sleep={wait_seconds:.1f}s", flush=True)
            time.sleep(wait_seconds)
    raise RuntimeError(last_error or "Gemini request failed")


def run_one(
    row: dict[str, Any],
    condition: str,
    args: argparse.Namespace,
    system: str,
    prompts: dict[str, str],
) -> dict[str, Any]:
    output = args.output_dir / "per_sample" / condition / f"{row['question_id']}.json"
    if args.resume and output.is_file():
        return {"status": "skipped", "question_id": row["question_id"], "condition": condition}
    parts, image_paths, condition_prompt = prompt_parts(row, condition, args.experiment_root, prompts)
    valid_options = sorted(str(key).upper() for key in row["answer_choices"])
    prediction, usage = generate(args.model, system, parts, valid_options, args.retries)
    result = {
        "status": "ok",
        "row_index": row["row_index"],
        "question_id": row["question_id"],
        "condition": condition,
        "model": args.model,
        "temperature": 0,
        "thinking_level": "HIGH",
        "response_format": "dynamic option enum",
        "system_prompt": system,
        "condition_prompt": condition_prompt,
        "prompt_hash": prompt_hash(system, condition_prompt, row, condition),
        "question": row["question"],
        "answer_choices": row["answer_choices"],
        "gold_answer": row["gold_answer"],
        "prediction": prediction,
        "correct": prediction == str(row["gold_answer"]).upper(),
        "image_paths": [str(Path(path).relative_to(args.experiment_root)) for path in image_paths],
        "image_count": len(image_paths),
        "usage_metadata": usage,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_json(output, result)
    return result


def collect_results(output_dir: Path, rows: list[dict[str, Any]], conditions: list[str]) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        for condition in conditions:
            path = output_dir / "per_sample" / condition / f"{row['question_id']}.json"
            if path.is_file():
                results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def write_aggregate(output_dir: Path, results: list[dict[str, Any]], conditions: list[str]) -> dict[str, Any]:
    result_path = output_dir / "results_by_sample.jsonl"
    result_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8"
    )
    csv_path = output_dir / "results_by_sample.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_index", "question_id", "condition", "gold_answer", "prediction", "correct", "image_count"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    by_condition = {}
    for condition in conditions:
        selected = [row for row in results if row["condition"] == condition]
        correct = sum(bool(row["correct"]) for row in selected)
        by_condition[condition] = {
            "n": len(selected),
            "correct": correct,
            "accuracy": correct / len(selected) if selected else None,
        }
    by_pair = {(row["question_id"], row["condition"]): row for row in results}

    def transition(left: str, right: str) -> dict[str, Any]:
        pairs = [
            (by_pair[(question_id, left)], by_pair[(question_id, right)])
            for question_id in sorted({row["question_id"] for row in results})
            if (question_id, left) in by_pair and (question_id, right) in by_pair
        ]
        return {
            "n": len(pairs),
            "prediction_changed": sum(a["prediction"] != b["prediction"] for a, b in pairs),
            "correct_to_incorrect": sum(a["correct"] and not b["correct"] for a, b in pairs),
            "incorrect_to_correct": sum(not a["correct"] and b["correct"] for a, b in pairs),
        }

    summary = {
        "model": results[0]["model"] if results else None,
        "result_rows": len(results),
        "by_condition": by_condition,
        "transitions": {
            "original_to_oracle_rgb": transition("original_only", "oracle_rgb"),
            "original_to_wrong_rgb": transition("original_only", "wrong_rgb"),
            "oracle_rgb_to_wrong_rgb": transition("oracle_rgb", "wrong_rgb"),
        },
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prompt-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.7-flash")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.experiment_root = args.experiment_root.resolve()
    args.manifest = (args.manifest or args.experiment_root / "data" / "manifest.jsonl").resolve()
    args.prompt_dir = (args.prompt_dir or args.experiment_root / "prompts").resolve()
    args.output_dir = (args.output_dir or args.experiment_root / "outputs" / "gemini37_high").resolve()
    load_env(args.env_file.resolve())
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set")
    rows = read_jsonl(args.manifest)
    if len(rows) != 200:
        raise ValueError(f"Expected 200 input rows, found {len(rows)}")
    if args.limit:
        rows = rows[: args.limit]
    elif not args.confirm_full_run:
        raise RuntimeError("Full 600-call run requires --confirm-full-run")
    prompts = {
        name: (args.prompt_dir / f"{name}.txt").read_text(encoding="utf-8").strip()
        for name in ("system", "original_only", "additional_rgb")
    }
    system = prompts.pop("system")
    if prompts["additional_rgb"] != (args.prompt_dir / "additional_rgb.txt").read_text(encoding="utf-8").strip():
        raise AssertionError("Matched additional-RGB prompt changed while loading")
    jobs = [(row, condition) for row in rows for condition in args.conditions]
    counts = Counter()
    print(f"Loaded {len(rows)} questions and {len(jobs)} jobs; workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, row, condition, args, system, prompts): (row["question_id"], condition)
            for row, condition in jobs
        }
        for future in as_completed(futures):
            question_id, condition = futures[future]
            try:
                result = future.result()
                counts[result["status"]] += 1
                print(f"[{sum(counts.values()):03d}/{len(jobs):03d}] {result['status']} {question_id} {condition}", flush=True)
            except Exception as exc:  # noqa: BLE001
                counts["failed"] += 1
                print(f"[FAILED] {question_id} {condition}: {type(exc).__name__}: {exc}", flush=True)
    results = collect_results(args.output_dir, rows, args.conditions)
    summary = write_aggregate(args.output_dir, results, args.conditions)
    expected = len(rows) * len(args.conditions)
    if len(results) != expected or counts["failed"]:
        raise SystemExit(f"Incomplete run: results={len(results)}/{expected}, counts={dict(counts)}")
    if len(rows) == 200 and set(args.conditions) == set(CONDITIONS):
        marker = {
            "status": "complete",
            "experiment": "multiview200_gemini37_oracle_wrong_rgb",
            "completed_rows": len(results),
            "question_count": len(rows),
            "conditions": list(CONDITIONS),
            "model": args.model,
            "summary": summary,
        }
        atomic_json(args.experiment_root / "EXPERIMENT_COMPLETE.json", marker)
    print(json.dumps({"counts": dict(counts), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
