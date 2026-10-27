from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CURRENT_RUN = HERE.parent
RUNS = CURRENT_RUN.parent
FIRST_VALIDATION = RUNS / "batch3_180_gap_validation_gemini37_20260921_104002"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest_path = CURRENT_RUN / "manifest.jsonl"
first_results_path = FIRST_VALIDATION / "results" / "results_by_sample.jsonl"
current_results_path = CURRENT_RUN / "results" / "results_by_sample.jsonl"
manifest = read_jsonl(manifest_path)
question_ids = {row["question_id"] for row in manifest}

first = {
    (row["question_id"], row["condition"]): row
    for row in read_jsonl(first_results_path)
    if row["question_id"] in question_ids
}
current = {
    (row["question_id"], row["condition"]): row
    for row in read_jsonl(current_results_path)
}

selection = {
    "text_only": (first, "batch3_180_gap_validation_gemini37_20260921_104002"),
    "start_view": (current, "batch3_160_start_endpoint_validation_gemini37_20260921_113511"),
    "start_final": (first, "batch3_180_gap_validation_gemini37_20260921_104002"),
}
rows = []
for item in manifest:
    qid = item["question_id"]
    conditions = {}
    for condition, (source, source_run) in selection.items():
        record = source[(qid, condition)]
        conditions[condition] = {
            "prediction": record["prediction"],
            "correct": bool(record["correct"]),
            "source_run": source_run,
            "completed_at": record["completed_at"],
        }
    rows.append({
        "question_id": qid,
        "trajectory_id": item["trajectory_id"],
        "question_type": item["question_type"],
        "gold_answer": item["gold_answer"],
        "conditions": conditions,
    })

assert len(rows) == len({row["question_id"] for row in rows}) == 160
correct = {
    condition: sum(row["conditions"][condition]["correct"] for row in rows)
    for condition in selection
}
assert correct == {"text_only": 42, "start_view": 91, "start_final": 105}

summary = {
    "status": "complete_verified",
    "designation": "User-designated Batch3 160-question independent result",
    "questions": 160,
    "trajectories": len({row["trajectory_id"] for row in rows}),
    "scores": {
        condition: {
            "correct": value,
            "n": 160,
            "accuracy": value / 160,
            "source_run": selection[condition][1],
        }
        for condition, value in correct.items()
    },
    "deltas": {
        "text_to_start": {
            "net_correct": correct["start_view"] - correct["text_only"],
            "percentage_points": 100 * (correct["start_view"] - correct["text_only"]) / 160,
        },
        "start_to_final": {
            "net_correct": correct["start_final"] - correct["start_view"],
            "percentage_points": 100 * (correct["start_final"] - correct["start_view"]) / 160,
        },
    },
    "provenance": {
        "manifest_sha256": sha256(manifest_path),
        "first_validation_results_sha256": sha256(first_results_path),
        "current_validation_results_sha256": sha256(current_results_path),
        "mixed_run_composite": True,
        "note": "This requested reporting record combines condition outputs from two API rounds; raw outputs remain unchanged.",
    },
}

(HERE / "results_by_question.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
)
(HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

readme = f"""# Designated Batch3 160-question independent result

This record uses the user-designated reporting values for the frozen 160-question manifest:

| Condition | Correct | Accuracy | Source API round |
|---|---:|---:|---|
| Text only | 42/160 | 26.25% | First 180-question validation, restricted to this membership |
| Start | 91/160 | 56.88% | Current 160-question validation |
| Start + endpoint | 105/160 | 65.62% | First 180-question validation, restricted to this membership |

Start improves over text-only by 49 questions / 30.63 points. Start + endpoint improves over Start by 14 questions / 8.75 points.

`results_by_question.jsonl` preserves the exact prediction and correctness state selected for every question and condition. `summary.json` records source-run IDs and hashes. This is a mixed-run reporting composite; the underlying raw model outputs are unchanged.
"""
(HERE / "README.md").write_text(readme, encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
