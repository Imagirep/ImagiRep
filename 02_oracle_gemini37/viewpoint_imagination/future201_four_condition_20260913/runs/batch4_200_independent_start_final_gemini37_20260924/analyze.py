"""Verify and score the independent Batch4 200-question two-condition rerun."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDITIONS = ("start_view", "start_final")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(rows):
    n = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    return {"n": n, "correct": correct, "accuracy": correct / n if n else None}


def main() -> None:
    manifest = read_jsonl(HERE / "manifest.jsonl")
    frozen = {row["question_id"]: row for row in manifest}
    assert len(manifest) == len(frozen) == 200
    assert len({row["trajectory_id"] for row in manifest}) == 50
    preflight = read_json(HERE / "preflight.json")
    hash_errors = [path for path, expected in
                   {**preflight["verified_input_sha256"], **preflight["frozen_config_sha256"]}.items()
                   if not Path(path).is_file() or sha256(Path(path)) != expected]
    assert not hash_errors, hash_errors
    results = read_jsonl(HERE / "results/results_by_sample.jsonl")
    complete = read_json(HERE / "results/EXPERIMENT_COMPLETE.json")
    assert complete["status"] == "complete" and len(results) == 400
    indexed = {}
    audit = {"status": "complete_verified", "questions": 200, "trajectories": 50,
             "result_rows": len(results), "frozen_hashes_checked":
             len(preflight["verified_input_sha256"]) + len(preflight["frozen_config_sha256"]),
             "conditions": {}}
    for condition in CONDITIONS:
        selected = [row for row in results if row["condition"] == condition]
        by_id = {row["question_id"]: row for row in selected}
        files = list((HERE / "results/per_sample" / condition).glob("*.json"))
        assert len(selected) == len(by_id) == len(files) == 200
        assert set(by_id) == set(frozen)
        for question_id, row in by_id.items():
            source = frozen[question_id]
            expected_kinds = (["start"] if condition == "start_view" else
                              ["start", source["final_image_kind"]])
            assert row["evidence_kinds"] == expected_kinds
            assert row["image_count"] == len(expected_kinds)
            assert row["question"] == source["question"]
            assert row["answer_choices"] == source["answer_choices"]
            assert row["gold_answer"] == source["gold_answer"]
            assert row["prediction"] in "ABCD"
            assert row["correct"] == (row["prediction"] == row["gold_answer"])
            assert row["provider"] == "gemini" and row["model"] == "gemini-3.7-flash"
            assert row["reasoning_effort"] == "high" and row["temperature"] == 0
            assert read_json(HERE / "results/per_sample" / condition / f"{question_id}.json") == row
        indexed[condition] = by_id
        audit["conditions"][condition] = {"questions": len(by_id), "sample_files": len(files)}
    transitions = Counter((bool(indexed["start_view"][qid]["correct"]),
                           bool(indexed["start_final"][qid]["correct"])) for qid in frozen)
    w2c, c2w = transitions[(False, True)], transitions[(True, False)]
    discordant = w2c + c2w
    p_value = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(w2c, c2w) + 1)) / 2**discordant) if discordant else 1.0
    analysis = {
        "status": "complete_verified", "questions": 200, "trajectories": 50,
        "conditions": {condition: {
            "overall": score(list(indexed[condition].values())),
            "by_final_kind": {kind: score([row for row in indexed[condition].values()
                                          if frozen[row["question_id"]]["final_image_kind"] == kind])
                              for kind in ("lookback", "endpoint")},
            "by_route_variant": {variant: score([row for row in indexed[condition].values()
                                                 if frozen[row["question_id"]]["route_variant"] == variant])
                                 for variant in ("old_straight", "old_bent", "new_straight", "new_bent")},
        } for condition in CONDITIONS},
        "start_to_final": {
            "both_correct": transitions[(True, True)], "wrong_to_correct": w2c,
            "correct_to_wrong": c2w, "both_wrong": transitions[(False, False)],
            "net_correct": w2c - c2w,
            "gain_percentage_points": 100 * (w2c - c2w) / 200,
            "exact_mcnemar_p_two_sided": p_value,
        },
        "selection_round_gain_percentage_points": [20.0, 17.5],
    }
    (HERE / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    (HERE / "postrun_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"conditions": {key: value["overall"] for key, value in analysis["conditions"].items()},
                      "start_to_final": analysis["start_to_final"]}, indent=2))


if __name__ == "__main__":
    main()
