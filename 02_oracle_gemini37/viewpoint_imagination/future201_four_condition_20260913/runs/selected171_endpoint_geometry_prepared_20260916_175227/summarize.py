"""Validate endpoint geometry results; compare the same171 historical settings."""
import hashlib
import json
from pathlib import Path

import run_endpoint_geometry as run

HERE = Path(__file__).resolve().parent
OLD_FILES = [
    HERE.parent / "selected171_validation_20260916_122242/results/results_by_sample.jsonl",
    HERE.parent / "selected171_full_rgb_20260916_171304/results/results_by_sample.jsonl",
    HERE.parent / "selected171_endpoint_modality_20260916_172748/results/results_by_sample.jsonl",
]
OLD_CONDITIONS = ["text_only", "start_view", "start_final", "start_full_trajectory",
                  "start_final_topdown", "start_final_text", "start_final_video"]


def main():
    rows = run.base.read_jsonl(HERE / "manifest.jsonl")
    current = run.base.read_jsonl(HERE / "results/results_by_sample.jsonl")
    earlier = [row for path in OLD_FILES for row in run.base.read_jsonl(path)]
    completion = json.loads((HERE / "results/EXPERIMENT_COMPLETE.json").read_text())
    errors = []
    if completion["status"] != "complete" or len(current) != 513:
        errors.append("Expected all 513 successful current cases")
    manifest = {r["question_id"]: r for r in rows}
    expected_cases = {(qid, c) for qid in manifest for c in run.CONDITIONS}
    actual = [(r["question_id"], r["condition"]) for r in current]
    if len(set(actual)) != 513 or set(actual) != expected_cases:
        errors.append("Duplicate or missing current IDs/conditions")
    try:
        run.verify_prepared_inputs()
    except ValueError as exc:
        errors.append(str(exc))
    for result in current:
        row = manifest[result["question_id"]]
        condition = result["condition"]
        for key, value in row.items():
            if result.get(key) != value:
                errors.append(f"{row['question_id']} {condition}: changed manifest field {key}")
        if result["input_signature"] != run.input_signature(row, condition, result["model"]):
            errors.append(f"{row['question_id']} {condition}: changed input signature")
        if result["image_prompts"] != run.image_prompts(row, condition):
            errors.append(f"{row['question_id']} {condition}: changed captions")
        if result["evidence_kinds"] != [i["kind"] for i in run.selected_images(row, condition)] or result["image_count"] != 3:
            errors.append(f"{row['question_id']} {condition}: wrong three-image contract")
        if result["endpoint_modality"] != row["endpoint_modalities"][run.modality(condition)]:
            errors.append(f"{row['question_id']} {condition}: wrong endpoint modality")
        if result["prediction"] not in row["answer_choices"] or result["correct"] != (result["prediction"] == row["gold_answer"]):
            errors.append(f"{row['question_id']} {condition}: invalid prediction/scoring")
        if result["model"] != "gemini-3.7-flash" or result["provider"] != "gemini" or result["system_prompt"] != run.base.SYSTEM_PROMPT:
            errors.append(f"{row['question_id']} {condition}: wrong provider/model/system")
    for result in earlier:
        row = manifest.get(result["question_id"])
        if row is None or any(result.get(k) != row[k] for k in
                              ("question", "answer_choices", "gold_answer", "show_lookback", "question_set", "question_type")):
            errors.append(f"Earlier result content differs: {result['question_id']}")
        if result["model"] != "gemini-3.7-flash" or result["system_prompt"] != run.base.SYSTEM_PROMPT:
            errors.append(f"Earlier model/system differs: {result['question_id']}")
    for condition in OLD_CONDITIONS:
        ids = [r["question_id"] for r in earlier if r["condition"] == condition]
        if len(ids) != 171 or set(ids) != set(manifest):
            errors.append(f"Earlier {condition} set differs from frozen171")
    conditions = [*OLD_CONDITIONS, *run.CONDITIONS]
    combined = earlier + current

    def statistics(selected_rows):
        stats = {}
        for condition in conditions:
            selected = [r for r in selected_rows if r["condition"] == condition]
            correct = sum(r["correct"] for r in selected)
            stats[condition] = {"n": len(selected), "correct": correct,
                                "accuracy_pct": 100 * correct / len(selected) if selected else None,
                                "source": "current_geometry_run" if condition in run.CONDITIONS else "earlier_run"}
        return stats

    endpoint = {r["question_id"]: r for r in earlier if r["condition"] == "start_final"}
    transitions = {}
    for condition in run.CONDITIONS:
        buckets = {"both_correct": [], "endpoint_wrong_modality_correct": [],
                   "endpoint_correct_modality_wrong": [], "both_wrong": []}
        for result in current:
            if result["condition"] != condition:
                continue
            qid = result["question_id"]
            old, new = endpoint[qid]["correct"], result["correct"]
            key = ("both_correct" if old and new else "endpoint_wrong_modality_correct" if new
                   else "endpoint_correct_modality_wrong" if old else "both_wrong")
            buckets[key].append(qid)
        transitions[condition] = buckets
    analysis = {"verification": "failed" if errors else "passed", "errors": errors,
                "overall": statistics(combined),
                "by_set": {s: statistics([r for r in combined if r["question_set"] == s])
                           for s in ("lookback", "non-lookback")},
                "by_type": {str(t): statistics([r for r in combined if r["question_type"] == t])
                            for t in sorted({r["question_type"] for r in rows})},
                "endpoint_to_modality_transitions": transitions,
                "comparison_result_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in OLD_FILES},
                "note": "Only endpoint depth/normals/panorama are newly measured; other settings are earlier same171 runs."}
    run.base.atomic_json(HERE / "analysis.json", analysis)
    labels = {"text_only": "Text only (earlier)", "start_view": "Start RGB (earlier)",
              "start_final": "Start + endpoint RGB (earlier)", "start_full_trajectory": "Start + full RGB (earlier)",
              "start_final_topdown": "Start + endpoint + top-down (earlier)",
              "start_final_text": "Start + endpoint + text trajectory (earlier)",
              "start_final_video": "Start + endpoint + video trajectory (earlier)",
              "start_final_depth": "Start + endpoint + endpoint depth",
              "start_final_surface_normals": "Start + endpoint + endpoint surface normals",
              "start_final_panorama": "Start + endpoint + endpoint panorama"}
    lines = ["# Selected171 endpoint geometry results", "",
             "Only the last three settings are newly measured; all other scores are from earlier same171 runs.", "",
             "| Setting | Correct / total | Accuracy | Delta vs earlier endpoint (pp) |", "|---|---:|---:|---:|"]
    previous = analysis["overall"]["start_final"]["accuracy_pct"]
    for condition in conditions:
        value = analysis["overall"][condition]
        lines.append(f"| {labels[condition]} | {value['correct']}/{value['n']} | {value['accuracy_pct']:.2f}% | {value['accuracy_pct'] - previous:+.2f} |")
    lines += ["", "## Lookback / non-lookback", "",
              "| Setting | Lookback | Non-lookback |", "|---|---:|---:|"]
    for condition in ["start_final", *run.CONDITIONS]:
        a, b = analysis["by_set"]["lookback"][condition], analysis["by_set"]["non-lookback"][condition]
        lines.append(f"| {labels[condition]} | {a['correct']}/{a['n']} ({a['accuracy_pct']:.2f}%) | {b['correct']}/{b['n']} ({b['accuracy_pct']:.2f}%) |")
    lines += ["", "## Question types", "", "| Type | N | Endpoint (earlier) | + Depth | + Normals | + Panorama |",
              "|---|---:|---:|---:|---:|---:|"]
    for qtype, values in analysis["by_type"].items():
        cells = [f"{values[c]['accuracy_pct']:.2f}%" for c in ["start_final", *run.CONDITIONS]]
        lines.append(f"| Q{qtype} | {values['start_final']['n']} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired changes against earlier endpoint", "",
              "| Added modality | Wrong to correct | Correct to wrong | Net correct |", "|---|---:|---:|---:|"]
    for condition, buckets in transitions.items():
        improved, regressed = len(buckets["endpoint_wrong_modality_correct"]), len(buckets["endpoint_correct_modality_wrong"])
        lines.append(f"| {run.modality(condition)} | {improved} | {regressed} | {improved - regressed:+d} |")
    lines += ["", "Cross-run changes include model answer variability; they do not isolate a modality-only causal effect.",
              "", "Verification: " + analysis["verification"], ""]
    (HERE / "accuracy_table.md").write_text("\n".join(lines), encoding="utf-8")
    run.base.atomic_json(HERE / "execution_status.json", {
        "status": "complete_verified" if not errors else "complete_validation_failed",
        "model": "gemini-3.7-flash", "questions": 171, "expected_cases": 513,
        "completed_cases": len(current), "conditions": list(run.CONDITIONS), "prompt_changed_since_review": False,
        "verification": analysis["verification"], "errors": errors})
    print(json.dumps({k: analysis[k] for k in ("verification", "errors", "overall", "by_set")}, indent=2))
    print("Transitions:", {c: {k: len(v) for k, v in b.items()} for c, b in transitions.items()})
    if errors:
        raise SystemExit("Post-run verification failed")


if __name__ == "__main__":
    main()
