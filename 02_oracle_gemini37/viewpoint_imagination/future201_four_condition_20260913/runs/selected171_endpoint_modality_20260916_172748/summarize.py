"""Validate 513 cases and compare to earlier same171 endpoint/full RGB runs."""
import hashlib
import json
from pathlib import Path

import run_modalities as run

HERE = Path(__file__).resolve().parent


def main():
    rows = run.base.read_jsonl(HERE / "manifest.jsonl")
    results = run.base.read_jsonl(HERE / "results/results_by_sample.jsonl")
    preflight = json.loads((HERE / "modality_preflight.json").read_text())
    completion = json.loads((HERE / "results/EXPERIMENT_COMPLETE.json").read_text())
    baseline_path = HERE.parent / "selected171_validation_20260916_122242/results/results_by_sample.jsonl"
    full_rgb_path = HERE.parent / "selected171_full_rgb_20260916_171304/results/results_by_sample.jsonl"
    baseline = run.base.read_jsonl(baseline_path)
    full_rgb = run.base.read_jsonl(full_rgb_path)
    errors = []
    if completion["status"] != "complete" or len(results) != 513:
        errors.append("Expected 513 completed cases")
    manifest = {r["question_id"]: r for r in rows}
    for old in baseline + full_rgb:
        current = manifest.get(old["question_id"])
        if current is None or any(old.get(k) != current[k] for k in
                                  ("question", "answer_choices", "gold_answer", "show_lookback", "question_set", "question_type")):
            errors.append(f"Earlier result differs from current frozen content: {old['question_id']}")
        if old["model"] != "gemini-3.7-flash" or old["system_prompt"] != run.base.SYSTEM_PROMPT:
            errors.append(f"Earlier result model/system mismatch: {old['question_id']}")
    for condition in ("text_only", "start_view", "start_final", "start_full_trajectory"):
        prior_ids = [r["question_id"] for r in baseline + full_rgb if r["condition"] == condition]
        if len(prior_ids) != 171 or set(prior_ids) != set(manifest):
            errors.append(f"Earlier {condition} cases differ from current171")
    expected_ids = {(qid, c) for qid in manifest for c in run.CONDITIONS}
    actual_ids = [(r["question_id"], r["condition"]) for r in results]
    if len(set(actual_ids)) != 513 or set(actual_ids) != expected_ids:
        errors.append("Duplicate or missing result IDs/conditions")
    for result in results:
        row = manifest[result["question_id"]]
        condition = result["condition"]
        for key, value in row.items():
            if result.get(key) != value:
                errors.append(f"{row['question_id']} {condition}: changed manifest field {key}")
        if result["input_signature"] != run.input_signature(row, condition, result["model"]):
            errors.append(f"{row['question_id']} {condition}: changed input signature")
        if result["image_prompts"] != run.image_prompts(row, condition) or result["modality_prompt"] != run.MODALITY_PROMPTS[run.modality(condition)]:
            errors.append(f"{row['question_id']} {condition}: changed captions")
        if result["evidence_kinds"] != [i["kind"] for i in run.selected_images(row, condition)] or result["image_count"] != len(run.selected_images(row, condition)):
            errors.append(f"{row['question_id']} {condition}: changed RGB/image contract")
        if result["prediction"] not in row["answer_choices"] or result["correct"] != (result["prediction"] == row["gold_answer"]):
            errors.append(f"{row['question_id']} {condition}: invalid prediction/scoring")
        expected_fps = 12 if run.modality(condition) == "video" else None
        if result["video_sampling_fps"] != expected_fps:
            errors.append(f"{row['question_id']} {condition}: wrong video sampling")
        if result["model"] != "gemini-3.7-flash" or result["provider"] != "gemini" or result["system_prompt"] != run.base.SYSTEM_PROMPT:
            errors.append(f"{row['question_id']} {condition}: model/system mismatch")
    for path, expected_hash in preflight["verified_input_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
            errors.append(f"Input asset changed during run: {path}")
    conditions = ["text_only", "start_view", "start_final", "start_full_trajectory", *run.CONDITIONS]
    combined = baseline + full_rgb + results

    def stats(selected_rows):
        output = {}
        for condition in conditions:
            selected = [r for r in selected_rows if r["condition"] == condition]
            correct = sum(r["correct"] for r in selected)
            output[condition] = {"n": len(selected), "correct": correct,
                                 "accuracy_pct": 100 * correct / len(selected) if selected else None,
                                 "source": "current_modality_run" if condition in run.CONDITIONS else "earlier_run"}
        return output

    endpoint = {r["question_id"]: r for r in baseline if r["condition"] == "start_final"}
    transitions = {}
    for condition in run.CONDITIONS:
        buckets = {"both_correct": [], "endpoint_wrong_modality_correct": [],
                   "endpoint_correct_modality_wrong": [], "both_wrong": []}
        for result in results:
            if result["condition"] != condition:
                continue
            qid = result["question_id"]
            old, new = endpoint[qid]["correct"], result["correct"]
            key = ("both_correct" if old and new else "endpoint_wrong_modality_correct" if new
                   else "endpoint_correct_modality_wrong" if old else "both_wrong")
            buckets[key].append(qid)
        transitions[condition] = buckets
    analysis = {"verification": "failed" if errors else "passed", "errors": errors,
                "overall": stats(combined),
                "by_set": {s: stats([r for r in combined if r["question_set"] == s]) for s in ("lookback", "non-lookback")},
                "by_type": {str(t): stats([r for r in combined if r["question_type"] == t]) for t in sorted({r["question_type"] for r in rows})},
                "endpoint_to_modality_transitions": transitions,
                "comparison_result_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in (baseline_path, full_rgb_path)},
                "note": "Only three single-modality additions are newly measured; baseline and full RGB are earlier runs."}
    run.base.atomic_json(HERE / "analysis.json", analysis)
    labels = {"text_only": "Text only (earlier)", "start_view": "Start RGB (earlier)",
              "start_final": "Start + endpoint RGB (earlier)", "start_full_trajectory": "Start + full RGB (earlier)",
              "start_final_topdown": "Start + endpoint + top-down", "start_final_text": "Start + endpoint + text trajectory",
              "start_final_video": "Start + endpoint + video trajectory (12 fps)"}
    lines = ["# Selected171 single-modality evaluation", "", "All rows use the same 171 frozen questions.", "",
             "Only the last three rows were newly measured. Baselines come from earlier separate runs.", "",
             "| Setting | Correct / total | Accuracy | Delta vs earlier endpoint (pp) |",
             "|---|---:|---:|---:|"]
    previous = analysis["overall"]["start_final"]["accuracy_pct"]
    for condition in conditions:
        value = analysis["overall"][condition]
        lines.append(f"| {labels[condition]} | {value['correct']}/{value['n']} | {value['accuracy_pct']:.2f}% | {value['accuracy_pct'] - previous:+.2f} |")
    lines += ["", "## Lookback and non-lookback", "",
              "| New setting | Lookback | Non-lookback |", "|---|---:|---:|"]
    for condition in run.CONDITIONS:
        a, b = analysis["by_set"]["lookback"][condition], analysis["by_set"]["non-lookback"][condition]
        lines.append(f"| {labels[condition]} | {a['correct']}/{a['n']} ({a['accuracy_pct']:.2f}%) | {b['correct']}/{b['n']} ({b['accuracy_pct']:.2f}%) |")
    lines += ["", "## Question types", "",
              "| Type | N | Endpoint (earlier) | + Top-down | + Text | + Video |", "|---|---:|---:|---:|---:|---:|"]
    for qtype, values in analysis["by_type"].items():
        cells = [f"{values[c]['accuracy_pct']:.2f}%" for c in ["start_final", *run.CONDITIONS]]
        lines.append(f"| Q{qtype} | {values['start_final']['n']} | " + " | ".join(cells) + " |")
    lines += ["", "## Changes relative to earlier endpoint", "",
              "| Added modality | Wrong to correct | Correct to wrong | Net correct |", "|---|---:|---:|---:|"]
    for condition, buckets in transitions.items():
        improved = len(buckets["endpoint_wrong_modality_correct"])
        regressed = len(buckets["endpoint_correct_modality_wrong"])
        lines.append(f"| {run.modality(condition)} | {improved} | {regressed} | {improved - regressed:+d} |")
    lines += ["", "These are separate model runs, so changes include model answer variability; they do not establish a modality-only causal effect."]
    lines += ["", "Verification: " + analysis["verification"], ""]
    (HERE / "accuracy_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: analysis[k] for k in ("verification", "errors", "overall", "by_set")}, indent=2))
    print("Transitions:", {c: {k: len(v) for k, v in values.items()} for c, values in transitions.items()})
    if errors:
        raise SystemExit("Post-run verification failed")


if __name__ == "__main__":
    main()
