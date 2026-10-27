"""Preview or execute the frozen Batch3 160-question Start/Endpoint rerun."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CONDITIONS = ("start_view", "start_final")


def verify() -> dict:
    report = json.loads((HERE / "preflight.json").read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError("Preflight did not pass")
    for path, expected in {**report["verified_input_sha256"], **report["frozen_config_sha256"]}.items():
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen input/config changed: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", default="gemini-3.7-flash")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()
    report = verify()
    if not args.execute:
        print(json.dumps({"status": "prepared_not_run", "model_calls": 0, "questions": 160,
                          "conditions": CONDITIONS, "planned_cases": 320,
                          "final_evidence": report["final_evidence"]}, indent=2))
        return
    command = [sys.executable, str(HERE / "runner_snapshot.py"), "--provider", "gemini",
               "--model", args.model, "--env-file", str(ROOT / ".env"),
               "--manifest", str(HERE / "manifest.jsonl"), "--expected-count", "160",
               "--conditions", *CONDITIONS, "--output-dir", str(HERE / "results"),
               "--workers", str(args.workers), "--retries", str(args.retries), "--resume"]
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()

