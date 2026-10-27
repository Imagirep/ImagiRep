"""Run the frozen Batch4 200-question Start and Start+final validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict:
    report = json.loads((HERE / "preflight.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    for path, expected in {**report["verified_input_sha256"], **report["frozen_config_sha256"]}.items():
        if sha256(Path(path)) != expected:
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
        print(json.dumps({"status": "prepared_not_run", "questions": 200,
                          "conditions": ["start_view", "start_final"], "planned_calls": 400,
                          "final_image_kind": report["final_image_kind"]}, indent=2))
        return
    if not (ROOT / ".env").is_file():
        raise FileNotFoundError("Configured model environment file is missing")
    command = [
        sys.executable, str(HERE / "runner_snapshot.py"),
        "--provider", "gemini", "--model", args.model,
        "--env-file", str(ROOT / ".env"),
        "--manifest", str(HERE / "manifest.jsonl"), "--expected-count", "200",
        "--conditions", "start_view", "start_final",
        "--output-dir", str(HERE / "results"),
        "--workers", str(args.workers), "--retries", str(args.retries), "--resume",
    ]
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
