"""Verify actual RGB endpoints and all three corresponding modality variants."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import run_modalities as run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RELEASE = ROOT / "benchmark_release"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rows = run.base.read_jsonl(HERE / "source_manifest.jsonl")
    index = {r["question_id"]: r for r in run.base.read_jsonl(RELEASE / "fo_selected171_modalities.jsonl")}
    published_hashes = json.loads((ROOT / "docs/experiments/fo_selected171_modality_completion_20260916/asset_sha256.json").read_text())
    hashes, videos, errors = {}, {}, []
    for row in rows:
        qid = row["question_id"]
        canonical = index[qid]
        if row["question"] != canonical["question"] or row["original_answer_choices"] != canonical["answer_choices"] or row["original_gold_answer"] != canonical["oracle_answer"]:
            errors.append(f"{qid}: canonical content mismatch")
        expected_kind = "lookback" if row["show_lookback"] else "endpoint"
        if canonical["final_image_kind"] != expected_kind:
            errors.append(f"{qid}: canonical final kind mismatch")
        pair = run.rgb_pair(row)
        row["rgb_pair_sha256"] = [digest(i["path"]) for i in pair]
        if row["rgb_pair_sha256"] != [digest(RELEASE / canonical["frames"][0]["rgb"]), digest(RELEASE / canonical["final_view"]["rgb"])]:
            errors.append(f"{qid}: RGB pair differs from actual canonical input")
        for image in pair:
            with Image.open(image["path"]) as im:
                im.verify()
            hashes[image["path"]] = digest(image["path"])
        row["additional_modalities"] = {}
        for kind in ("topdown", "text", "video"):
            relative = canonical[kind]
            path = RELEASE / relative
            checksum = digest(path)
            if checksum != published_hashes[relative]:
                errors.append(f"{qid}: {kind} differs from verified published asset")
            hashes[str(path)] = checksum
            row["additional_modalities"][kind] = {"kind": kind, "path": str(path), "sha256": checksum}
        with Image.open(row["additional_modalities"]["topdown"]["path"]) as im:
            im.verify()
        text = (RELEASE / canonical["text"]).read_text(encoding="utf-8")
        if not text.strip() or ("oracle lookback camera" in text) != row["show_lookback"]:
            errors.append(f"{qid}: wrong text trajectory variant")
        video_path = RELEASE / canonical["video"]
        metadata_path = RELEASE / canonical["video_manifest"]
        metadata = json.loads(metadata_path.read_text())
        last = metadata["frames"][-1]
        pose = canonical["final_view"]["pose"]
        if last["viewpoint_id"] != pose["viewpoint_id"] or abs(math.remainder(last["heading"] - pose["heading_rad"], 2 * math.pi)) > 1e-6 or abs(last["elevation"] - pose["elevation_rad"]) > 1e-6:
            errors.append(f"{qid}: video final camera mismatch")
        if str(video_path) not in videos:
            capture = cv2.VideoCapture(str(video_path))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            capture.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
            ok, image = capture.read()
            capture.release()
            expected = cv2.resize(cv2.imread(pair[-1]["path"]), (640, 480))
            correlation = float(np.corrcoef(expected.ravel(), image.ravel())[0, 1]) if ok else 0
            if count != metadata["frame_count"] or fps != run.VIDEO_FPS or correlation <= .95:
                errors.append(f"{qid}: video decode/frame count/final RGB mismatch")
            videos[str(video_path)] = {"frame_count": count, "fps": fps, "duration_seconds": count / fps,
                                      "bytes": video_path.stat().st_size, "final_rgb_correlation": correlation}
        row["additional_modalities"]["video"].update(videos[str(video_path)])
        request_size = video_path.stat().st_size + sum(Path(i["path"]).stat().st_size for i in pair)
        if request_size * 4 / 3 + 100_000 >= 20_000_000:
            errors.append(f"{qid}: video request too large for conservative inline limit")
    if len(rows) != 171 or len({r["trajectory_id"] for r in rows}) != 50:
        errors.append("Expected frozen171 / 50 trajectories")
    report = {"status": "failed" if errors else "passed", "questions": len(rows),
              "trajectories": len({r["trajectory_id"] for r in rows}),
              "conditions": list(run.CONDITIONS), "expected_cases": len(rows) * 3,
              "final_image_kinds": dict(Counter(run.rgb_pair(r)[-1]["kind"] for r in rows)),
              "video_sampling_fps": run.VIDEO_FPS, "verified_videos": videos,
              "source_sha256": digest(HERE / "source_manifest.jsonl"),
              "modality_index_sha256": digest(RELEASE / "fo_selected171_modalities.jsonl"),
              "verified_input_sha256": hashes, "errors": errors}
    run.base.atomic_json(HERE / "modality_preflight.json", report)
    print({k: v for k, v in report.items() if k not in {"verified_input_sha256", "verified_videos"}})
    print("Videos:", len(videos), "max duration:", max(v["duration_seconds"] for v in videos.values()),
          "max bytes:", max(v["bytes"] for v in videos.values()))
    if errors:
        raise SystemExit("Preflight failed; do not make model calls")
    (HERE / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
