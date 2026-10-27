"""Convert the frozen benchmark rows into the MindJourney/AVIC input format (2026-10-06).

Inputs are OBSERVED views only:
  FO 622 (canonical_622_rows.jsonl): the Start image.
  MVC 325 (run_mvc_modalities.rows()): the numbered images in displayed order (settings.original_only).
Answer choices are passed as the choice texts in the frozen shuffled A-D order; correct_answer is the gold text.
Writes <out>/<set>/test.json (+ manifest.json) and copies images to <out>/images/ (server path prefix applied).
  python build_inputs.py <out_dir> <server_root>
"""
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FO_ROWS = ROOT / "docs/experiments/fo_estimated_seven_conditions_622_v5_20261005/canonical_622_rows.jsonl"
MVC_RUNNER = ROOT / "docs/experiments/mvc_oracle_modalities_325_20261006/run_mvc_modalities.py"
SMOKE = ["FO-0001-Q2", "FO-0003-Q3", "FOB2-0001", "FOB3", "FOB4"]  # prefixes; first match per prefix


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(out, server_root):
    out = Path(out); img_dir = out / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    srv = server_root.rstrip("/") + "/inputs/images/"

    def stage(src, name):
        src = Path(src)
        dst = img_dir / name
        if not dst.exists():
            shutil.copy(src, dst)
        return srv + name

    sets = {"fo622": [], "mvc325": []}
    for line in FO_ROWS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        start = [e for e in r["evidence_images"] if e["kind"] == "start"]
        assert len(start) == 1, r["question_id"]
        ch = [r["answer_choices"][k] for k in "ABCD"]
        assert len({c.lower() for c in ch}) == 4, (r["question_id"], ch)
        sets["fo622"].append(dict(
            database_idx=r["question_id"], question_type=r["category"], question=r["question"], answer_choices=ch,
            correct_answer=r["answer_choices"][r["gold_answer"]], gold_letter=r["gold_answer"],
            img_paths=[stage(start[0]["path"], f"fo_{r['trajectory_id']}_start.png")],
            camera_pitch_deg=0.0,  # all FO start views are level (trajectory texts: "Your initial view is level")
            source=dict(benchmark="FO622", question_set=r["question_set"], trajectory_id=r["trajectory_id"],
                        start_sha256=sha(start[0]["path"]))))

    spec = importlib.util.spec_from_file_location("mvc_run", MVC_RUNNER)
    mvc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mvc)
    for r in mvc.rows():
        ch = [r["answer_choices"][k] for k in "ABCD"]
        assert len({c.lower() for c in ch}) == 4, (r["uid"], ch)
        imgs = r["settings"]["original_only"]
        assert [a["label"] for a in imgs] == [f"Image {i}" for i in range(1, len(imgs) + 1)], r["uid"]
        paths = [stage(ROOT / a["path"], f"mvc_{r['uid']}_img{i}.png") for i, a in enumerate(imgs, 1)]  # same resolution as run_mvc_modalities.parts()
        sets["mvc325"].append(dict(
            database_idx=r["uid"], question_type=r["question_type"], question=r["question"], answer_choices=ch,
            correct_answer=r["answer_choices"][r["gold_answer"]], gold_letter=r["gold_answer"], img_paths=paths,
            camera_pitch_deg=-15.0,  # all MVC views are 15 deg below level (view_transition.txt)
            source=dict(benchmark="MVC325", cohort=r["cohort"], resource_id=r["resource_id"])))

    # smoke: 5 FO (mixed batches/lookback) + 5 MVC (both types, incl. a 3-view set)
    fo, mv = sets["fo622"], sets["mvc325"]
    smoke = []
    for pref in SMOKE:
        smoke.append(next(q for q in fo if q["database_idx"].startswith(pref) and q not in smoke))
    for t in ("cross_view_closest", "cross_view_farthest", "ordering"):
        smoke.append(next(q for q in mv if q["question_type"] == t))
    smoke += [q for q in mv if len(q["img_paths"]) == 3][:1]
    smoke.append(next(q for q in mv if q["source"]["cohort"] == "last50" and q not in smoke))
    sets["smoke10"] = smoke

    for name, qs in sets.items():
        d = out / name; d.mkdir(exist_ok=True)
        (d / "test.json").write_text(json.dumps(qs, indent=1, ensure_ascii=False), encoding="utf-8")
        print(name, len(qs))
    (out / "manifest.json").write_text(json.dumps(dict(
        fo_rows=str(FO_ROWS.relative_to(ROOT)), fo_rows_sha256=sha(FO_ROWS), mvc_runner=str(MVC_RUNNER.relative_to(ROOT)),
        counts={k: len(v) for k, v in sets.items()}, images=len(list(img_dir.iterdir()))), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
