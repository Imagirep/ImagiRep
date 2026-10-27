"""Build compact camera-pose / viewpoint-id files for the ImagiRep release (no images).

Inputs (local repo, read-only):
  outputs/fo_all_batches_text_natural_v2_20261001/{live_index.jsonl, verification.json}  (exact FO geometry)
  Matterport3DSimulator/connectivity/<scan>_connectivity.json                           (viewpoint positions)
  outputs/mvc300_metadata_export_20261009/MVC300_benchmark922_v2/{index.csv, scenes/*/*/camera_poses.json}
  question_index.tsv (from the release, for uid sets and scan ids)
"""
import csv, json, math, sys
from pathlib import Path

ROOT = Path(r"<authors_workspace>")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
qidx = {r["uid"]: r for r in csv.DictReader(open(sys.argv[2], encoding="utf-8"), delimiter="\t")}
fo_uids = sorted(u for u, r in qidx.items() if r["benchmark"] == "FO622")
mvc_uids = sorted(u for u, r in qidx.items() if r["benchmark"] == "MVC300")

# ---------------- Viewpoint Imagination
fo_dir = ROOT / "outputs/fo_all_batches_text_natural_v2_20261001"
live = {r["question_id"]: r for r in map(json.loads, open(fo_dir / "live_index.jsonl", encoding="utf-8"))}
variants = {v["text_path"]: v for v in json.load(open(fo_dir / "verification.json", encoding="utf-8"))["exact_geometry"]}
conn = {}


def position(scan, vp):
    if scan not in conn:
        data = json.load(open(ROOT / f"Matterport3DSimulator/connectivity/{scan}_connectivity.json", encoding="utf-8"))
        conn[scan] = {d["image_id"]: d["pose"] for d in data}
    p = conn[scan][vp]
    return [round(p[3], 5), round(p[7], 5), round(p[11], 5)]


n_fo = 0
with open(OUT / "viewpoint_imagination_poses.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for u in fo_uids:
        q = live[u]; v = variants[q["text"]]; scan = qidx[u]["scan_id"]
        assert v["trajectory_id"] == q["trajectory_id"] == qidx[u]["trajectory_or_view_group"] and v["lookback"] == q["show_lookback"]
        acts = v["action_sequence"]
        path = [acts[0]["source_viewpoint"]] + [a["target_viewpoint"] for a in acts]
        poses = []
        for c in v["camera_poses"]:
            poses.append(dict(name=c["name"], kind=c["kind"], viewpoint_id=c["viewpoint_id"],
                              position_xyz_m=position(scan, c["viewpoint_id"]),
                              heading_deg=round(math.degrees(float(c["heading_rad"])) % 360, 4),
                              elevation_deg=round(math.degrees(float(c.get("elevation_rad") or 0)), 4),
                              vfov_deg=c.get("vfov_degrees", 60.0)))
        rec = dict(uid=u, trajectory_id=q["trajectory_id"], scan_id=scan, show_lookback=q["show_lookback"],
                   path_viewpoints=path,
                   moves=[dict(step=a["step_index"], from_viewpoint=a["source_viewpoint"], to_viewpoint=a["target_viewpoint"],
                               distance_m=round(a["distance_m"], 4), travel_heading_deg=round(math.degrees(a["heading_rad"]) % 360, 4))
                          for a in acts],
                   camera_poses=poses,
                   evidence={"observed": "start", "endpoint": poses[-1]["name"]})
        f.write(json.dumps(rec) + "\n"); n_fo += 1

# ---------------- Cross-view Integration
exp = ROOT / "outputs/mvc300_metadata_export_20261009/MVC300_benchmark922_v2"
rows = {r["uid"]: r for r in csv.DictReader(open(exp / "index.csv", encoding="utf-8-sig"))}
n_cv = 0
with open(OUT / "crossview_integration_poses.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for u in mvc_uids:
        r = rows[u]
        assert r["scan_id"] == qidx[u]["scan_id"], u
        cam = json.load(open(exp / r["folder"] / "camera_poses.json", encoding="utf-8"))
        f.write(json.dumps(dict(uid=u, view_group=r["scene"], **cam)) + "\n"); n_cv += 1

print("VI", n_fo, "CV", n_cv, "scans loaded", len(conn))
