"""Export the MVC 300 multi-view part of Benchmark 922 v2 as one folder per question (2026-10-09).

Source of truth: docs/experiments/benchmark922_v2_20261008/questions.jsonl (+ SHA256SUMS). Every image/video that v2
references by <sha256[:16]>.<ext> is located in the local MVC asset folders by content hash and checked against the
full v2 SHA256SUMS entry, so the exported files are byte-identical to what the models were shown.
Camera poses come from the labeler render manifests (build_all_mvc.spec), re-indexed to the DISPLAYED image numbers.
Usage: python export_mvc300.py <out_dir>      (writes <out_dir>/MVC300_benchmark922_v2/ ; zip separately)
No model calls; source assets read-only.
"""
import csv, hashlib, json, math, shutil, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
V2 = ROOT / 'docs/experiments/benchmark922_v2_20261008'
sys.path.insert(0, str(ROOT / 'outputs/mvc_modalities_20261005'))
import build_all_mvc as B  # noqa: E402
pr = B.pr

COHORTS = {'ready100': ROOT / 'outputs/mvc_ready100_three_settings_20260930',
           'last50': ROOT / 'outputs/mvc_batch1_last50_scenes_three_settings_20261001',
           'mvc131': ROOT / 'outputs/mvc_mvc131_supplement_20261006'}
OUT = Path(sys.argv[1]) / 'MVC300_benchmark922_v2'
W, H = 1280, 960

# condition -> (caption prefix that identifies the modality part, exported file stem)
MODALITY = {'oracle_topdown': 'topdown', 'oracle_text': 'view_transition_text', 'oracle_video': 'view_transition_video',
            'oracle_depth': 'oracle_depth', 'oracle_surface_normals': 'oracle_surface_normals', 'oracle_panorama': 'oracle_panorama'}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cam(position, heading, pitch):
    """Camera-to-world 4x4, OpenCV axes (x right, y down, z forward), MP3D world frame (metres, z up)."""
    phys = -pr.world_basis(heading, pitch)            # columns: right, up, forward (world)
    R = np.stack([phys[:, 0], -phys[:, 1], phys[:, 2]], axis=1)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = position
    return dict(position_xyz_m=[round(float(v), 6) for v in position], heading_deg=round(float(heading) % 360, 6),
                pitch_deg=round(float(pitch), 6), forward_xyz=[round(float(v), 6) for v in phys[:, 2]],
                cam_to_world_opencv=[[round(float(v), 6) for v in row] for row in T])


def main():
    sums = {}
    for l in open(V2 / 'SHA256SUMS', encoding='utf-8'):
        h, n = l.split(maxsplit=1); sums[n.strip().split('/')[-1]] = h
    qs = [json.loads(l) for l in open(V2 / 'questions.jsonl', encoding='utf-8')]
    qs = [q for q in qs if q['benchmark'] == 'MVC300']
    assert len(qs) == 300

    rows = {}
    for c, d in COHORTS.items():
        for l in open(d / 'model_inputs.jsonl', encoding='utf-8'):
            r = json.loads(l); rows[f"{c}__{r['question_id']}"] = r
    res = B.resources()
    sup = json.load(open(COHORTS['mvc131'] / 'resources.json', encoding='utf-8'))
    for x in (sup['items'] if isinstance(sup, dict) and 'items' in sup else sup if isinstance(sup, list) else sup.values()):
        res.setdefault(x['id'], x)

    if OUT.exists(): shutil.rmtree(OUT)
    (OUT / 'scenes').mkdir(parents=True)
    index, n_files = [], 0
    for q in qs:
        r = rows[q['uid']]; rid = r['resource_id']; cohort = q['uid'].split('__')[0]
        # candidate local files for this question, indexed by sha256[:16]
        cands = [ROOT / a['path'] for s in r['settings'].values() for a in s]
        cands += [ROOT / r['topdown_image']] if r.get('topdown_image') else []
        ea = ROOT / 'outputs/mvc_modalities_20261005/eval_assets' / rid
        cands += [p for p in ea.rglob('*') if p.is_file()] + [p for p in (B.OUT / rid).glob('*') if p.is_file()]
        local = {}
        for p in cands:
            if p.exists(): local.setdefault(sha(p)[:16], p)

        qdir = OUT / 'scenes' / rid / q['uid'].split('__')[1]
        qdir.mkdir(parents=True)
        names, files = {}, {}

        def place(ref, stem):
            h, ext = ref.rsplit('.', 1)
            src = local[h]; full = sha(src)
            assert full == sums[ref], (q['uid'], ref)
            name = f'{stem}.{ext}'
            if ref in names: return names[ref]
            shutil.copy2(src, qdir / name); names[ref] = name; files[name] = dict(sha256=full, benchmark_v2_name=ref)
            return name

        # observed numbered images and oracle RGB (labels taken from the oracle condition)
        cond = q['conditions']; label = None
        for p in cond['oracle']:
            if isinstance(p, dict):
                label = p['text'].strip().rstrip(':')
            else:
                place(p, 'oracle_rgb' if label == 'Oracle RGB' else label.lower().replace(' ', '_'))
        n_obs = sum(isinstance(p, str) for p in cond['observed'])
        evidence = {}
        for c, parts in cond.items():
            ev = []
            for p in parts:
                if isinstance(p, dict): ev.append({'text': p['text']})
                else:
                    ext = p.rsplit('.', 1)[1]
                    ev.append({'video' if ext == 'mp4' else 'image': names.get(p) or place(p, MODALITY[c])})
            evidence[c] = ev
        txt = [p['text'] for p in cond['oracle_text'][len(cond['oracle']):]]
        (qdir / 'view_transition_text.txt').write_text(''.join(txt), encoding='utf-8')

        # camera poses, re-indexed to displayed image numbers
        s = B.spec(rid, res[rid])
        src2disp = {int(k): v for k, v in r['randomization']['source_to_display_image_number'].items()}
        views = {src2disp[n]: (h, p) for n, h, p, _ in s['views']}
        assert sorted(views) == list(range(1, n_obs + 1)), q['uid']
        man = json.loads((B.OUT / rid / 'modalities_manifest.json').read_text(encoding='utf-8'))
        fx, fy = (W / 2) / math.tan(math.radians(pr.HFOV) / 2), (H / 2) / math.tan(math.radians(pr.VFOV) / 2)
        poses = dict(
            scan_id=s['scan'], schema=man.get('schema'),
            observation_viewpoint_id=s['vp'], oracle_viewpoint_id=s['ovp'],
            same_position_for_all_numbered_images=True,
            oracle_at_same_position=bool(s['ovp'] == s['vp']),
            intrinsics=dict(width=W, height=H, hfov_deg=pr.HFOV, vfov_deg=pr.VFOV, fx=round(fx, 4), fy=round(fy, 4),
                            cx=(W - 1) / 2, cy=(H - 1) / 2, model='pinhole, no distortion'),
            views={f'Image {k}': cam(s['pose'][:3, 3], *views[k]) for k in sorted(views)},
            oracle_rgb=cam(s['opose'][:3, 3], s['oh'], s['op']),
            depth_colour_range_m=[man['depth']['near_m'], man['depth']['far_m']] if 'depth' in man else None,
            source_image_number_for_displayed={str(v): k for k, v in src2disp.items()})
        (qdir / 'camera_poses.json').write_text(json.dumps(poses, indent=1), encoding='utf-8')

        meta = dict(uid=q['uid'], question_id=r['question_id'], scene_resource_id=rid, cohort=cohort, scan_id=s['scan'],
                    category=q['category'], question=r['question'], options=r['answer_choices'], gold=q['gold'],
                    valid=q['valid'], num_numbered_images=n_obs,
                    randomization=r['randomization'], system_prompt=q['system'], question_prompt=q['question_prompt'],
                    conditions_evidence=evidence, files=files)
        (qdir / 'question.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding='utf-8')
        n_files += len(files)
        index.append(dict(uid=q['uid'], scene=rid, question_id=r['question_id'], cohort=cohort, scan_id=s['scan'],
                          category=q['category'], num_numbered_images=n_obs, gold=q['gold'],
                          question=r['question'], folder=f"scenes/{rid}/{r['question_id']}"))

    with open(OUT / 'index.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(index[0])); w.writeheader(); w.writerows(index)
    print(len(index), 'questions,', len({x['scene'] for x in index}), 'scenes,', n_files, 'files (all sha256-verified)')


if __name__ == '__main__':
    main()
