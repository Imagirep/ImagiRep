"""Per-question MVC modality assets that match the DISPLAYED image numbering of the frozen 325-question cohort.

Frozen rows: Ready100 (210) + Last50 round 1 (115) model_inputs.jsonl (options/image order already randomized).
Per resource (from outputs/mvc_modalities_20261005/resources/<rid>): oracle depth (+metre scale), oracle panorama.
Per resource, re-encoded to the single-view convention: surface normals with R=camera right, G=camera up,
B=camera forward (a surface facing the camera has low B), black = missing.
Per (resource, displayed order):
  transition text + video   in-place rotation Image 1 -> (Oracle RGB, two-view sets) -> Image 2 (-> Image 3)
  top-down                  labeler renderer (make_dense_topdown) with arrows View k = displayed Image k, plus Oracle
No model calls; source assets read-only.
"""
import json, math, shutil, subprocess, sys
from pathlib import Path
import numpy as np, cv2

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'outputs/mvc_modalities_20261005')); sys.path.insert(0, str(ROOT / 'scripts'))
import build_all_mvc as B  # noqa: E402
from add_mvc001_oracle_states import make_dense_topdown  # noqa: E402
pr = B.pr
SRC = ROOT / 'outputs/mvc_modalities_20261005/resources'
OUT = ROOT / 'outputs/mvc_modalities_20261005/eval_assets'
COHORTS = ['outputs/mvc_ready100_three_settings_20260930', 'outputs/mvc_batch1_last50_scenes_three_settings_20261001']


def rows():
    out = []
    for d in COHORTS:
        for l in open(ROOT / d / 'model_inputs.jsonl', encoding='utf-8'):
            r = json.loads(l); r['cohort'] = Path(d).name; out.append(r)
    return out


def display_order(r):
    m = r['randomization']['source_to_display_image_number']        # source image number -> displayed number
    return tuple(int(src) for src, disp in sorted(m.items(), key=lambda kv: kv[1]))   # displayed 1.. -> source


def turn(a, b):
    d = (b - a + 180) % 360 - 180
    return ('right' if d > 0 else 'left'), abs(d)


def per_resource(rid):
    d = OUT / rid; d.mkdir(parents=True, exist_ok=True)
    m = json.loads((SRC / rid / 'modalities_manifest.json').read_text(encoding='utf-8'))
    for f in ('oracle_depth.png', 'oracle_panorama.png'):
        if not (d / f).exists(): shutil.copy2(SRC / rid / f, d / f)
    if not (d / 'oracle_surface_normals_sv.png').exists():
        n = cv2.imread(str(SRC / rid / 'oracle_surface_normals.png'))           # BGR; B channel = toward camera
        valid = n.any(axis=2)
        n[..., 0] = np.where(valid, 255 - n[..., 0], 0)                         # -> camera-forward component
        cv2.imwrite(str(d / 'oracle_surface_normals_sv.png'), n)
    return m


def per_order(rid, order, res, m):
    key = ''.join(map(str, order)); d = OUT / rid / f'order_{key}'
    if (d / 'done.json').exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    s = B.spec(rid, res[rid])
    head = {n: h for n, h, _, _ in s['views']}; pitch = s['views'][0][2]
    disp = [(i + 1, head[src]) for i, src in enumerate(order)]                  # (displayed number, heading)
    two = len(order) == 2
    stops = [(disp[0][1], 'Image 1')] + ([(s['oh'], 'Oracle RGB')] if two else []) + [(h, f'Image {k}') for k, h in disp[1:]]
    lines = ['All numbered images were taken from the same position; between them the camera only rotates in place.',
             f'The view is tilted {abs(pitch):.1f} degrees below level throughout.', '', 'Start facing as in Image 1.']
    for (a, _), (b, name) in zip(stops, stops[1:]):
        t, ang = turn(a, b); lines.append(f'Turn {t} {ang:.1f} degrees in place to face as in {name}.')
    if not two:
        dp = s['opose'][:3, 3] - s['pose'][:3, 3]; dist = float(np.linalg.norm(dp[:2]))
        t, ang = turn(disp[0][1], math.degrees(math.atan2(float(dp[0]), float(dp[1]))))
        lines += ['', f'Oracle RGB was taken from a different position, {dist:.2f} m away, located {ang:.1f} degrees to the {t} of the Image 1 viewing direction.']
    (d / 'view_transition.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    erp, *_ = pr.panorama(s['scan'], s['vp'])
    seq = [stops[0][0]] * 12
    for (a, _), (b, _) in zip(stops, stops[1:]):
        dd = (b - a + 180) % 360 - 180; n = max(12, int(math.ceil(abs(dd) / 3)))
        seq += [a + dd * (3 * x * x - 2 * x ** 3) for x in np.linspace(0, 1, n + 1)[1:]] + [b] * 12
    fr = d / '_frames'; fr.mkdir(exist_ok=True)
    for i, h in enumerate(seq):
        cv2.imwrite(str(fr / f'{i:04d}.jpg'), cv2.cvtColor(pr.extract(erp, s['pose'], h, pitch, 1280, 960)[0], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '12', '-i', str(fr / '%04d.jpg'), '-c:v', 'libx264', '-crf', '20',
                    '-pix_fmt', 'yuv420p', str(d / 'view_transition.mp4')], check=True)
    first = cv2.imread(str(fr / '0000.jpg')); shutil.rmtree(fr)
    # top-down with displayed numbering; MatterSim heading -> map arrow uses the same convention as the labeler
    cam = s['pose'][:3, 3]
    arrows = {f'View {k}': (cam, math.radians(h)) for k, h in disp}
    arrows['Oracle'] = (s['opose'][:3, 3], math.radians(s['oh']))
    tmp = d / '_td'; tmp.mkdir(exist_ok=True)
    make_dense_topdown(s['scan'], cam, arrows, tmp).replace(d / 'topdown.png'); shutil.rmtree(tmp, ignore_errors=True)
    # check: first video frame equals the displayed Image 1 (labeler image of source number order[0])
    img1 = cv2.imread(str(next(p for n, *_, p in s['views'] if n == order[0])))
    mae = float(np.abs(cv2.resize(first, (img1.shape[1], img1.shape[0])).astype(float) - img1.astype(float)).mean())
    (d / 'done.json').write_text(json.dumps(dict(order=order, stops=[x for _, x in stops], frames=len(seq), first_frame_vs_displayed_image1_mae=mae)), encoding='utf-8')
    return d


def main():
    res = B.resources(); R = rows(); seen = {}
    for r in R:
        rid = r['resource_id']; order = display_order(r)
        assert sorted(order) == list(range(1, len(order) + 1)), r['question_id']
        if rid not in seen:
            seen[rid] = per_resource(rid)
        d = per_order(rid, order, res, seen[rid])
        print(r['question_id'], rid, order, flush=True)
    print('rows', len(R), 'resources', len(seen))


if __name__ == '__main__':
    main()
