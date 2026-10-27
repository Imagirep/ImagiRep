"""Run navigator on a list of questions in parallel; each job gets its own remote env."""
import json, os, sys, concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from navigator import Navigator
REPO = Path(__file__).resolve().parents[2]
SC = REPO / 'outputs/hy_autonomy_200_20261002/inputs'
LATEST = '<external>/hyworld2_nlp26_full_20261001/outputs/hy_latest_scenes_20261002/'
def job(sid, q, tag):
    out = REPO / 'outputs/hy_nav_pilot_20261002' / tag / q['id']
    eye = None
    if os.environ.get('NAV_SCALE') == 'moge':   # per-scene scale from the initial image (MoGe) vs generated depth
        f = REPO / 'outputs/hy_nav_pilot_20261002/scale' / f'{sid}.json'
        if not f.exists():
            raise RuntimeError('missing MoGe scale for ' + sid)
        eye = json.loads(f.read_text())['implied_eye_height_m']
    return Navigator(q['id'], SC / 'images' / f'{sid}.png', q['movement_condition'], LATEST + sid, out, eye_m=eye).run()
if __name__ == '__main__':
    tag, ids = sys.argv[1], sys.argv[2].split(',')
    m = json.loads((SC / 'scenes.json').read_text())
    work = [(s['id'], q) for s in m['scenes'] for q in s['questions'] if q['id'] in ids]
    with cf.ThreadPoolExecutor(4) as ex:
        for r in ex.map(lambda a: job(*a, tag), work):
            print(r['question_id'], r['status'], r.get('error', ''), round(r['seconds']), flush=True)
