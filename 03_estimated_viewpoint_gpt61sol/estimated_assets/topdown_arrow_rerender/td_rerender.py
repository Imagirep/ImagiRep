"""Server side: for every job, reproduce the old topdown (must be pixel-identical to the asset), then write the new
larger-arrow topdown to new/<asset>. Nothing existing is modified. Usage: python td_rerender.py jobs.json new/ [procs]"""
import json, os, sys
from collections import defaultdict
import multiprocessing as mp
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerender_topdown_arrow import load_mesh_vertices, draw

JOBS, NEW = sys.argv[1], sys.argv[2]


def scene_jobs(args):
    scene, js = args
    verts, eye_units = load_mesh_vertices(scene)
    out = []
    for j in js:
        kfs = json.load(open(j['plan']))['keyframes']
        old = draw(verts, eye_units, j['eye'], kfs, kfs[-1], new=False)
        cur = cv2.imread(j['ref'])
        if cur is None or cur.shape != old.shape or not np.array_equal(cur, old):
            out.append((j['asset'], 'MISMATCH', None if cur is None else int(np.abs(cur.astype(int) - old).sum()))); continue
        cv2.imwrite(os.path.join(NEW, j['asset']), draw(verts, eye_units, j['eye'], kfs, kfs[-1], new=True))
        out.append((j['asset'], 'ok', 0))
    return out


if __name__ == '__main__':
    os.makedirs(NEW, exist_ok=True)
    by = defaultdict(list)
    for j in json.load(open(JOBS)):
        by[j['scene']].append(j)
    res = []
    lim = int(os.environ.get('LIMIT_SCENES', '0')) or None
    items = list(by.items())[:lim]
    with mp.get_context('spawn').Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 16) as p:
        for r in p.imap_unordered(scene_jobs, items):
            res += r
    bad = [r for r in res if r[1] != 'ok']
    json.dump(res, open(os.path.join(NEW, '..', 'rerender_result.json'), 'w'))
    print(json.dumps({'ok': len(res) - len(bad), 'mismatch': len(bad), 'examples': bad[:5]}))
