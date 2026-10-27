"""Run run_scene.main for every scene whose generated 3D is complete (skips ones already packaged under tag)."""
import json, sys, traceback, concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_scene
REPO = Path(__file__).resolve().parents[2]
def complete_ids():
    sh = run_scene.Shell()
    try:
        sel = json.loads(sh.run(f'cat {run_scene.TASK}/outputs/hy_latest_scenes_20261002/latest_selection.json'))
    finally:
        sh.close()
    return [x['id'] for x in sel['selected_scenes']]
def one(sid, tag):
    if (REPO / 'outputs/hy_estimated_packages_20261002' / tag / sid / 'pipeline_status.json').exists():
        return sid, 'skip'
    try:
        run_scene.main(sid, tag); return sid, 'ok'
    except Exception as exc:
        traceback.print_exc(); return sid, repr(exc)
if __name__ == '__main__':
    tag = sys.argv[1]; ids = sys.argv[2].split(',') if len(sys.argv) > 2 else complete_ids()
    order = [s['id'] for s in json.loads((run_scene.INP / 'scenes.json').read_text())['scenes']]
    ids = sorted(set(ids), key=order.index)
    print('scenes', len(ids), ids, flush=True)
    with cf.ThreadPoolExecutor(2) as ex:
        for sid, st in ex.map(lambda s: one(s, tag), ids):
            print('DONE', sid, st, flush=True)
