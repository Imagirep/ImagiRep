"""MVC (multi-view) oracle-modality evaluation on the frozen 325-question cohort (Ready100 210 + Last50 round-1 115).

Mirrors the single-view 2026-09-27 modality protocol: the base input is the existing oracle_rgb setting (numbered
observed images + Oracle RGB, frozen option/image randomization, frozen system prompt and additional_rgb prompt);
each condition appends ONE oracle-anchored representation with a caption adapted from the single-view captions
(FINAL VIEW -> Oracle RGB). Assets match the displayed image numbering (build_eval_assets.py).
Model/settings identical to the cohort runs: Gemini 3.7 Flash, temperature 0, HIGH thinking, A-D enum, 8192 tokens.
STRICT serial by default; 2026-10-06 resumed with --workers 24 (user-approved concurrency, same payload/config). original_only and oracle_rgb are reused from the cohort runs (not re-run).
Usage: python run_mvc_modalities.py --preflight | --execute [--workers N] | --summarize
"""
import hashlib, importlib.util, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
COHORTS = {'ready100': ROOT / 'outputs/mvc_ready100_three_settings_20260930', 'last50': ROOT / 'outputs/mvc_batch1_last50_scenes_three_settings_20261001'}
spec = importlib.util.spec_from_file_location('mvc_base', COHORTS['last50'] / 'base_runner_snapshot.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
types = base.types
_tl = __import__('threading').local()
def _client_with_timeout():  # 300 s per-request timeout so a stalled request cannot hang a worker forever
    c = getattr(_tl, 'c', None)
    if c is None:
        import os; c = _tl.c = base.genai.Client(api_key=os.environ['GEMINI_API_KEY'], http_options=types.HttpOptions(timeout=300_000))
    return c
base.client = _client_with_timeout
ASSETS = ROOT / 'outputs/mvc_modalities_20261005/eval_assets'
MODEL = 'gemini-3.7-flash'
CONDS = ('oracle_rgb_topdown', 'oracle_rgb_text', 'oracle_rgb_video', 'oracle_rgb_depth', 'oracle_rgb_surface_normals', 'oracle_rgb_panorama')
REUSED = ('original_only', 'oracle_rgb')
TYPES = ('cross_view_closest', 'cross_view_farthest', 'ordering')
CAPTIONS = {
    'topdown': ("ADDITIONAL TOP-DOWN MAP:\n"
                "The circle marks the camera position of the numbered images. Arrows labeled View 1, View 2 (and View 3) "
                "show the facing directions of Image 1, Image 2 (and Image 3); the arrow labeled Oracle shows the facing "
                "direction of Oracle RGB from its camera position. Use this map together with the numbered images and Oracle RGB."),
    'text': ("ADDITIONAL TEXT VIEW TRANSITION:\n"
             "The description gives the in-place camera rotations between the numbered images and Oracle RGB. "
             "Turns are relative to the current view. Use it together with the numbered images and Oracle RGB."),
    'video': ("ADDITIONAL VIEW-TRANSITION VIDEO:\n"
              "The video shows the camera rotating in place from Image 1 to the last numbered image, passing through the "
              "Oracle RGB view when it lies between them. Use it together with the numbered images and Oracle RGB."),
    'depth': ("ORACLE DEPTH MAP (same camera pose and field of view as Oracle RGB):\n"
              "Colors encode camera-forward distance (z-depth), not object color. Red/orange is nearer, green is intermediate, "
              "and blue/purple is farther; black means missing depth, not a nearby surface. This image's color scale runs from "
              "{near_m:.3f} m (near/red) to {far_m:.3f} m (far/purple), with values outside this range clipped. "
              "Combine this map with the numbered images and Oracle RGB."),
    'surface_normals': ("ORACLE SURFACE NORMAL MAP (same camera pose and field of view as Oracle RGB):\n"
                        "Colors encode surface orientation, not object color or distance. RGB channels encode normal components "
                        "along camera-right (R), camera-up (G), and camera-forward (B): 0 = -1, about 128 = 0, and 255 = +1. "
                        "Higher R/G/B means a stronger rightward/upward/forward component; a surface facing the camera has low B. "
                        "Black means missing geometry. Combine this map with the numbered images and Oracle RGB."),
    'panorama': ("ORACLE PANORAMA (360-degree RGB at the Oracle RGB position):\n"
                 "The panorama is centered on the Oracle RGB facing direction. At mid-height, the center is forward, "
                 "one-quarter of the width is left, three-quarters is right, and the two joined edges are behind. "
                 "Top/bottom look up/down; colors are natural RGB, not geometric labels. "
                 "Combine it with the numbered images and Oracle RGB."),
}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows():
    out = []
    for c, d in COHORTS.items():
        for l in open(d / 'model_inputs.jsonl', encoding='utf-8'):
            r = json.loads(l); r['cohort'] = c; r['uid'] = f"{c}__{r['question_id']}"; out.append(r)
    return out


def order_dir(r):
    m = r['randomization']['source_to_display_image_number']
    order = ''.join(src for src, disp in sorted(m.items(), key=lambda kv: kv[1]))
    return ASSETS / r['resource_id'] / f'order_{order}'


def modality_asset(r, kind):
    rd, od = ASSETS / r['resource_id'], order_dir(r)
    return {'topdown': od / 'topdown.png', 'text': od / 'view_transition.txt', 'video': od / 'view_transition.mp4',
            'depth': rd / 'oracle_depth.png', 'surface_normals': rd / 'oracle_surface_normals_sv.png', 'panorama': rd / 'oracle_panorama.png'}[kind]


def depth_scale(r):
    m = json.loads((ROOT / 'outputs/mvc_modalities_20261005/resources' / r['resource_id'] / 'modalities_manifest.json').read_text(encoding='utf-8'))
    return dict(near_m=m['depth']['near_m'], far_m=m['depth']['far_m'])


def caption(r, kind):
    return CAPTIONS[kind].format(**depth_scale(r)) if kind == 'depth' else CAPTIONS[kind]


def prompt_text(r):
    p = (COHORTS[r['cohort']] / 'prompts/additional_rgb.txt').read_text(encoding='utf-8').strip()
    return f"{p}\n\nQuestion: {r['question']}\n{base.option_text(r['answer_choices'])}\nAnswer:"


def parts(r, cond):
    kind = cond.removeprefix('oracle_rgb_'); out = []
    for a in r['settings']['oracle_rgb']:
        out += [base.text_part(a['label'] + ':\n'), base.image_part(ROOT / a['path'])]
    p = modality_asset(r, kind)
    if kind == 'text':
        out.append(base.text_part(caption(r, kind) + '\n' + p.read_text(encoding='utf-8')))
    elif kind == 'video':
        out += [base.text_part(caption(r, kind) + '\n'),
                types.Part(inline_data=types.Blob(mime_type='video/mp4', data=p.read_bytes()), video_metadata=types.VideoMetadata(fps=12))]
    else:
        out += [base.text_part(caption(r, kind) + '\n'), base.image_part(p)]
    out.append(base.text_part(prompt_text(r)))
    return out


def signature(r, cond):
    kind = cond.removeprefix('oracle_rgb_')
    return hashlib.sha256(json.dumps(dict(system=(COHORTS[r['cohort']] / 'prompts/system.txt').read_text(encoding='utf-8').strip(),
        prompt=prompt_text(r), caption=caption(r, kind), images=[(a['label'], sha(ROOT / a['path'])) for a in r['settings']['oracle_rgb']],
        modality=sha(modality_asset(r, kind))), sort_keys=True).encode()).hexdigest()


def preflight():
    R = rows(); missing = []
    for r in R:
        assert len({x['uid'] for x in R}) == len(R)
        for c in CONDS:
            p = modality_asset(r, c.removeprefix('oracle_rgb_'))
            if not p.exists(): missing.append(str(p))
        for c in REUSED:
            assert (COHORTS[r['cohort']] / 'results/per_sample' / c / f"{r['question_id']}.json").exists(), (r['uid'], c)
    status = dict(questions=len(R), conditions=CONDS, planned_calls=len(R) * len(CONDS), missing_assets=len(missing), model_calls=0,
                  types=dict(Counter(r['question_type'] for r in R)), cohorts=dict(Counter(r['cohort'] for r in R)))
    (HERE / 'preflight.json').write_text(json.dumps(status, indent=2)); print(json.dumps(status)); return not missing


def execute():
    # 2026-10-06: optional --workers N (default 1 = original strict serial). Requests are independent; each result is
    # written to its own per-sample file, so concurrency changes throughput only, not the request payload/config.
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 1
    assert preflight(), 'missing assets'
    base.load_env(ROOT / '.env'); R = rows(); total = len(R) * len(CONDS)
    jobs = [(k, c, r) for k, (c, r) in enumerate(((c, r) for c in CONDS for r in R), 1)
            if not (HERE / 'results/per_sample' / c / f"{r['uid']}.json").exists()]
    print(f'pending {len(jobs)}/{total} workers={workers}', flush=True)
    lock = threading.Lock(); failed = []
    def job(a):
        try: run_one(*a, total, lock)
        except SystemExit: failed.append(a[2]['uid'] + '|' + a[1])
    with ThreadPoolExecutor(workers) as ex: list(ex.map(job, jobs))
    if failed:
        print(f'FAILED {len(failed)} (re-run --execute to retry)', flush=True)
        (HERE / 'execution_status.json').write_text(json.dumps(dict(state='incomplete', failed=len(failed), total=total))); return
    summarize()
    (HERE / 'execution_status.json').write_text(json.dumps(dict(state='complete', total=total)))


def run_one(k, c, r, total, lock):
    p = HERE / 'results/per_sample' / c / f"{r['uid']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    system = (COHORTS[r['cohort']] / 'prompts/system.txt').read_text(encoding='utf-8').strip()
    valid = sorted(str(x).upper() for x in r['answer_choices'])
    for attempt in range(1, 4):
        try:
            t0 = time.monotonic(); pred, usage = base.generate(MODEL, system, parts(r, c), valid, 6); break
        except Exception as exc:
            print(f'Retry {r["uid"]} {c}: {type(exc).__name__}', flush=True)   # never persist exception text
            if attempt == 3: raise SystemExit('exhausted retries')
            time.sleep(60)
    rec = dict(uid=r['uid'], cohort=r['cohort'], question_id=r['question_id'], resource_id=r['resource_id'], question_type=r['question_type'],
               condition=c, model=MODEL, temperature=0, thinking_level='HIGH', question=r['question'], answer_choices=r['answer_choices'],
               gold_answer=r['gold_answer'], prediction=pred, correct=pred == str(r['gold_answer']).upper(), signature=signature(r, c),
               caption=caption(r, c.removeprefix('oracle_rgb_')), modality_asset=str(modality_asset(r, c.removeprefix('oracle_rgb_')).relative_to(ROOT)),
               usage_metadata=usage, request_duration_seconds=time.monotonic() - t0, completed_at=datetime.now().astimezone().isoformat(timespec='seconds'))
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8'); tmp.replace(p)
    with lock:
        (HERE / 'execution_status.json').write_text(json.dumps(dict(state='running', index=k, total=total, condition=c, uid=r['uid'], updated=rec['completed_at'])))
        print(f'[{k}/{total}] OK {r["uid"]} {c} {pred}', flush=True)


def summarize():
    R = rows(); look = {}
    for r in R:
        for c in REUSED:
            x = json.loads((COHORTS[r['cohort']] / 'results/per_sample' / c / f"{r['question_id']}.json").read_text(encoding='utf-8'))
            assert x['gold_answer'] == r['gold_answer'] and x['answer_choices'] == r['answer_choices']
            look[(r['uid'], c)] = bool(x['correct'])
        for c in CONDS:
            p = HERE / 'results/per_sample' / c / f"{r['uid']}.json"
            if p.exists():
                x = json.loads(p.read_text(encoding='utf-8')); assert x['signature'] == signature(r, c); look[(r['uid'], c)] = bool(x['correct'])
    allc = REUSED + CONDS
    def acc(sel, c):
        v = [look[(r['uid'], c)] for r in sel if (r['uid'], c) in look]
        return dict(n=len(v), correct=sum(v), accuracy=sum(v) / len(v) if v else None)
    def pos(sel, c):
        v = [look[(r['uid'], 'oracle_rgb')] or look[(r['uid'], c)] for r in sel if (r['uid'], c) in look]
        return dict(n=len(v), correct=sum(v), accuracy=sum(v) / len(v) if v else None)
    groups = {'all': R, **{t: [r for r in R if r['question_type'] == t] for t in TYPES}}
    out = dict(accuracy={g: {c: acc(s, c) for c in allc} for g, s in groups.items()},
               positive_only={g: {c: pos(s, c) for c in CONDS} for g, s in groups.items()},
               complete=all((r['uid'], c) in look for r in R for c in allc))
    (HERE / 'results').mkdir(exist_ok=True); (HERE / 'results/summary.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({g: {c: round(100 * v['accuracy'], 1) if v['accuracy'] is not None else None for c, v in d.items()} for g, d in out['accuracy'].items()}, indent=1))
    return out


if __name__ == '__main__':
    {'--preflight': preflight, '--execute': execute, '--summarize': summarize}[sys.argv[1]]()
