"""Oracle trajectory RGB (start_full_trajectory) on the canonical 622 single-view rows.

Input = start RGB + every intermediate step RGB + endpoint RGB (+ separate oracle lookback when show_lookback), captioned
exactly as in the frozen runner snapshot (final image uses the FINAL VIEW prompt). Same model and settings as all other
conditions: Gemini 3.7 Flash, temperature 0, HIGH thinking, A-D enum, 8192 tokens; strictly serial.
Reuse: a prior gemini-3.7-flash start_full_trajectory record is reused only if question text, shuffled choices, gold and
the exact ordered image list + captions are identical. No exception text is ever persisted.
Usage: python run_trajectory.py --plan | --execute
"""
import glob, hashlib, importlib.util, json, os, sys, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
MODEL, COND = 'gemini-3.7-flash', 'start_full_trajectory'
spec = importlib.util.spec_from_file_location('base', ROOT / 'docs/experiments/fo_batch2_eight_conditions_20260927/runner_snapshot.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
canon = {json.loads(l)['question_id']: json.loads(l) for l in open(ROOT / 'docs/experiments/fo_estimated_seven_conditions_622_v5_20261005/canonical_622_rows.jsonl', encoding='utf-8')}
OUT = HERE / 'results/per_sample' / COND


def norm(p): return str(p).replace(chr(92), '/').split('IndividualResearch/')[-1]
def local(p): return ROOT / norm(p)
def key(row):
    imgs = [norm(i['path']) for i in base.selected_images(row, COND)]
    return json.dumps([row['question'], row['answer_choices'], row['gold_answer'], imgs, base.image_prompts(row, COND)], sort_keys=True)


def plan():
    want = {q: key(r) for q, r in canon.items()}
    for q, r in canon.items():           # every image must exist and the final image must be the declared final view
        imgs = base.selected_images(r, COND)
        assert imgs[0]['kind'] == 'start' and all(local(i['path']).exists() for i in imgs), q
        assert imgs[-1]['kind'] == ('lookback' if r['show_lookback'] else 'endpoint'), q
    found = {}
    for p in glob.glob(str(ROOT / 'docs/experiments/**/results_by_sample.jsonl'), recursive=True):
        for l in open(p, encoding='utf-8-sig'):
            if not l.strip(): continue
            r = json.loads(l); q = r.get('question_id')
            if r.get('condition') != COND or q not in want or q in found or r.get('model') != MODEL: continue
            if r.get('system_prompt') != base.SYSTEM_PROMPT or 'evidence_images' not in r or r.get('prediction') not in r.get('answer_choices', {}): continue
            r.setdefault('show_lookback', canon[q]['show_lookback'])
            if key(r) == want[q]:
                found[q] = dict(source=os.path.relpath(p, ROOT), prediction=r['prediction'], correct=r['prediction'] == canon[q]['gold_answer'])
    todo = sorted(set(canon) - set(found))
    n_img = [len(base.selected_images(canon[q], COND)) for q in todo]
    (HERE / 'trajectory_plan.json').write_text(json.dumps(dict(reused=found, todo=todo), indent=1))
    print(f'reusable {len(found)}  todo {len(todo)}  images/request min {min(n_img)} max {max(n_img)} mean {sum(n_img)/len(n_img):.1f}')


def execute():
    plan_ = json.loads((HERE / 'trajectory_plan.json').read_text())
    base.load_env(ROOT / '.env'); OUT.mkdir(parents=True, exist_ok=True)
    todo = plan_['todo']
    for i, q in enumerate(todo, 1):
        p = OUT / f'{q}.json'
        if p.exists(): continue
        row = canon[q]
        for attempt in range(1, 7):
            try:
                t0 = time.monotonic(); pred, usage, raw = base.generate_gemini(MODEL, row, COND)
                if pred not in row['answer_choices']: raise ValueError('invalid enum')
                break
            except Exception as exc:
                print(f'Retry {q}: {type(exc).__name__}', flush=True)
                if attempt == 6: raise SystemExit('exhausted retries')
                time.sleep(min(45, 2 ** attempt))
        imgs = base.selected_images(row, COND)
        rec = dict(question_id=q, condition=COND, model=MODEL, question=row['question'], answer_choices=row['answer_choices'], gold_answer=row['gold_answer'],
                   show_lookback=row['show_lookback'], prediction=pred, correct=pred == row['gold_answer'], system_prompt=base.SYSTEM_PROMPT,
                   image_prompts=base.image_prompts(row, COND), evidence_kinds=[x['kind'] for x in imgs],
                   evidence=[dict(path=norm(x['path']), sha256=hashlib.sha256(local(x['path']).read_bytes()).hexdigest()) for x in imgs],
                   temperature=0, reasoning_effort='high', max_output_tokens=8192, usage_metadata=usage, raw_response=raw, attempts=attempt,
                   request_duration_seconds=time.monotonic() - t0, completed_at=datetime.now().astimezone().isoformat(timespec='seconds'))
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
        (HERE / 'trajectory_status.json').write_text(json.dumps(dict(state='running', done=i, total=len(todo), updated=rec['completed_at'])))
        print(f'[{i}/{len(todo)}] OK {q} {pred}', flush=True)
    res = {q: v['correct'] for q, v in plan_['reused'].items()}
    for p in OUT.glob('*.json'):
        r = json.loads(p.read_text(encoding='utf-8')); res[r['question_id']] = r['correct']
    assert set(res) == set(canon), (len(res), len(canon))
    n = sum(res.values())
    summary = dict(condition=COND, n=len(res), correct=n, accuracy=n / len(res), reused=len(plan_['reused']), new=len(list(OUT.glob('*.json'))))
    (HERE / 'results/summary_trajectory.json').write_text(json.dumps(summary, indent=2))
    (HERE / 'results/trajectory_by_question.json').write_text(json.dumps(res, indent=1))
    (HERE / 'trajectory_status.json').write_text(json.dumps(dict(state='complete', **summary)))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    plan() if '--plan' in sys.argv else execute()
