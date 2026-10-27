"""Question-only (text_only) accuracy on the canonical 622 single-view rows.

Reuses prior Gemini 3.7 Flash text_only results whose question text, shuffled choices and gold answer are identical
(reuse_plan.json from plan_reuse.py); evaluates the remaining questions with the frozen runner snapshot
(same system prompt, temperature 0, HIGH thinking, A-D enum, 8192 tokens), strictly serial.
"""
import importlib.util, json, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
MODEL = 'gemini-3.7-flash'
spec = importlib.util.spec_from_file_location('base', ROOT / 'docs/experiments/fo_batch2_eight_conditions_20260927/runner_snapshot.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
canon = {json.loads(l)['question_id']: json.loads(l) for l in open(ROOT / 'docs/experiments/fo_estimated_seven_conditions_622_v5_20261005/canonical_622_rows.jsonl', encoding='utf-8')}
plan = json.loads((HERE / 'reuse_plan.json').read_text())
out = HERE / 'results/per_sample/text_only'; out.mkdir(parents=True, exist_ok=True)
base.load_env(ROOT / '.env')
todo = [k.split('|')[0] for k in plan['todo']]
for i, q in enumerate(todo, 1):
    p = out / f'{q}.json'
    if p.exists():
        continue
    row = canon[q]
    for attempt in range(1, 7):
        try:
            t0 = time.monotonic(); pred, usage, raw = base.generate_gemini(MODEL, row, 'text_only')
            if pred not in row['answer_choices']:
                raise ValueError('invalid enum')
            break
        except Exception as exc:  # never persist exception text (may contain credentials)
            print(f'Retry {q}: {type(exc).__name__}', flush=True)
            if attempt == 6:
                raise SystemExit('exhausted retries')
            time.sleep(min(45, 2 ** attempt))
    rec = dict(question_id=q, condition='text_only', model=MODEL, question=row['question'], answer_choices=row['answer_choices'],
               gold_answer=row['gold_answer'], prediction=pred, correct=pred == row['gold_answer'], system_prompt=base.SYSTEM_PROMPT,
               temperature=0, reasoning_effort='high', max_output_tokens=8192, usage_metadata=usage, raw_response=raw, attempts=attempt,
               request_duration_seconds=time.monotonic() - t0, completed_at=datetime.now().astimezone().isoformat(timespec='seconds'))
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[{i}/{len(todo)}] OK {q} {pred}', flush=True)
# summary over all 622
res = {}
for k, v in plan['reused'].items():
    res[k.split('|')[0]] = dict(correct=v['correct'], source=v['source'])
for p in out.glob('*.json'):
    r = json.loads(p.read_text(encoding='utf-8')); res[r['question_id']] = dict(correct=r['correct'], source='fo_622_question_only_trajectory_20261006')
assert set(res) == set(canon), (len(res), len(canon))
n = sum(v['correct'] for v in res.values())
summary = dict(condition='text_only', n=len(res), correct=n, accuracy=n / len(res), reused=len(plan['reused']), new=len(list(out.glob('*.json'))))
(HERE / 'results/summary_text_only.json').write_text(json.dumps(summary, indent=2))
(HERE / 'results/text_only_by_question.json').write_text(json.dumps(res, indent=1))
print(json.dumps(summary))
