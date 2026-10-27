"""Which canonical-622 (question, condition) pairs already have a valid Gemini 3.7 Flash result. No model calls."""
import glob, hashlib, json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'docs/experiments/fo_batch2_eight_conditions_20260927'))
import importlib.util
spec = importlib.util.spec_from_file_location('base', ROOT / 'docs/experiments/fo_batch2_eight_conditions_20260927/runner_snapshot.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
CONDS = ('text_only',)
canon = {json.loads(l)['question_id']: json.loads(l) for l in open(ROOT / 'docs/experiments/fo_estimated_seven_conditions_622_v5_20261005/canonical_622_rows.jsonl', encoding='utf-8')}
def norm(p): return str(p).replace(chr(92), '/').split('IndividualResearch/')[-1]
def key(row, c):
    imgs = [norm(i['path']) for i in base.selected_images(row, c)]
    return json.dumps([row['question'], row['answer_choices'], row['gold_answer']], sort_keys=True)  # question-only input
want = {(q, c): key(r, c) for q, r in canon.items() for c in CONDS}
found = {}
for p in glob.glob(str(ROOT / 'docs/experiments/**/results_by_sample.jsonl'), recursive=True):
    for l in open(p, encoding='utf-8-sig'):
        if not l.strip(): continue
        r = json.loads(l); c = r.get('condition'); q = r.get('question_id')
        if c not in CONDS or (q, c) not in want or r.get('model') != 'gemini-3.7-flash' or r.get('system_prompt') != base.SYSTEM_PROMPT: continue
        if r.get('prediction') not in r.get('answer_choices', {}): continue
        if 'evidence_images' not in r: continue
        if key(r, c) == want[(q, c)] and (q, c) not in found:
            found[(q, c)] = dict(source=os.path.relpath(p, ROOT), prediction=r['prediction'], correct=r['prediction'] == canon[q]['gold_answer'])
todo = sorted(k for k in want if k not in found)
(HERE / 'reuse_plan.json').write_text(json.dumps(dict(reused={f'{q}|{c}': v for (q, c), v in found.items()}, todo=[f'{q}|{c}' for q, c in todo]), indent=1))
for c in CONDS: print(c, 'reusable', sum(1 for k in found if k[1] == c), 'todo', sum(1 for k in todo if k[1] == c))
