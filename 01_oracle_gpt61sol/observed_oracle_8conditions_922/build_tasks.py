"""Build GPT-6.1-sol request tasks for the 922 benchmark questions x {observed, +oracle} (2026-10-07).

Prompts/images are IDENTICAL to the standard Gemini evaluation:
  FO 622 (canonical_622_rows): system = runner_snapshot.SYSTEM_PROMPT
    observed = ["Start:" + start image] ; oracle = + [FINAL_VIEW_PROMPT + final image (lookback if show_lookback else endpoint)]
    then runner_snapshot.question_text(row)
  MVC 300 (fixed set, summary_mvc300.json uids): system = cohort prompts/system.txt
    observed = settings.original_only ; oracle = settings.oracle_rgb ; label + image each,
    then "<original_only.txt | additional_rgb.txt>\n\nQuestion: ...\n<options>\nAnswer:"
Output: tasks.jsonl (one line per request; images referenced by name in images/), images/ (copied once).
"""
import hashlib, importlib.util, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'payload'
(OUT / 'images').mkdir(parents=True, exist_ok=True)


def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


FO = load('fo_runner', ROOT / 'docs/experiments/fo_622_supplement_eight_conditions_20261005/runner_snapshot.py')
MV = load('mvc_run', ROOT / 'docs/experiments/mvc_oracle_modalities_325_20261006/run_mvc_modalities.py')
base = MV.base
names = {}


def img(path):
    p = Path(path); h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    n = names.get(h)
    if n is None:
        n = names[h] = f'{h}{p.suffix}'; shutil.copy(p, OUT / 'images' / n)
    return n


tasks = []
for l in open(ROOT / 'docs/experiments/fo_estimated_seven_conditions_622_v5_20261005/canonical_622_rows.jsonl', encoding='utf-8'):
    r = json.loads(l)
    start = [a for a in r['evidence_images'] if a['kind'] == 'start']
    final = [a for a in r['evidence_images'] if a['kind'] == ('lookback' if r['show_lookback'] else 'endpoint')]
    assert len(start) == 1 and len(final) == 1, r['question_id']
    for cond, ev in [('observed', [('Start:', start[0])]), ('oracle', [('Start:', start[0]), (FO.FINAL_VIEW_PROMPT, final[0])])]:
        parts = []
        for cap, a in ev:
            parts += [{'text': cap + '\n'}, {'image': img(a['path'])}]
        parts.append({'text': FO.question_text(r)})
        tasks.append(dict(uid=r['question_id'], benchmark='FO622', category=r['category'], condition=cond, system=FO.SYSTEM_PROMPT,
                          parts=parts, valid=sorted(r['answer_choices']), gold=r['gold_answer']))

uids = json.load(open(ROOT / 'docs/experiments/mvc_oracle_modalities_325_20261006/results/summary_mvc300.json'))['uids']
R = {r['uid']: r for r in MV.rows()}
SUP = ROOT / 'outputs/mvc_mvc131_supplement_20261006'
for l in open(SUP / 'model_inputs.jsonl', encoding='utf-8'):
    r = json.loads(l); r['cohort'] = 'mvc131'; r['uid'] = 'mvc131__' + r['question_id']; R[r['uid']] = r
for u in uids:
    r = R[u]
    pdir = (SUP if r['cohort'] == 'mvc131' else MV.COHORTS[r['cohort']]) / 'prompts'
    system = (pdir / 'system.txt').read_text(encoding='utf-8').strip()
    for cond, setting, pf in [('observed', 'original_only', 'original_only.txt'), ('oracle', 'oracle_rgb', 'additional_rgb.txt')]:
        parts = []
        for a in r['settings'][setting]:
            parts += [{'text': a['label'] + ':\n'}, {'image': img(ROOT / a['path'])}]
        prompt = (pdir / pf).read_text(encoding='utf-8').strip()
        parts.append({'text': f"{prompt}\n\nQuestion: {r['question']}\n{base.option_text(r['answer_choices'])}\nAnswer:"})
        tasks.append(dict(uid=u, benchmark='MVC300', category=r['question_type'], condition=cond, system=system,
                          parts=parts, valid=sorted(r['answer_choices']), gold=r['gold_answer']))

with open(OUT / 'tasks.jsonl', 'w', encoding='utf-8') as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + '\n')
print(len(tasks), 'tasks;', len(names), 'unique images')
