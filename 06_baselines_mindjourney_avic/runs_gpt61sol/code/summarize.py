"""Summary of the GPT-6.1-sol MJ/AVIC variant-A runs (accuracy % by question type).
Columns: GPT-6.1-sol observed (our standard prompt), each method's raw pipeline answer, its own no-imagination base,
and the FINAL config (MJ top3+label, AVIC label), plus imagination usage (questions whose final prompt has >= 1 imagined
view). Writes summary.json next to this script's parent and prints Markdown."""
import collections, glob, json, os
T = '06_baselines_mindjourney_avic/runs_gpt61sol'
OBS = {r['uid']: r['correct'] for r in json.load(open(
    '01_oracle_gpt61sol/observed_oracle_8conditions_922/summary.json'))['per_question'] if r['condition'] == 'observed'}
FINAL = {'mj': 'top3+label', 'avic': 'label'}
out = {}


def raw(method, s):
    pat = f'{T}/results/A/{method}_{s}*/*/*/gpt.json' if method == 'mj' else f'{T}/results/A/{method}_{s}*/*/*/step_0/gpt.json'
    res = {}
    for g in glob.glob(pat):
        x = json.load(open(g)); res[x['question']['database_idx']] = (x['question']['question_type'], x.get('result') == 'correct')
    return res


def qa(method, s, v):
    return {r['qid']: r for r in (json.load(open(f)) for f in glob.glob(f'{T}/qa/{s}/{method}/{v}/*.json'))}


pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None
for s in ('fo622', 'mvc300'):
    for m in ('mj', 'avic'):
        rw = raw(m, s); base = qa(m, s, 'base'); fin = qa(m, s, FINAL[m])
        types = sorted({t for t, _ in rw.values()})
        rows = {}
        for t in types + ['All']:
            us = [u for u, (tt, _) in rw.items() if t in ('All', tt)]
            rows[t] = dict(n=len(us), observed=pct([OBS[u] for u in us if u in OBS]), raw=pct([rw[u][1] for u in us]),
                           base=pct([base[u]['correct'] for u in us if u in base]), final=pct([fin[u]['correct'] for u in us if u in fin]),
                           imagination_used=sum(fin[u]['n_imagined'] > 0 for u in us if u in fin))
        out[f'{m}_{s}'] = rows
        print(f'\n### {m.upper()} {s} (final = {FINAL[m]}; completed {len(rw)}, base {len(base)}, final {len(fin)})\n')
        print('| type | n | GPT observed | raw pipeline | own base | FINAL | imagination used |\n|---|---|---|---|---|---|---|')
        for t, r in rows.items():
            print(f"| {t} | {r['n']} | {r['observed']} | {r['raw']} | {r['base']} | {r['final']} | {r['imagination_used']} |")
usd = collections.defaultdict(float)
for f in glob.glob(f'{T}/usage/*.jsonl'):
    for l in open(f):
        usd[os.path.basename(f).split('_c')[0]] += json.loads(l).get('usd', 0)
print('\nGPT cost (USD):', {k: round(v, 2) for k, v in sorted(usd.items())}, 'total', round(sum(usd.values()), 2))
json.dump(dict(tables=out, usd=usd), open(f'{T}/summary.json', 'w'), indent=1)
