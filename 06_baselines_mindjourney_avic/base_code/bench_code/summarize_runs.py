"""Summarize pipeline runs and QA-replay variants per question type.
  python summarize_runs.py <ids.json> <label>=<results-dir-glob>[:mj|avic] ...   (stored pipeline results)
                                     <label>=qa:<qa_variants dir>                  (replayed variants)
Writes per-question correctness to stdout-adjacent JSON (--dump path) for joining with other baselines.
"""
import glob, json, os, sys, collections

R = '<external>/imagination_baselines_mj_avic_20261006'
ids = json.load(open(sys.argv[1]))
ids = [x['database_idx'] if isinstance(x, dict) else x for x in ids]
dump = sys.argv[sys.argv.index('--dump') + 1] if '--dump' in sys.argv else None
specs = [a for a in sys.argv[2:] if '=' in a]
out = {}
for spec in specs:
    label, src = spec.split('=', 1)
    rows = {}
    if src.startswith('qa:'):
        for f in glob.glob(f'{R}/results/{src[3:]}/*.json'):
            x = json.load(open(f)); rows[x['qid']] = (bool(x['correct']), x['n_imagined'], x.get('category'))
    else:
        d, m = src.rsplit(':', 1)
        pat = f'{R}/results/{d}/*/*/gpt.json' if m == 'mj' else f'{R}/results/{d}/*/*/step_0/gpt.json'
        for g in glob.glob(pat):
            x = json.load(open(g)); q = x['question']
            n = sum(1 for it in x['prompt']['content'] if len(it) == 2) - len(q['img_paths'])
            rows[q['database_idx']] = (x.get('result') == 'correct', n, q['question_type'])
    sel = [q for q in ids if q in rows]
    by = collections.defaultdict(list)
    for q in sel:
        by[rows[q][2]].append(rows[q][0])
    acc = 100 * sum(rows[q][0] for q in sel) / max(1, len(sel))
    used = sum(rows[q][1] > 0 for q in sel)
    print(f'{label:28s} n={len(sel):3d} acc={acc:5.1f}  imagined/q={sum(rows[q][1] for q in sel)/max(1,len(sel)):.2f}  used={used}/{len(sel)}  ' +
          ' '.join(f'{k[:12]}={100*sum(v)/len(v):.1f}' for k, v in sorted(by.items())))
    out[label] = {q: rows[q][0] for q in sel}
if dump:
    json.dump(out, open(dump, 'w'))
