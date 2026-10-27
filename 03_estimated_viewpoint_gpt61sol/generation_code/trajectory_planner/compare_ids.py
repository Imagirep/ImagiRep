"""LOCAL-ONLY: compare plan sets on an explicit question list. Args: ids_file label=globpattern ..."""
import glob, json, statistics as S, sys, re, os
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import score, REF
ids = open(sys.argv[1]).read().split(',')
def stats(v):
    return (f"n={len(v):2d} end<1m {sum(r['endpoint_err_m']<1 for r in v)/len(v):.0%} end<1.5m {sum(r['endpoint_err_m']<1.5 for r in v)/len(v):.0%} "
            f"end<2m {sum(r['endpoint_err_m']<2 for r in v)/len(v):.0%} yaw<20 {sum(r['yaw_err_deg']<20 for r in v)/len(v):.0%} "
            f"mean {S.mean(r['endpoint_err_m'] for r in v):.2f} med {S.median(r['endpoint_err_m'] for r in v):.2f} med_yaw {S.median(r['yaw_err_deg'] for r in v):.1f}")
sets = {}
for arg in sys.argv[2:]:
    label, pat = arg.split('=', 1)
    plans = {}
    for p in glob.glob(pat, recursive=True):
        d = json.load(open(p))
        if d.get('status') == 'complete': plans[d['question_id']] = d
    sets[label] = plans
common = [q for q in ids if all(q in s for s in sets.values())]
print('questions in all sets:', len(common), 'of', len(ids))
for label, plans in sets.items():
    rows = [score(plans[q]) for q in common]
    print(f'{label:14s} ALL', stats(rows))
    for name, pat in [('to/near', r'\b(walk|go|move|stop) (to|near|next to|beside)\b|\bnear the\b'), ('through/past', r'\b(through|into|past)\b')]:
        v = [score(plans[q]) for q in common if re.search(pat, plans[q]['motion_text'].lower())]
        if v: print(f'{label:14s} {name:12s}', stats(v))
    h = [score(plans[q]) for q in common if REF[q]['split'] == 'holdout']
    if h: print(f'{label:14s} holdout     ', stats(h))
