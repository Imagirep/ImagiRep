import json,glob,sys,re,statistics as S
sys.path.insert(0,'scripts/hy_nav_pilot')
from evaluate import score
def stats(rows):
    v=[r for r in rows if 'endpoint_err_m' in r]
    return f"n={len(v):2d} end<1m {sum(r['endpoint_err_m']<1 for r in v)/len(v):.0%} end<1.5m {sum(r['endpoint_err_m']<1.5 for r in v)/len(v):.0%} end<2m {sum(r['endpoint_err_m']<2 for r in v)/len(v):.0%} yaw<20 {sum(r['yaw_err_deg']<20 for r in v)/len(v):.0%} mean_end {S.mean(r['endpoint_err_m'] for r in v):.2f} med_end {S.median(r['endpoint_err_m'] for r in v):.2f} med_yaw {S.median(r['yaw_err_deg'] for r in v):.1f}"
for t in sys.argv[1:]:
    rows=[(score(d),d) for p in glob.glob(f'outputs/hy_nav_pilot_20261002/{t}/*/plan.json') for d in [json.load(open(p))]]
    print(t,'ALL',stats([r for r,_ in rows]))
    print(t,'through/past',stats([r for r,d in rows if re.search(r'\b(through|into|past)\b',d['motion_text'].lower())]))
