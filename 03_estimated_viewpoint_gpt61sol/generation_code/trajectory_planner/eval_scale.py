"""LOCAL-ONLY: compare per-scene scale methods by rescaling existing plans (positions scale linearly; yaw unchanged).
Global factors are fitted leave-one-scene-out so no scene's own Oracle endpoint influences its scale."""
import glob, json, math, statistics as S, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import REF  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
tag = sys.argv[1] if len(sys.argv) > 1 else 'r9'
scale = {p.stem: json.loads(p.read_text()) for p in (REPO / 'outputs/hy_nav_pilot_20261002/scale').glob('*.json')}
rows = []
for p in glob.glob(str(REPO / f'outputs/hy_nav_pilot_20261002/{tag}/*/plan.json')):
    d = json.loads(Path(p).read_text())
    if d.get('status') != 'complete':
        continue
    sid = d['question_id'].rsplit('-', 1)[0]
    if sid not in scale:
        continue
    o = REF[d['question_id']]['points_m'][-1][:2]
    e = [d['final']['x'], d['final']['y']]
    rows.append({'qid': d['question_id'], 'sid': sid, 'o': o, 'e175': e, 'mpu175': d['m_per_unit'], 'moge': scale[sid]['m_per_unit_moge']})


def fit_k(sub, key):
    num = den = 0.0
    for r in sub:
        e = [v * r[key] for v in r['unit']]
        num += e[0] * r['o'][0] + e[1] * r['o'][1]; den += e[0] ** 2 + e[1] ** 2
    return num / den if den else 1.0


for r in rows:
    r['unit'] = [v / r['mpu175'] for v in r['e175']]          # positions in HY units
methods = {}
methods['eye 1.75 (current)'] = [math.dist([v * r['mpu175'] for v in r['unit']], r['o']) for r in rows]
methods['eye 1.60 (prior)'] = [math.dist([v * r['mpu175'] * 1.6 / 1.75 for v in r['unit']], r['o']) for r in rows]
methods['MoGe raw'] = [math.dist([v * r['moge'] for v in r['unit']], r['o']) for r in rows]
loso = []
for r in rows:
    k = fit_k([x for x in rows if x['sid'] != r['sid']], 'moge')
    loso.append(math.dist([v * r['moge'] * k for v in r['unit']], r['o']))
methods['MoGe x global k (leave-one-scene-out)'] = loso
print('questions', len(rows), 'scenes', len({r['sid'] for r in rows}))
print('global k on MoGe (all scenes, for reference):', round(fit_k(rows, 'moge'), 3))
for name, errs in methods.items():
    print(f'{name:40s} median {S.median(errs):.2f}  mean {S.mean(errs):.2f}  <1m {sum(e < 1 for e in errs)/len(errs):.0%}  <1.5m {sum(e < 1.5 for e in errs)/len(errs):.0%}  <2m {sum(e < 2 for e in errs)/len(errs):.0%}')
implied = [scale[s]['implied_eye_height_m'] for s in {r['sid'] for r in rows}]
print('MoGe implied eye height: median', round(S.median(implied), 2), 'range', round(min(implied), 2), '-', round(max(implied), 2))
