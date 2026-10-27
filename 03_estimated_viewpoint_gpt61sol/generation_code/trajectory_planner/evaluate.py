"""LOCAL-ONLY offline scoring against Oracle reference geometry. Never imported by navigator/env."""
import json, math, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
REF = json.loads((REPO / 'outputs/hy_autonomy_200_20261002/evaluation_only/reference_geometry.json').read_text())
def wrap(a): return (a + 180) % 360 - 180
def length(pts): return sum(math.dist(a[:2], b[:2]) for a, b in zip(pts, pts[1:]))
def score(plan):
    r = REF[plan['question_id']]; o = r['points_m']
    if plan.get('status') != 'complete': return {'qid': plan['question_id'], 'status': plan.get('status'), 'error': plan.get('error')}
    f = plan['final']; route = plan['route_m']
    return {'qid': plan['question_id'], 'split': r['split'], 'endpoint_err_m': round(math.dist([f['x'], f['y']], o[-1][:2]), 2),
            'yaw_err_deg': round(abs(wrap(f['yaw_deg'] - math.degrees(r['final_yaw']))), 1),
            'pitch_err_deg': round(abs(f['pitch_deg'] - math.degrees(r['final_pitch'])), 1),
            'len_m': round(length(route), 2), 'oracle_len_m': round(length(o), 2), 'len_ratio': round(length(route) / max(length(o), 1e-6), 2),
            'oracle_end': [round(v, 2) for v in o[-1][:2]], 'our_end': [round(f['x'], 2), round(f['y'], 2)]}
if __name__ == '__main__':
    rows = [score(json.loads(Path(p).read_text())) for p in sys.argv[1:]]
    for r in rows: print(json.dumps(r))
    ok = [r for r in rows if 'endpoint_err_m' in r]
    if ok:
        import statistics as S
        print('N', len(ok), 'median endpoint', S.median(r['endpoint_err_m'] for r in ok), 'median yaw', S.median(r['yaw_err_deg'] for r in ok), 'median len ratio', S.median(r['len_ratio'] for r in ok))
