"""Gating ablation for GPT-6.1-sol with estimated evidence (2026-10-11; 1/3 random subsets, then all questions).

Settings (final answers are looked up; only No-gate VI answers and the neutral gate-2 choices are new model calls):
  No gate : every question gets ALL estimated evidence at once.
            VI = observed + est RGB + top-down + text + video + depth + normals + panorama (new run `start_final_allreps`);
            CV = observed + est RGB + est depth (existing `est_rgb_depth`; these are all CV estimated representations).
  +Gate1  : rule A. Observed answer == est-RGB answer -> keep it; otherwise -> the No-gate answer.
  +Gate2  : every question goes to gate 2, which picks one option; answer looked up. The gate-2 prompt is the frozen
            S2 (VI) / S2MV (CV) prompt minus the sentence saying the two answers disagreed (versions S2N / S2MVN).
  Gate1+2 : the reported pipeline. Rule A; otherwise S2 / S2MV choice (existing results).
Subsets: stratified by question type, 1/3 of each type (rounded), seed 20261011 (make_subsets()).
Usage: python ablation_eval.py subsets | python ablation_eval.py eval [--full]
"""
import collections, json, random, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; EXP = HERE.parent
G = EXP / 'gating_gpt61sol_est_20261009'; M = EXP / 'gpt61sol_mvc_estimated_300_20261010'
VP = json.load(open(G / 'inputs/gpt_est_predictions.json'))          # uid -> condition -> {pred, gold}
VCAT = json.load(open(G / 'inputs/categories.json'))
CP = json.load(open(M / 'mvc_preds.json'))                           # uid -> {category, gold, est_rgb, est_rgb_depth, observed}
VTYPES = ['open_space_direction', 'nearest_obstacle_direction', 'visibility', 'nearest_object', 'farthest_object',
          'object_relative_direction']
CTYPES = ['cross_view_closest', 'cross_view_farthest', 'ordering']
NAMES = {'open_space_direction': 'Open-space Direction', 'nearest_obstacle_direction': 'Nearest-obstacle Direction',
         'visibility': 'Visibility', 'nearest_object': 'Nearest Object', 'farthest_object': 'Farthest Object',
         'object_relative_direction': 'Object-relative Direction', 'cross_view_closest': 'Cross-view Nearest Object',
         'cross_view_farthest': 'Cross-view Farthest Object', 'ordering': 'Spatial Ordering'}


def load(d):
    return {f.stem: json.loads(f.read_text(encoding='utf-8')) for f in Path(d).glob('*.json')}


def make_subsets(seed=20261011):
    rng = random.Random(seed); out = {}
    for name, cat, types in (('vi', VCAT, VTYPES), ('cv', {u: v['category'] for u, v in CP.items()}, CTYPES)):
        us = []
        for t in types:
            cu = sorted(u for u in cat if cat[u] == t); us += sorted(rng.sample(cu, round(len(cu) / 3)))
        out[name] = us; json.dump(us, open(HERE / f'{name}_third_uids.json', 'w'), indent=0)
        print(name, len(us), dict(collections.Counter(cat[u] for u in us)))
    return out


def vi_answers(allreps, s2, s2n):
    ok = lambda u, c: VP[u][c]['pred'] == VP[u][c]['gold']
    obs_ok = lambda u: VP[u]['observed']['pred'] == VP[u]['observed']['gold']
    agree = lambda u: VP[u]['observed']['pred'] == VP[u]['start_final']['pred']
    look = lambda u, ch: obs_ok(u) if ch == 'start_only' else ok(u, 'start_final' if ch == 'est_rgb' else 'start_final_' + ch)
    nog = lambda u: allreps[u]['correct'] if u in allreps else None
    return {
        'Observed view': obs_ok, '+ Est. RGB': lambda u: ok(u, 'start_final'),
        'No gate': nog,
        '+Gate1': lambda u: ok(u, 'start_final') if agree(u) else nog(u),
        '+Gate2': lambda u: look(u, s2n[u]['choice']) if u in s2n else None,
        'Gate1+Gate2': lambda u: ok(u, 'start_final') if agree(u) else (look(u, s2[u]['choice']) if u in s2 else None),
    }, agree


def cv_answers(s2mv, s2mvn):
    ok = lambda u, c: CP[u][c] == CP[u]['gold']
    agree = lambda u: CP[u]['observed'] == CP[u]['est_rgb']
    look = lambda u, ch: ok(u, {'start_only': 'observed', 'est_rgb': 'est_rgb', 'depth': 'est_rgb_depth'}[ch])
    return {
        'Observed view': lambda u: ok(u, 'observed'), '+ Est. RGB': lambda u: ok(u, 'est_rgb'),
        'No gate': lambda u: ok(u, 'est_rgb_depth'),
        '+Gate1': lambda u: ok(u, 'est_rgb') if agree(u) else ok(u, 'est_rgb_depth'),
        '+Gate2': lambda u: look(u, s2mvn[u]['choice']) if u in s2mvn else None,
        'Gate1+Gate2': lambda u: ok(u, 'est_rgb') if agree(u) else (look(u, s2mv[u]['choice']) if u in s2mv else None),
    }, agree


def boot(a, b, us, n=2000, seed=0):
    rng = random.Random(seed); d = [int(a(u)) - int(b(u)) for u in us]; xs = []
    for _ in range(n):
        s = [d[rng.randrange(len(d))] for _ in d]; xs.append(100 * sum(s) / len(s))
    xs.sort(); return 100 * sum(d) / len(d), xs[int(0.025 * n)], xs[int(0.975 * n) - 1]


def table(title, F, us, types, cat):
    cols = list(F)
    print(f'\n### {title} (n={len(us)})\n')
    print('| Question type | ' + ' | '.join(cols) + ' |'); print('|---' * (len(cols) + 1) + '|')
    for t in types + ['All']:
        cu = us if t == 'All' else [u for u in us if cat[u] == t]
        cells = []
        for c in cols:
            v = [F[c](u) for u in cu]
            cells.append('n/a' if any(x is None for x in v) else f'{100 * sum(v) / len(v):.1f}')
        print(f'| {"**All**" if t == "All" else NAMES[t]} (n={len(cu)}) | ' + ' | '.join(cells) + ' |')
    full = F['Gate1+Gate2']
    if all(full(u) is not None and all(F[c](u) is not None for c in cols) for u in us):
        print('\nGate1+Gate2 minus each setting, paired bootstrap 95% CI:')
        for c in cols[:-1]:
            m, lo, hi = boot(full, F[c], us); print(f'  vs {c:14s} {m:+.1f} [{lo:+.1f}, {hi:+.1f}]')


if __name__ == '__main__':
    if sys.argv[1] == 'subsets':
        make_subsets(); sys.exit()
    R = HERE / 'results'; ALL = '--all' in sys.argv   # --all: full sets, 8-option VI gate 2 (S2NE / S2E)
    allreps = load(R / 'start_final_allreps'); s2mvn = load(R / 'gate2sel_S2MVN')
    s2n = load(R / ('gate2sel_S2NE' if ALL else 'gate2sel_S2N'))
    s2 = load(G / ('results/gate2sel_S2E' if ALL else 'results/gate2sel_S2')); s2mv = load(M / 'results_gate2/S2MV')
    FV, agv = vi_answers(allreps, s2, s2n); FC, agc = cv_answers(s2mv, s2mvn)
    if '--full' in sys.argv:   # reproduce the reported full-benchmark numbers (Gate1+Gate2 only)
        for nm, F, us in (('VI', FV, sorted(VCAT)), ('CV', FC, sorted(CP))):
            print(nm, len(us), 'Gate1+Gate2 %.1f' % (100 * sum(F['Gate1+Gate2'](u) for u in us) / len(us)))
        sys.exit()
    if ALL:
        vi, cv = sorted(VCAT), sorted(CP)
    else:
        vi = json.load(open(HERE / 'vi_third_uids.json')); cv = json.load(open(HERE / 'cv_third_uids.json'))
    print('VI gate-1 disagreement on subset:', sum(not agv(u) for u in vi), '/', len(vi),
          '| CV:', sum(not agc(u) for u in cv), '/', len(cv))
    for nm, rs in (('VI S2N', s2n), ('CV S2MVN', s2mvn)):
        print(nm, 'choices', dict(collections.Counter(r['choice'] for r in rs.values())))
    table('Viewpoint Imagination, ' + ('all' if ALL else '1/3 subset'), FV, vi, VTYPES, VCAT)
    table('Cross-view Integration, ' + ('all' if ALL else '1/3 subset'), FC, cv, CTYPES, {u: v['category'] for u, v in CP.items()})
