#!/usr/bin/env python3
"""Recompute every paper number from raw per-question files (no model calls).

Run:  cd 07_verification/scripts && python3 verify_from_raw.py > ../verification_output.txt
Writes verification_values.json next to this script.
Sources: server experiment folders (read-only) + local_only_copies/ (Gemini 3.7 runs, nav plans, local summaries).
Accuracy = 100 * correct / n, rounded to 1 decimal.
Recovery score (paper Tables 1, 4, 7, 8): correct with endpoint RGB OR correct with endpoint RGB + that representation.
Oracle selection: correct with endpoint RGB OR with ANY of the six representations (observed-only NOT included).
"""
import collections, glob, importlib.util, json, math, os, statistics, sys

import os as _os
E = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..')) + '/'  # bundle root
REG = E + '07_verification'  # local-only copies live here
LOC = E  # public layout: former local-only copies are in the bundle
VALS = {}
CHECKS = []  # (item, paper, verified, status)
pct = lambda k, n: round(100.0 * k / n, 1) if n else None
VI_T = ['open_space_direction', 'nearest_obstacle_direction', 'visibility', 'nearest_object', 'farthest_object', 'object_relative_direction']
CV_T = ['cross_view_closest', 'cross_view_farthest', 'ordering']
REPS = ['topdown', 'text', 'video', 'depth', 'surface_normals', 'panorama']


def truthy(v):
    return v is True or (isinstance(v, str) and v.strip().lower() == 'true')


def check(item, paper, got):
    st = 'OK' if paper is None or got is None or abs(float(paper) - float(got)) < 0.051 else 'MISMATCH'
    if paper is None:
        st = 'NO PAPER VALUE'
    CHECKS.append((item, paper, got, st))
    VALS[item] = got


def load_dir(d):
    out = {}
    for p in glob.glob(d + '/*.json'):
        r = json.load(open(p)); out[r.get('uid') or os.path.basename(p)[:-5]] = r
    return out


def mtimes(d):
    ts = [os.path.getmtime(p) for p in glob.glob(d + '/*.json')]
    import datetime
    f = lambda t: datetime.datetime.fromtimestamp(t).isoformat(timespec='minutes')
    return (f(min(ts)), f(max(ts))) if ts else (None, None)


def rows_by_type(uids, cat, fn, types):
    res = {}
    for t in types + ['ALL']:
        us = [u for u in uids if t == 'ALL' or cat[u] == t]
        res[t] = (pct(sum(bool(fn(u)) for u in us), len(us)), sum(bool(fn(u)) for u in us), len(us))
    return res


def emit_table(name, cols, types, paper):
    """cols: list of (colname, {type: (acc,k,n)}); paper: {colname: [values in types+ALL order]}"""
    print(f'\n### {name}')
    print('| type | ' + ' | '.join(c for c, _ in cols) + ' |')
    print('|---' * (len(cols) + 1) + '|')
    for i, t in enumerate(types + ['ALL']):
        cells = []
        for c, r in cols:
            acc, k, n = r[t]
            pv = paper.get(c, [None] * (len(types) + 1))[i]
            mark = '' if pv is None or abs(pv - acc) < 0.051 else f' (paper {pv}) **MISMATCH**'
            cells.append(f'{acc} [{k}/{n}]{mark}')
            check(f'{name} | {c} | {t}', pv, acc)
        print(f'| {t} | ' + ' | '.join(cells) + ' |')


# ---------------------------------------------------------------- uid sets
fo_uids = sorted(json.loads(l)['uid'] for l in open(E + 'benchmark/benchmark922/questions.jsonl') if json.loads(l)['benchmark'] == 'FO622')
mvc_final = json.load(open(E + '02_oracle_gemini37/crossview_integration/mvc_oracle_modalities_325_20261006/results/summary_mvc300.json'))['uids']
v2_mvc = sorted(json.loads(l)['uid'] for l in open(E + 'benchmark/benchmark922/questions.jsonl') if json.loads(l)['benchmark'] == 'MVC300')
print('FO622 uids:', len(fo_uids), '| MVC300 uids:', len(mvc_final), '| benchmark MVC == final set:', sorted(mvc_final) == v2_mvc)

# ---------------------------------------------------------------- GPT-6.1-sol oracle (Table 1, Table 2)
G = E + '01_oracle_gpt61sol/observed_oracle_8conditions_922/results/'
gconds = ['observed', 'oracle'] + ['oracle_' + m for m in REPS]
GO = {c: load_dir(G + c) for c in gconds}
print('\nGPT oracle run models:', collections.Counter(r['model'] for c in gconds for r in GO[c].values()),
      'file mtimes', {c: mtimes(G + c) for c in ['observed', 'oracle_video']})
cat = {u: r['category'] for u, r in GO['observed'].items()}
assert set(u for u, r in GO['observed'].items() if r['benchmark'] == 'FO622') == set(fo_uids)
assert set(u for u, r in GO['observed'].items() if r['benchmark'] == 'MVC300') == set(mvc_final)
C = lambda c, u: truthy(GO[c][u]['correct'])
P_T1_VI = {'Observed view': [60.4, 38.2, 39.4, 51.0, 68.4, 70.0, 57.2], '+ Endpoint RGB': [57.1, 49.1, 64.4, 64.4, 58.2, 82.9, 66.1],
           'Top-down': [74.7, 61.8, 73.1, 76.9, 75.5, 88.2, 77.5], 'Text': [69.2, 58.2, 71.2, 73.1, 67.3, 87.6, 74.0],
           'Video': [65.9, 60.0, 76.0, 76.0, 72.4, 89.4, 76.2], 'Depth': [64.8, 63.6, 73.1, 76.9, 70.4, 86.5, 74.9],
           'Normals': [61.5, 58.2, 75.0, 71.2, 67.3, 85.9, 72.7], 'Panorama': [73.6, 65.5, 76.9, 80.8, 71.4, 91.8, 79.3],
           'Oracle selection': [89.0, 81.8, 80.8, 87.5, 87.8, 92.4, 87.5]}
P_T1_CV = {'Observed view': [56.8, 59.8, 79.6, 68.0], '+ Endpoint RGB': [74.1, 54.9, 89.1, 75.7], 'Top-down': [79.0, 70.7, 92.0, 82.7],
           'Text': [79.0, 67.1, 95.6, 83.3], 'Video': [77.8, 69.5, 94.2, 83.0], 'Depth': [84.0, 69.5, 94.2, 84.7],
           'Normals': [80.2, 59.8, 92.7, 80.3], 'Panorama': [76.5, 64.6, 95.6, 82.0], 'Oracle selection': [87.7, 85.4, 97.8, 91.7]}
NM = dict(zip(REPS, ['Top-down', 'Text', 'Video', 'Depth', 'Normals', 'Panorama']))
for name, uids, types, paper in [('T1 GPT oracle VI', fo_uids, VI_T, P_T1_VI), ('T1 GPT oracle CV', mvc_final, CV_T, P_T1_CV)]:
    cols = [('Observed view', rows_by_type(uids, cat, lambda u: C('observed', u), types)),
            ('+ Endpoint RGB', rows_by_type(uids, cat, lambda u: C('oracle', u), types))]
    for m in REPS:
        cols.append((NM[m], rows_by_type(uids, cat, lambda u, m=m: C('oracle', u) or C('oracle_' + m, u), types)))
    cols.append(('Oracle selection', rows_by_type(uids, cat, lambda u: C('oracle', u) or any(C('oracle_' + m, u) for m in REPS), types)))
    emit_table(name, cols, types, paper)

# GPT question only
Q = load_dir(E + '01_oracle_gpt61sol/question_only_922/results/question_only')
print('\nGPT question-only models', collections.Counter(r['model'] for r in Q.values()), mtimes(E + '01_oracle_gpt61sol/question_only_922/results/question_only'))
check('T2 GPT question-only VI', 31.0, pct(sum(truthy(Q[u]['correct']) for u in fo_uids), 622))
check('T2 GPT question-only CV', 31.0, pct(sum(truthy(Q[u]['correct']) for u in mvc_final), 300))
check('T2 GPT observed VI', 57.2, VALS['T1 GPT oracle VI | Observed view | ALL'])
check('T2 GPT observed CV', 68.0, VALS['T1 GPT oracle CV | Observed view | ALL'])
check('T2 GPT +oracle RGB VI', 66.1, VALS['T1 GPT oracle VI | + Endpoint RGB | ALL'])
check('T2 GPT +oracle RGB CV', 75.7, VALS['T1 GPT oracle CV | + Endpoint RGB | ALL'])

# ---------------------------------------------------------------- Gemini 3.7 Flash VI oracle (Table 7, 8, 9; Table 2)
spec = importlib.util.spec_from_file_location('sv', E + '02_oracle_gemini37/viewpoint_imagination/single_view_modality_tables_b1_4_20261005/sv622.py')
sv = importlib.util.module_from_spec(spec); cwd = os.getcwd(); spec.loader.exec_module(sv)
qs, look, gcat, batch = sv.table622(); os.chdir(cwd)
assert sorted(qs) == fo_uids, 'Gemini 622 != benchmark FO622'
print('\nGemini VI batches (question counts by source batch):', dict(collections.Counter(batch.values())))
L = lambda q, c: truthy(look[(q, c)]) or look[(q, c)] is True
GM = {'start_view': 'Observed view', 'start_final': '+ Endpoint RGB'}
P_T7_VI = {'Observed view': [52.7, 36.4, 32.7, 51.9, 66.3, 66.5, 53.7], '+ Endpoint RGB': [58.2, 45.5, 49.0, 65.4, 65.3, 79.4, 63.7],
           'Top-down': [67.0, 61.8, 57.7, 75.0, 75.5, 86.5, 73.0], 'Text': [63.7, 54.5, 57.7, 70.2, 68.4, 84.1, 69.3],
           'Video': [63.7, 56.4, 55.8, 68.3, 69.4, 87.6, 69.9], 'Depth': [69.2, 52.7, 56.7, 71.2, 73.5, 84.1, 70.7],
           'Normals': [63.7, 52.7, 51.9, 69.2, 69.4, 82.9, 67.8], 'Panorama': [74.7, 56.4, 55.8, 70.2, 71.4, 87.1, 72.0],
           'Oracle selection': [85.7, 74.5, 69.2, 80.8, 80.6, 93.5, 82.5]}
cols = [('Observed view', rows_by_type(qs, gcat, lambda u: L(u, 'start_view'), VI_T)),
        ('+ Endpoint RGB', rows_by_type(qs, gcat, lambda u: L(u, 'start_final'), VI_T))]
for m in REPS:
    cols.append((NM[m], rows_by_type(qs, gcat, lambda u, m=m: L(u, 'start_final') or L(u, 'start_final_' + m), VI_T)))
cols.append(('Oracle selection', rows_by_type(qs, gcat, lambda u: L(u, 'start_final') or any(L(u, 'start_final_' + m) for m in REPS), VI_T)))
emit_table('T7 Gemini oracle VI', cols, VI_T, P_T7_VI)
# Table 9 per-type selection (Gemini VI)
print('\n### T9 per-type best representation (Gemini VI): endpoint correct / rescued per representation')
P_T9 = {'open_space_direction': ('panorama', 53, 15, 68, 16.5), 'nearest_obstacle_direction': ('topdown', 25, 9, 34, 16.4),
        'visibility': ('text', 51, 9, 60, 8.7), 'nearest_object': ('topdown', 68, 10, 78, 9.6), 'farthest_object': ('topdown', 64, 10, 74, 10.2),
        'object_relative_direction': ('video', 135, 14, 149, 8.2)}
tot_e = tot_r = 0
for t in VI_T:
    us = [q for q in qs if gcat[q] == t]; e = sum(L(q, 'start_final') for q in us)
    resc = {m: sum((not L(q, 'start_final')) and L(q, 'start_final_' + m) for q in us) for m in REPS}
    best = max(resc.values()); bests = [m for m in REPS if resc[m] == best]
    rep, pe, pr, ptot, pg = P_T9[t]
    print(f'{t}: n={len(us)} endpoint={e} rescued={resc} best={bests} | paper {rep} {pe}+{pr}={ptot} gain {pg}')
    check(f'T9 {t} endpoint correct', pe, e); check(f'T9 {t} rescued ({rep})', pr, resc[rep]); check(f'T9 {t} gain pp', pg, pct(resc[rep], len(us)))
    if rep not in bests:
        CHECKS.append((f'T9 {t} chosen representation', rep, bests, 'MISMATCH'))
    tot_e += e; tot_r += resc[rep]
check('T9 All endpoint correct', 396, tot_e); check('T9 All rescued', 67, tot_r); check('T9 All gain pp', 10.8, pct(tot_r, 622))
# Table 8 deltas (count-based)
e = sum(L(q, 'start_final') for q in qs)
for m, pv in zip(REPS, [9.3, 5.6, 6.3, 7.1, 4.2, 8.4]):
    check(f'T8 Gemini VI delta {m} (count-based)', pv, pct(sum(L(q, 'start_final') or L(q, 'start_final_' + m) for q in qs) - e, 622))
check('T8 Gemini VI delta oracle selection', 18.8, pct(sum(L(q, 'start_final') or any(L(q, 'start_final_' + m) for m in REPS) for q in qs) - e, 622))
# Gemini VI question only
qo = json.load(open(E + '02_oracle_gemini37/viewpoint_imagination/fo_622_question_only_trajectory_20261006/results/text_only_by_question.json'))
assert sorted(qo) == fo_uids
check('T2 Gemini question-only VI', 31.7, pct(sum(truthy(v['correct']) for v in qo.values()), 622))
check('T2 Gemini observed VI', 53.7, VALS['T7 Gemini oracle VI | Observed view | ALL'])
check('T2 Gemini +oracle RGB VI', 63.7, VALS['T7 Gemini oracle VI | + Endpoint RGB | ALL'])

# ---------------------------------------------------------------- Gemini 3.7 Flash MVC oracle
COH = {'ready100': '02_oracle_gemini37/crossview_integration/mvc_ready100_three_settings_20260930', 'last50': '02_oracle_gemini37/crossview_integration/mvc_batch1_last50_scenes_three_settings_20261001',
       'mvc131': '02_oracle_gemini37/crossview_integration/mvc_mvc131_supplement_20261006'}
MODS = ['oracle_rgb_' + m for m in REPS]
GC, GT, models, times = {}, {}, collections.Counter(), []


def gload(u, cond):
    c, q = u.split('__')
    if cond in ('text_only', 'original_only', 'oracle_rgb'):
        p = E + COH[c] + f'/results/per_sample/{cond}/{q}.json'
    elif c == 'mvc131':
        p = E + f'02_oracle_gemini37/crossview_integration/mvc_mvc131_supplement_20261006/modalities/results/per_sample/{cond}/{u}.json'
    else:
        p = E + f'02_oracle_gemini37/crossview_integration/mvc_oracle_modalities_325_20261006/results/per_sample/{cond}/{u}.json'
    if not os.path.exists(p):
        return None
    d = json.load(open(p)); models[(cond.split('_')[0] if cond.startswith('oracle_rgb_') else cond, d.get('model'))] += 1
    if d.get('completed_at'):
        times.append(d['completed_at'])
    return d


for u in sorted(mvc_final):
    for cond in ['text_only', 'original_only', 'oracle_rgb'] + MODS:
        d = gload(u, cond)
        if d is not None:
            GC[(u, cond)] = truthy(d['correct']) or d['correct'] is True
            if d.get('question_type'):
                GT[u] = d['question_type']
for u in mvc_final:
    GT.setdefault(u, cat[u])
print('\nGemini MVC models', dict(models), 'completed_at range', min(times), max(times))
missing = [(u, c) for u in mvc_final for c in ['text_only', 'original_only', 'oracle_rgb'] + MODS if (u, c) not in GC]
print('Gemini MVC final-set missing records:', len(missing))
gc = {u: GT.get(u) for u in mvc_final}
P_T7_CV = {'Observed view': [60.5, 59.8, 73.7, 66.3], '+ Endpoint RGB': [75.3, 62.2, 90.5, 78.7], 'Top-down': [80.2, 73.2, 94.9, 85.0],
           'Text': [80.2, 65.9, 95.6, 83.3], 'Video': [80.2, 72.0, 96.4, 85.3], 'Depth': [81.5, 68.3, 94.9, 84.0],
           'Normals': [79.0, 68.3, 95.6, 83.7], 'Panorama': [80.2, 69.5, 93.4, 83.3], 'Oracle selection': [84.0, 79.3, 97.8, 89.0],
           'Question only': [None, None, None, 29.7]}
K = lambda u, c: GC[(u, c)]
cols = [('Observed view', rows_by_type(mvc_final, gc, lambda u: K(u, 'original_only'), CV_T)),
        ('+ Endpoint RGB', rows_by_type(mvc_final, gc, lambda u: K(u, 'oracle_rgb'), CV_T))]
for m in REPS:
    cols.append((NM[m], rows_by_type(mvc_final, gc, lambda u, m=m: K(u, 'oracle_rgb') or K(u, 'oracle_rgb_' + m), CV_T)))
cols.append(('Oracle selection', rows_by_type(mvc_final, gc, lambda u: K(u, 'oracle_rgb') or any(K(u, x) for x in MODS), CV_T)))
cols.append(('Question only', rows_by_type(mvc_final, gc, lambda u: K(u, 'text_only'), CV_T)))
emit_table('T7 Gemini oracle CV', cols, CV_T, P_T7_CV)
check('T2 Gemini question-only CV', 29.7, VALS['T7 Gemini oracle CV | Question only | ALL'])
check('T2 Gemini observed CV', 66.3, VALS['T7 Gemini oracle CV | Observed view | ALL'])
check('T2 Gemini +oracle RGB CV', 78.7, VALS['T7 Gemini oracle CV | + Endpoint RGB | ALL'])
for m, pv in zip(REPS, [6.3, 4.7, 6.7, 5.3, 5.0, 4.7]):
    e2 = sum(K(u, 'oracle_rgb') for u in mvc_final)
    check(f'T8 Gemini CV delta {m} (count-based)', pv, pct(sum(K(u, 'oracle_rgb') or K(u, 'oracle_rgb_' + m) for u in mvc_final) - e2, 300))
check('T8 Gemini CV delta oracle selection', 10.3, pct(sum(K(u, 'oracle_rgb') or any(K(u, x) for x in MODS) for u in mvc_final) - e2, 300))

# ---------------------------------------------------------------- GPT estimated VI (Table 4)
ES = E + '03_estimated_viewpoint_gpt61sol/gpt_runs/results/'
EST = {c: load_dir(ES + c) for c in ['start_final'] + ['start_final_' + m for m in REPS] + ['start_final_allreps']}
print('\nGPT estimated VI models', collections.Counter(r['model'] for d in EST.values() for r in d.values()), mtimes(ES + 'start_final'))
EC = lambda c, u: truthy(EST[c][u]['correct'])
P_T4 = {'Observed view': [60.4, 38.2, 39.4, 51.0, 68.4, 70.0, 57.2], '+ Est. endpoint RGB': [52.7, 45.5, 47.1, 49.0, 53.1, 67.6, 54.7],
        'Top-down': [63.7, 56.4, 50.0, 57.7, 62.2, 72.4, 61.9], 'Text': [60.4, 54.5, 51.9, 56.7, 69.4, 73.5, 62.9],
        'Video': [62.6, 61.8, 54.8, 59.6, 70.4, 72.4, 64.6], 'Depth': [57.1, 54.5, 52.9, 61.5, 66.3, 71.2, 62.2],
        'Normals': [57.1, 52.7, 52.9, 58.7, 67.3, 71.2, 61.7], 'Panorama': [70.3, 60.0, 56.7, 59.6, 63.3, 75.3, 65.6]}
cols = [('Observed view', rows_by_type(fo_uids, cat, lambda u: C('observed', u), VI_T)),
        ('+ Est. endpoint RGB', rows_by_type(fo_uids, cat, lambda u: EC('start_final', u), VI_T))]
for m in REPS:
    cols.append((NM[m], rows_by_type(fo_uids, cat, lambda u, m=m: EC('start_final', u) or EC('start_final_' + m, u), VI_T)))
emit_table('T4 GPT estimated VI', cols, VI_T, P_T4)
ptb = {}
for t in VI_T:
    us = [u for u in fo_uids if cat[u] == t]
    ptb[t] = max(sum(EC('start_final', u) or EC('start_final_' + m, u) for u in us) for m in REPS)
    check(f'T4 GPT estimated VI | Per-type best | {t}', [70.3, 61.8, 56.7, 61.5, 70.4, 75.3][VI_T.index(t)], pct(ptb[t], len(us)))
check('T4 GPT estimated VI | Per-type best | ALL (sum of per-type maxima / 622)', 67.2, pct(sum(ptb.values()), 622))

# ---------------------------------------------------------------- GPT estimated CV (Table 5)
MV = E + '04_estimated_crossview_gpt61sol/gpt_runs/results/'
EM = {c: load_dir(MV + c) for c in ['est_rgb', 'est_rgb_depth']}
print('\nGPT estimated CV models', collections.Counter(r['model'] for d in EM.values() for r in d.values()), mtimes(MV + 'est_rgb'))
MC = lambda c, u: truthy(EM[c][u]['correct'])
cols = [('Observed view', rows_by_type(mvc_final, cat, lambda u: C('observed', u), CV_T)),
        ('+ Est. RGB', rows_by_type(mvc_final, cat, lambda u: MC('est_rgb', u), CV_T)),
        ('+ Est. depth', rows_by_type(mvc_final, cat, lambda u: MC('est_rgb', u) or MC('est_rgb_depth', u), CV_T))]
emit_table('T5 GPT estimated CV', cols, CV_T, {'Observed view': [56.8, 59.8, 79.6, 68.0], '+ Est. RGB': [74.1, 54.9, 89.8, 76.0],
                                                 '+ Est. depth': [79.0, 65.9, 93.4, 82.0]})

# ---------------------------------------------------------------- GEARS + gating ablation (Table 6, Table 10)
obsP = lambda u: GO['observed'][u]['prediction']
S2 = load_dir(E + '05_gears/viewpoint_gating_routing/results/gate2sel_S2')
S2NE = load_dir(E + '05_gears/ablation/results/gate2sel_S2NE')
S2MV = load_dir(E + '05_gears/crossview_gating_routing/results_gate2/S2MV')
S2MVN = load_dir(E + '05_gears/ablation/results/gate2sel_S2MVN')
NG = load_dir(E + '05_gears/ablation/results/start_final_allreps')
print('\nGate-2 choices: S2', collections.Counter(r['choice'] for r in S2.values()), '| S2MV', collections.Counter(r['choice'] for r in S2MV.values()),
      '| S2NE', collections.Counter(r['choice'] for r in S2NE.values()), '| S2MVN', collections.Counter(r['choice'] for r in S2MVN.values()))


def vi_lookup(u, ch):
    if ch == 'start_only':
        return C('observed', u)
    if ch in ('est_rgb', 'start_final'):
        return EC('start_final', u)
    return EC('start_final_' + ch, u)


def cv_lookup(u, ch):
    return C('observed', u) if ch == 'start_only' else MC('est_rgb', u) if ch == 'est_rgb' else MC('est_rgb_depth', u)


vi_agree = lambda u: obsP(u) == EST['start_final'][u]['prediction']
cv_agree = lambda u: obsP(u) == EM['est_rgb'][u]['prediction']
dis_vi = [u for u in fo_uids if not vi_agree(u)]; dis_cv = [u for u in mvc_final if not cv_agree(u)]
print('gate-1 disagreements VI', len(dis_vi), 'covered by S2:', set(dis_vi) == set(S2), '| CV', len(dis_cv), 'covered by S2MV:', set(dis_cv) == set(S2MV))
gears_vi = lambda u: EC('start_final', u) if vi_agree(u) else vi_lookup(u, S2[u]['choice'])
gears_cv = lambda u: MC('est_rgb', u) if cv_agree(u) else cv_lookup(u, S2MV[u]['choice'])
cols = [('No gate', rows_by_type(fo_uids, cat, lambda u: truthy(NG[u]['correct']), VI_T)),
        ('+Gating', rows_by_type(fo_uids, cat, lambda u: EC('start_final', u) if vi_agree(u) else truthy(NG[u]['correct']), VI_T)),
        ('+Routing', rows_by_type(fo_uids, cat, lambda u: vi_lookup(u, S2NE[u]['choice']), VI_T)),
        ('Gating+Routing', rows_by_type(fo_uids, cat, gears_vi, VI_T))]
emit_table('T10a gating ablation VI', cols, VI_T, {'No gate': [52.7, 49.1, 44.2, 54.8, 58.2, 61.8, 54.7], '+Gating': [60.4, 52.7, 44.2, 54.8, 63.3, 68.8, 58.8],
                                                     '+Routing': [56.0, 49.1, 43.3, 52.9, 68.4, 70.0, 58.5], 'Gating+Routing': [63.7, 54.5, 48.1, 55.8, 68.4, 70.0, 61.4]})
cols = [('No gate', rows_by_type(mvc_final, cat, lambda u: MC('est_rgb_depth', u), CV_T)),
        ('+Gating', rows_by_type(mvc_final, cat, lambda u: MC('est_rgb', u) if cv_agree(u) else MC('est_rgb_depth', u), CV_T)),
        ('+Routing', rows_by_type(mvc_final, cat, lambda u: cv_lookup(u, S2MVN[u]['choice']), CV_T)),
        ('Gating+Routing', rows_by_type(mvc_final, cat, gears_cv, CV_T))]
emit_table('T10b gating ablation CV', cols, CV_T, {'No gate': [72.8, 61.0, 89.8, 77.3], '+Gating': [71.6, 61.0, 88.3, 76.3],
                                                     '+Routing': [74.1, 61.0, 89.8, 77.7], 'Gating+Routing': [74.1, 61.0, 89.8, 77.7]})

# ---------------------------------------------------------------- MJ / AVIC (Table 6)
T = E + '06_baselines_mindjourney_avic/runs_gpt61sol'


def qa(s, m, v):
    return {r['qid']: r for r in (json.load(open(f)) for f in glob.glob(f'{T}/qa/{s}/{m}/{v}/*.json'))}


def raw(m, s):
    pat = f'{T}/results/A/{m}_{s}*/*/*/gpt.json' if m == 'mj' else f'{T}/results/A/{m}_{s}*/*/*/step_0/gpt.json'
    res = {}
    for g in glob.glob(pat):
        x = json.load(open(g)); res[x['question']['database_idx']] = x.get('result') == 'correct'
    return res


B = {('fo622', 'mj'): qa('fo622', 'mj', 'top3+label'), ('fo622', 'avic'): qa('fo622', 'avic', 'label'),
     ('mvc300', 'mj'): qa('mvc300', 'mj', 'top3+label'), ('mvc300', 'avic'): qa('mvc300', 'avic', 'label')}
RAW = {(s, m): raw(m, s) for s in ('fo622', 'mvc300') for m in ('mj', 'avic')}
print('\nMJ/AVIC final-QA record counts', {k: len(v) for k, v in B.items()}, '| raw pipeline counts', {k: len(v) for k, v in RAW.items()})
fin = lambda s, m, u: truthy(B[(s, m)][u]['correct']) if u in B[(s, m)] else RAW[(s, m)][u]
P6V = {'MindJourney': [53.8, 52.7, 45.2, 53.8, 69.4, 67.1, 58.4], 'AVIC': [51.6, 49.1, 46.2, 51.0, 63.3, 62.4, 55.1],
       'GEARS': [63.7, 54.5, 48.1, 55.8, 68.4, 70.0, 61.4]}
P6C = {'MindJourney': [61.7, 62.2, 81.8, 71.0], 'AVIC': [59.3, 51.2, 85.4, 69.0], 'GEARS': [74.1, 61.0, 89.8, 77.7]}
emit_table('T6 baselines VI', [('MindJourney', rows_by_type(fo_uids, cat, lambda u: fin('fo622', 'mj', u), VI_T)),
                               ('AVIC', rows_by_type(fo_uids, cat, lambda u: fin('fo622', 'avic', u), VI_T)),
                               ('GEARS', rows_by_type(fo_uids, cat, gears_vi, VI_T))], VI_T, P6V)
emit_table('T6 baselines CV', [
    ('MindJourney', rows_by_type(mvc_final, cat, lambda u: fin('mvc300', 'mj', u), CV_T)),
    ('AVIC', rows_by_type(mvc_final, cat, lambda u: fin('mvc300', 'avic', u), CV_T)),
    ('GEARS', rows_by_type(mvc_final, cat, gears_cv, CV_T))], CV_T, P6C)

# ---------------------------------------------------------------- prose numbers derived from the above
P = VALS
check('Intro/4.2: GPT Farthest Object observed', 68.4, P['T1 GPT oracle VI | Observed view | farthest_object'])
check('Intro/4.2: GPT Farthest Object + oracle RGB', 58.2, P['T1 GPT oracle VI | + Endpoint RGB | farthest_object'])
check('Intro: GPT Farthest Object top-down', 75.5, P['T1 GPT oracle VI | Top-down | farthest_object'])
check('4.2: Gemini observed VI (prose 53.7)', 53.7, P['T2 Gemini observed VI'])
check('4.3: GPT panorama open-space gain', 16.5, round(P['T1 GPT oracle VI | Panorama | open_space_direction'] - P['T1 GPT oracle VI | + Endpoint RGB | open_space_direction'], 1))
check('4.3: Gemini panorama open-space gain', 16.5, round(P['T7 Gemini oracle VI | Panorama | open_space_direction'] - P['T7 Gemini oracle VI | + Endpoint RGB | open_space_direction'], 1))
wins = 0
for name, types in [('T6 baselines VI', VI_T), ('T6 baselines CV', CV_T)]:
    for t in types:
        g = P[f'{name} | GEARS | {t}']
        wins += g >= max(P[f'{name} | MindJourney | {t}'], P[f'{name} | AVIC | {t}'])
check('Sec5: GEARS best on N of 9 question types', 7, wins)

# ---------------------------------------------------------------- benchmark statistics + nav stats
canon = [json.loads(l) for l in open(E + '03_estimated_viewpoint_gpt61sol/canonical_622_rows/canonical_622_rows.jsonl')]
assert sorted(r['question_id'] for r in canon) == fo_uids
check('Sec3.4: VI routes (trajectories)', 200, len({r['trajectory_id'] for r in canon}))
check('Sec3.4: VI scenes', 73, len({r['scan_id'] for r in canon}))
check('Sec3.4: CV view groups', 143, len({u.split('__')[1].rsplit('-', 1)[0] for u in mvc_final}))
scans = {}
for u in mvc_final:
    rid = u.split('__')[1].rsplit('-', 1)[0]
    mf = E + f'02_oracle_gemini37/crossview_integration/mvc_modalities_20261005/resources/{rid}/modalities_manifest.json'
    scans[rid] = json.load(open(mf)).get('scan_id') if os.path.exists(mf) else None
print('MVC resources without a local manifest copy:', sorted(r for r, s in scans.items() if s is None))
check('Sec3.4: CV scenes', 21, len({s for s in scans.values() if s}))
for t, pv in zip(VI_T + CV_T, [91, 55, 104, 104, 98, 170, 81, 82, 137]):
    check(f'T3 question count {t}', pv, sum(cat[u] == t for u in fo_uids + mvc_final))
# nav stats (navstat.py logic) + oracle route length range
REF = json.load(open(E + '03_estimated_viewpoint_gpt61sol/generation_code/reference_geometry/evaluation_only/reference_geometry.json'))
scale = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(E + '03_estimated_viewpoint_gpt61sol/generation_code/trajectory_plans_v5/scale/*.json')}
wrap = lambda a: (a + 180) % 360 - 180
errs, yaws, olens, edges = [], [], [], []
for p in glob.glob(E + '03_estimated_viewpoint_gpt61sol/generation_code/trajectory_plans_v5/v5/*/*/plan.json'):
    d = json.load(open(p, encoding='utf-8')); q = d['question_id']; sid = q.rsplit('-', 1)[0]; r = REF[q]
    f = d['final']; mpu = d['m_per_unit']; unit = [f['x'] / mpu, f['y'] / mpu]
    errs.append(math.dist([v * scale[sid]['m_per_unit_moge'] for v in unit], r['points_m'][-1][:2]))
    yaws.append(abs(wrap(f['yaw_deg'] - math.degrees(r['final_yaw']))))
    olens.append(sum(math.dist(a[:2], b[:2]) for a, b in zip(r['points_m'], r['points_m'][1:]))); edges.append(len(r['points_m']) - 1)
print(f'\nNav plans: n={len(errs)}; oracle route length min {min(olens):.2f} m, max {max(olens):.2f} m, median {statistics.median(olens):.2f}; '
      f'points-1 per route {dict(collections.Counter(edges))}')
check('Appendix: median planned endpoint error m (MoGe scale)', 1.4, round(statistics.median(errs), 1))
check('Appendix: median planned heading error deg', 14, round(statistics.median(yaws)))
check('Appendix: median route length m', 4.2, round(statistics.median(olens), 1))

print('\n\n## ALL CHECKS (paper vs recomputed)')
print('| item | paper | recomputed | status |\n|---|---|---|---|')
for it, pv, gv, st in CHECKS:
    print(f'| {it} | {pv} | {gv} | {st} |')
bad = [c for c in CHECKS if c[3] == 'MISMATCH']
print(f'\n{len(CHECKS)} checks, {len(bad)} mismatches')
for c in bad:
    print('MISMATCH:', c)
json.dump({'values': VALS, 'checks': CHECKS}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verification_values.json'), 'w'), indent=1, default=str)
