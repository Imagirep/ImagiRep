"""Server side: map every estimated topdown asset used by the 622 estimated requests to its package, plan, scene and eye."""
import glob, hashlib, json, os, re, sys
TASK = '<external>/hyworld2_nlp26_full_20261001'
E = '03_estimated_viewpoint_gpt61sol/estimated_assets/request_source'
LATEST = TASK + '/outputs/hy_latest_scenes_20261002/'
OUT = sys.argv[1]

need = {}
for l in open(E + '/tasks.jsonl'):
    t = json.loads(l)
    if t['condition'] != 'start_final_topdown':
        continue
    ps = t['parts']
    k = next(i for i, p in enumerate(ps) if 'text' in p and p['text'].startswith('ADDITIONAL TOP-DOWN MAP'))
    img = next(p['image'] for p in ps[k:] if 'image' in p)
    need.setdefault(img, []).append(t['uid'])
print('questions', sum(map(len, need.values())), 'unique topdown assets', len(need))

h16 = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
asset_name_rule = all(h16(f'{E}/assets/{a}')[:16] == a[:-4] for a in list(need)[:20])
print('asset name = sha256[:16]:', asset_name_rule)

cands = glob.glob(TASK + '/outputs/hy_nav_pilot_20261002/packages/*/*/*/Topdown.png') + \
        glob.glob(TASK + '/outputs/*/variants/*/Topdown.png')
by_hash = {}
for c in cands:
    by_hash.setdefault(h16(c)[:16], []).append(c)
print('candidate package topdowns', len(cands))

jobs, missing = [], []
for a, uids in need.items():
    hits = by_hash.get(a[:-4], [])
    if not hits:
        missing.append(a); continue
    d = os.path.dirname(hits[0]); man = json.load(open(d + '/manifest.json'))
    eye = float(re.search(r'eye height prior ([0-9.]+) m', man['scale']).group(1))
    if '/variants/' in d:
        name = os.path.basename(d); plan = os.path.dirname(os.path.dirname(d)) + f'/plans/{name}.json'
        sid = name.split('__')[0]
    else:
        tag, sid = d.split('/')[-3], d.split('/')[-2]
        plan = TASK + f'/outputs/hy_nav_pilot_20261002/plans/{tag}/{sid}/{man["question_id"]}.json'
    jobs.append({'asset': a, 'uids': uids, 'package_topdowns': hits, 'ref': f'{E}/assets/{a}', 'plan': plan,
                 'scene': LATEST + sid, 'eye': eye, 'plan_exists': os.path.exists(plan), 'scene_exists': os.path.isdir(LATEST + sid)})
print('jobs', len(jobs), 'missing', len(missing), missing[:5])
print('plan missing', sum(not j['plan_exists'] for j in jobs), 'scene missing', sum(not j['scene_exists'] for j in jobs))
print('package copies per asset', {n: sum(len(j['package_topdowns']) == n for j in jobs) for n in {len(j['package_topdowns']) for j in jobs}})
json.dump(jobs, open(OUT, 'w'), indent=1)
