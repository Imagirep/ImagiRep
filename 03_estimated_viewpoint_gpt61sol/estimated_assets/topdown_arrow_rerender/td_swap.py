"""Server side: switch the estimated template to the larger-arrow topdown and remove the old topdown files.
Usage: python td_swap.py jobs.json new/ [--apply]   (without --apply: checks only)"""
import glob, hashlib, json, os, shutil, sys
E = '03_estimated_viewpoint_gpt61sol/estimated_assets/request_source'
jobs = json.load(open(sys.argv[1])); NEW = sys.argv[2]; APPLY = '--apply' in sys.argv
h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
old2new = {}
for j in jobs:
    new_name = h(f"{NEW}/{j['asset']}")[:16] + '.png'
    assert new_name != j['asset']
    old2new[j['asset']] = new_name

# old names must occur only as topdown images in the template
tasks = [json.loads(l) for l in open(E + '/tasks.jsonl')]
for t in tasks:
    for p in t['parts']:
        if p.get('image') in old2new:
            assert t['condition'] == 'start_final_topdown', (t['uid'], t['condition'])
# other server copies with an old name (content-hash naming) in experiment asset/image folders
extra = [f for d in glob.glob('<original_experiments_root>/*/assets') + glob.glob('<original_experiments_root>/*/images')
         for f in (os.path.join(d, a) for a in old2new) if os.path.exists(f) and not f.startswith(E + '/')]
pk = [(p, j['asset']) for j in jobs for p in j['package_topdowns']]
print(json.dumps({'old_assets': len(old2new), 'package_copies': len(pk), 'other_experiment_copies': extra}))
if not APPLY:
    sys.exit()
for o, n in old2new.items():
    shutil.copyfile(f'{NEW}/{o}', f'{E}/assets/{n}')
    assert h(f'{E}/assets/{n}')[:16] == n[:-4]
for t in tasks:
    if t['condition'] == 'start_final_topdown':
        t['parts'] = [dict(p, image=old2new[p['image']]) if p.get('image') in old2new else p for p in t['parts']]
tmp = E + '/tasks.jsonl.tmp'
with open(tmp, 'w') as f:
    for t in tasks:
        f.write(json.dumps(t) + '\n')
os.replace(tmp, E + '/tasks.jsonl')
for p, o in pk:
    shutil.copyfile(f'{NEW}/{o}', p)
# verify, then delete the old template assets
tasks = [json.loads(l) for l in open(E + '/tasks.jsonl')]
used = {p['image'] for t in tasks for p in t['parts'] if 'image' in p}
assert not used & set(old2new) and set(old2new.values()) <= used
assert all(h(p)[:16] == old2new[o][:-4] for p, o in pk)
for o in old2new:
    os.remove(f'{E}/assets/{o}')
for f in extra:
    os.remove(f)
json.dump(old2new, open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), 'old2new.json'), 'w'), indent=1)
print(json.dumps({'swapped': len(old2new), 'packages_overwritten': len(pk), 'old_assets_deleted': len(old2new),
                  'other_copies_deleted': len(extra)}))
