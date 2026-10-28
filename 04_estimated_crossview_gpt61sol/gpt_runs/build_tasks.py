"""GPT-6.1-sol on MVC300 with estimated (generated) evidence (2026-10-10).

Generated data: inputs/estimated_rgb_depth/, one folder per scene
(<MVC-xxx-Qn>/rgb.png, depth.png). MVC300 has 143 scenes; all questions of a scene share the same observed images (same
order), the same Oracle RGB and the same oracle depth, so each scene's generated pair applies to all its questions.

Requests start from the GPT-6.1-sol payload (gpt61sol_observed_oracle_922_20261007):
  est_rgb        = the `oracle` request with the Oracle RGB image replaced by the generated rgb.png
  est_rgb_depth  = the `oracle_depth` request with the Oracle RGB replaced by rgb.png and the oracle depth by depth.png
Text changes only (user-approved): "Oracle RGB" -> "Estimated RGB", "ORACLE DEPTH MAP" -> "ESTIMATED DEPTH MAP", and the
depth caption sentence giving this image's metric colour range is removed (no metric scale for the generated depth).
Usage: python build_tasks.py <out_dir> <generated_dir>
"""
import collections, hashlib, json, os, re, shutil, sys
G = '01_oracle_gpt61sol/observed_oracle_8conditions_922'
OUT, GEN = sys.argv[1], sys.argv[2]
os.makedirs(OUT + '/images', exist_ok=True)
scene = lambda u: u.split('__')[-1].rsplit('-', 1)[0]
folder = {f.rsplit('-', 1)[0]: f for f in os.listdir(GEN)}
RANGE = re.compile(r" This image's color scale runs from [0-9.]+ m \(near/red\) to [0-9.]+ m \(far/purple\), with values outside this range clipped\.")


def put(path):
    data = open(path, 'rb').read(); n = hashlib.sha256(data).hexdigest()[:16] + '.png'
    if not os.path.exists(f'{OUT}/images/{n}'):
        open(f'{OUT}/images/{n}', 'wb').write(data)
    return n


def text(s):
    s2 = RANGE.sub('', s).replace('ORACLE DEPTH MAP', 'ESTIMATED DEPTH MAP').replace('Oracle RGB', 'Estimated RGB')
    assert 'racle' not in s2 and 'ORACLE' not in s2, s2[:200]
    return s2


tasks, n_range = [], 0
for fn, cond, new in [('tasks.jsonl', 'oracle', 'est_rgb'), ('tasks_mod.jsonl', 'oracle_depth', 'est_rgb_depth')]:
    for l in open(f'{G}/{fn}'):
        t = json.loads(l)
        if t['benchmark'] != 'MVC300' or t['condition'] != cond:
            continue
        f = folder[scene(t['uid'])]
        rgb, dep = put(f'{GEN}/{f}/rgb.png'), put(f'{GEN}/{f}/depth.png')
        parts, nxt = [], None
        for p in t['parts']:
            if 'text' in p:
                if p['text'].startswith('Oracle RGB:'):
                    nxt = rgb
                elif p['text'].startswith('ORACLE DEPTH MAP'):
                    nxt = dep; n_range += bool(RANGE.search(p['text']))
                else:
                    nxt = None
                parts.append({'text': text(p['text'])})
            else:
                if nxt:
                    parts.append({'image': nxt}); nxt = None
                else:
                    src = f"{G}/images/{p['image']}"
                    if not os.path.exists(f"{OUT}/images/{p['image']}"):
                        shutil.copyfile(src, f"{OUT}/images/{p['image']}")
                    parts.append({'image': p['image']})
        assert rgb in [p.get('image') for p in parts] and (new == 'est_rgb' or dep in [p.get('image') for p in parts])
        tasks.append(dict(t, condition=new, parts=parts, generated_folder=f))
with open(OUT + '/tasks.jsonl', 'w') as fo:
    for t in tasks:
        fo.write(json.dumps(t) + '\n')
print(json.dumps({'tasks': len(tasks), 'by_condition': collections.Counter(t['condition'] for t in tasks),
                  'depth captions with metric range removed': n_range, 'images': len(os.listdir(OUT + '/images'))}))
