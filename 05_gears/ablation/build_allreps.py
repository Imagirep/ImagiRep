"""No-gate VI requests: observed + est RGB + ALL six estimated representations in one request (2026-10-11).
Built from the GPT-6.1-sol estimated payload (gpt61sol_estimated_622_20261009): the `start_final` request (Start, FINAL VIEW,
question) with the extra parts of each `start_final_<m>` request inserted before the question, in the order
top-down, text, video (16 frames), depth, surface normals, panorama. Captions are unchanged.
Usage: python build_allreps.py <payload_dir> <uids.json>   Output: <payload_dir>/tasks_allreps.jsonl
"""
import json, sys
P, U = sys.argv[1], set(json.load(open(sys.argv[2])))
MODS = ['topdown', 'text', 'video', 'depth', 'surface_normals', 'panorama']
T = {}
for l in open(f'{P}/tasks.jsonl', encoding='utf-8'):
    t = json.loads(l)
    if t['uid'] in U:
        T[(t['uid'], t['condition'])] = t
out = []
for u in sorted(U):
    base = T[(u, 'start_final')]; parts = base['parts'][:4]
    assert [list(p)[0] for p in parts] == ['text', 'image', 'text', 'image'] and parts[2]['text'].startswith('FINAL VIEW')
    for m in MODS:
        x = T[(u, 'start_final_' + m)]['parts']
        assert x[:4] == base['parts'][:4] and x[-1] == base['parts'][-1]
        parts = parts + x[4:-1]
    out.append({**base, 'condition': 'start_final_allreps', 'parts': parts + [base['parts'][-1]]})
with open(f'{P}/tasks_allreps.jsonl', 'w', encoding='utf-8') as f:
    for t in out:
        f.write(json.dumps(t) + '\n')
n_img = [sum('image' in p for p in t['parts']) for t in out]
print(json.dumps({'tasks': len(out), 'images_per_request': [min(n_img), max(n_img)]}))
