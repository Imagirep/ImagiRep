"""Server side: GPT-6.1-sol tasks for the estimated single-view table (622 questions x 7 conditions).

Source: the Gemini 3.7 estimated requests (fo_estimated_622_repeat_rounds_20261007/tasks.jsonl), unchanged except:
  - topdown: the re-rendered larger-arrow topdown (topdown_arrow_rerender_20261009/new/<old asset>);
  - video: the Responses API takes no video, so (as in gpt61sol_observed_oracle_922_20261007/build_mod_tasks.py) the
    caption gets one sentence and the video is given as 16 frames sampled uniformly in time (first and last included).
System prompt = the estimated run's frozen system prompt. Output: <OUT>/tasks.jsonl, <OUT>/images/.
"""
import hashlib, json, os, shutil, sys
import cv2
E = '03_estimated_viewpoint_gpt61sol/estimated_assets/request_source'
NEWTD = '03_estimated_viewpoint_gpt61sol/estimated_assets/topdown_arrow_rerender/new'
OUT = sys.argv[1]; os.makedirs(OUT + '/images', exist_ok=True)
NF = 16
VIDEO_NOTE = f"The video is provided as {NF} frames sampled uniformly in time, shown in order (Frame 1 = first, Frame {NF} = last)."
SYSTEM = json.load(open(E + '/config.json'))['system_prompt']


def put(data, ext):
    n = hashlib.sha256(data).hexdigest()[:16] + ext
    if not os.path.exists(f'{OUT}/images/{n}'):
        open(f'{OUT}/images/{n}', 'wb').write(data)
    return n


def frames(video):
    cap = cv2.VideoCapture(video); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); names = []
    for k in range(NF):
        idx = round(k * (n - 1) / (NF - 1)); cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, fr = cap.read()
        assert ok, (video, idx)
        names.append(put(cv2.imencode('.png', fr)[1].tobytes(), '.png'))
    return names


tasks, n_td = [], 0
for l in open(E + '/tasks.jsonl'):
    t = json.loads(l); parts = []; td_next = False
    for p in t['parts']:
        if 'text' in p:
            td_next = p['text'].startswith('ADDITIONAL TOP-DOWN MAP')
            parts.append(dict(p))
        elif 'video' in p:
            assert 'text' in parts[-1]
            parts[-1]['text'] = parts[-1]['text'].rstrip('\n') + '\n' + VIDEO_NOTE + '\n'
            for i, f in enumerate(frames(f"{E}/assets/{p['video']}"), 1):
                parts += [{'text': f'Frame {i}:\n'}, {'image': f}]
        else:
            src = f"{NEWTD}/{p['image']}" if td_next else f"{E}/assets/{p['image']}"
            n_td += td_next
            parts.append({'image': put(open(src, 'rb').read(), '.png')})
    tasks.append(dict(uid=t['uid'], benchmark='FO622', category=t['category'], condition=t['condition'], system=SYSTEM,
                      parts=parts, valid=t['valid'], gold=t['gold']))
with open(OUT + '/tasks.jsonl', 'w') as f:
    for t in tasks:
        f.write(json.dumps(t) + '\n')
import collections
print(json.dumps({'tasks': len(tasks), 'by_condition': collections.Counter(t['condition'] for t in tasks),
                  'topdown_images_swapped': n_td, 'images': len(os.listdir(OUT + '/images'))}))
