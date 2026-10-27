"""Repeat rounds of the Table-5 estimated-evidence run with Gemini 3.7 Flash (same requests, same settings).

Usage: python run_rounds.py <payload_dir> --rounds 1,2 --workers N [--limit K]
Request = make_parts() order from the original runner; config from <payload>/config.json (temperature 0, HIGH
thinking, 8192 tokens, text/x.enum over the row's choices, frozen system prompt); video inline at 12 fps.
Results: <payload>/results/round<k>/<condition>/<uid>.json (resume-safe). Exception text and keys are never stored.
"""
import json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from google import genai
from google.genai import types

P = Path(sys.argv[1]); arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ROUNDS = [int(x) for x in arg('--rounds', '1,2').split(',')]; WORKERS = int(arg('--workers', '8')); LIMIT = arg('--limit')
CFG = json.loads((P / 'config.json').read_text()); assert CFG['thinking'] == 'HIGH' and CFG['temperature'] == 0
TASKS = [json.loads(l) for l in open(P / 'tasks.jsonl', encoding='utf-8')]
if LIMIT:
    TASKS = TASKS[:int(LIMIT)]
_tl = threading.local(); _lock = threading.Lock(); stats = {'ok': 0, 'retries': 0, 'failed': 0, 'lat': []}


def client():
    if getattr(_tl, 'c', None) is None:
        _tl.c = genai.Client(api_key=os.environ['GEMINI_API_KEY'], http_options=types.HttpOptions(timeout=300_000))
    return _tl.c


def parts(t):
    out = []
    for p in t['parts']:
        if 'text' in p:
            out.append(types.Part.from_text(text=p['text']))
        elif 'image' in p:
            out.append(types.Part.from_bytes(data=(P / 'assets' / p['image']).read_bytes(), mime_type='image/png'))
        else:
            out.append(types.Part(inline_data=types.Blob(mime_type='video/mp4', data=(P / 'assets' / p['video']).read_bytes()),
                                  video_metadata=types.VideoMetadata(fps=p['fps'])))
    return out


def job(item):
    rnd, t = item
    out = P / 'results' / f'round{rnd}' / t['condition'] / f"{t['uid']}.json"
    if out.exists():
        return
    conf = types.GenerateContentConfig(system_instruction=CFG['system_prompt'], temperature=0, max_output_tokens=CFG['max_output_tokens'],
                                       thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
                                       response_mime_type='text/x.enum', response_schema={'type': 'STRING', 'enum': t['valid']})
    for attempt in range(1, 9):
        t0 = time.monotonic()
        try:
            r = client().models.generate_content(model=CFG['model'], contents=[types.Content(role='user', parts=parts(t))], config=conf)
            pred = (r.text or '').strip().upper()
            if pred not in t['valid']:
                raise ValueError('invalid enum')
            break
        except Exception as e:
            with _lock:
                stats['retries'] += 1
            print(f'retry r{rnd} {t["uid"]} {t["condition"]} {type(e).__name__}', flush=True)
            if attempt == 8:
                with _lock:
                    stats['failed'] += 1
                return
            time.sleep(min(60, 2 ** attempt) + random.random())
    lat = time.monotonic() - t0; u = r.usage_metadata
    rec = {'uid': t['uid'], 'category': t['category'], 'condition': t['condition'], 'round': rnd, 'model': CFG['model'],
           'prediction': pred, 'gold': t['gold'], 'correct': pred == t['gold'], 'latency_s': round(lat, 2), 'attempts': attempt,
           'usage': u.model_dump(mode='json') if u else {}}
    out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec), encoding='utf-8'); os.replace(tmp, out)
    with _lock:
        stats['ok'] += 1; stats['lat'].append(lat)
        if stats['ok'] % 200 == 0:
            print(f"ok {stats['ok']} retries {stats['retries']} failed {stats['failed']}", flush=True)


# interleave rounds so both progress together
items = [(rnd, t) for t in TASKS for rnd in ROUNDS]
t0 = time.time()
with ThreadPoolExecutor(WORKERS) as ex:
    list(ex.map(job, items))
lat = sorted(stats['lat'])
print(json.dumps({'workers': WORKERS, 'ok': stats['ok'], 'retries': stats['retries'], 'failed': stats['failed'],
                  'wall_s': round(time.time() - t0), 'lat_med': lat[len(lat) // 2] if lat else None}), flush=True)
