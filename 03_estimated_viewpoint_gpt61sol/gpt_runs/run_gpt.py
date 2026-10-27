"""Run GPT-6.1-sol on tasks.jsonl (2026-10-07).
  python run_gpt.py <payload_dir> --workers N [--limit K] [--tasks tasks_mod.jsonl]
Settings follow the existing OpenAI path of the FO runner: Responses API, reasoning effort high, JSON-schema enum
over the valid letters, verbosity low, store=False; max_output_tokens 8192 (reasoning tokens count toward it).
Per-request results: <payload>/results/<condition>/<uid>.json (resume-safe). Exception text is never stored.
"""
import base64, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

P = sys.argv[1]
W = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 8
LIMIT = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
TASKS = sys.argv[sys.argv.index('--tasks') + 1] if '--tasks' in sys.argv else 'tasks.jsonl'
MODEL = 'gpt-6.1-sol'
_tl = threading.local(); lock = threading.Lock(); stats = {'ok': 0, 'retry': 0, 'fail': 0, 'lat': []}


KEYS = ['OPENAI_LAB_API_KEY', 'OPENAI_LAB_API_KEY_2']   # second key once the first has no credit left
_key = {'i': 0}


def client():
    if getattr(_tl, 'k', None) != _key['i']:
        _tl.c = OpenAI(api_key=os.environ[KEYS[_key['i']]].strip(), timeout=600); _tl.k = _key['i']
    return _tl.c


def content(t):
    out = []
    for p in t['parts']:
        if 'text' in p:
            out.append({'type': 'input_text', 'text': p['text']})
        else:
            b = base64.b64encode(open(f'{P}/images/{p["image"]}', 'rb').read()).decode()
            out.append({'type': 'input_image', 'image_url': f'data:image/png;base64,{b}'})
    return out


def run(t):
    out = f'{P}/results/{t["condition"]}/{t["uid"]}.json'
    if os.path.exists(out):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(1, 9):
        t0 = time.monotonic()
        try:
            r = client().responses.create(
                model=MODEL, instructions=t['system'], input=[{'role': 'user', 'content': content(t)}],
                reasoning={'effort': 'high'}, max_output_tokens=8192, store=False,
                text={'verbosity': 'low', 'format': {'type': 'json_schema', 'name': 'answer_letter', 'strict': True,
                      'schema': {'type': 'object', 'properties': {'answer': {'type': 'string', 'enum': t['valid']}},
                                 'required': ['answer'], 'additionalProperties': False}}})
            raw = r.output_text or ''
            pred = json.loads(raw)['answer'].strip().upper()
            assert pred in t['valid']
            rec = dict(uid=t['uid'], benchmark=t['benchmark'], category=t['category'], condition=t['condition'], model=MODEL,
                       prediction=pred, gold=t['gold'], correct=pred == t['gold'], latency_s=round(time.monotonic() - t0, 2),
                       usage=r.usage.model_dump(mode='json') if r.usage else {}, attempts=attempt, status=getattr(r, 'status', None))
            json.dump(rec, open(out + '.tmp', 'w')); os.replace(out + '.tmp', out)
            with lock:
                stats['ok'] += 1; stats['lat'].append(rec['latency_s'])
            return
        except Exception as e:  # noqa: BLE001
            code = getattr(e, 'status_code', None)
            with lock:
                stats['retry'] += 1
                if code == 429 and ('insufficient_quota' in str(e) or 'credit_balance_exhausted' in str(e)) and _key['i'] == 0 and os.environ.get(KEYS[1]):
                    _key['i'] = 1; print('switching to', KEYS[1], flush=True)
            print(f'retry {attempt} {t["uid"]} {t["condition"]} {type(e).__name__} {code}', flush=True)
            time.sleep(min(120, 2 ** attempt + random.random()))
    with lock:
        stats['fail'] += 1


if __name__ == '__main__':
    tasks = [json.loads(l) for l in open(f'{P}/{TASKS}', encoding='utf-8')]
    if LIMIT:
        tasks = tasks[:LIMIT]
    t0 = time.time()
    with ThreadPoolExecutor(W) as ex:
        list(ex.map(run, tasks))
    dt = time.time() - t0; lat = sorted(stats['lat'])
    print(json.dumps(dict(workers=W, ok=stats['ok'], retries=stats['retry'], failed=stats['fail'], wall_s=round(dt),
                          per_min=round(60 * stats['ok'] / max(1, dt), 1), lat_med=lat[len(lat) // 2] if lat else None,
                          lat_max=lat[-1] if lat else None)), flush=True)
