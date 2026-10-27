"""Two-step gating for GPT-6.1-sol on MVC300 with estimated evidence (2026-10-10; same design as single view).
Step 1 (rule A): if the observed-view answer and the observed + estimated RGB answer agree, keep the latter.
Step 2 (this script, version S2MV): for the remaining questions a prompt picks one option; the final answer is looked up:
  start_only -> observed answer;  est_rgb -> observed + estimated RGB answer;  depth -> observed + estimated RGB + depth answer.
Usage: python gate2_mvc.py <payload_dir> --uids f [--workers N]   Output: results_gate2/S2MV/<uid>.json
"""
import base64, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI
HERE = Path(__file__).resolve().parent
P = sys.argv[1]
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
W = int(arg('--workers', '12')); OUT = HERE / 'results_gate2' / 'S2MV'
keep = set(json.load(open(arg('--uids'))))
TASKS = [t for t in (json.loads(l) for l in open(f'{P}/tasks.jsonl')) if t['condition'] == 'est_rgb' and t['uid'] in keep]
OPTIONS = ['start_only', 'est_rgb', 'depth']
SYSTEM = 'You decide what evidence to use for a spatial reasoning question. You do not answer the question.'
NOTE = ('Note: the Estimated RGB above was not photographed. It was generated as an additional view of the same scene and '
        'may be blurry or wrong in places.')
TASK = ('Answering from the numbered images alone and answering from the numbered images plus the Estimated RGB gave '
        'DIFFERENT answers to this question. Choose ONE option for the final answer:\n'
        '- start_only: rely on the numbered images alone and set the Estimated RGB aside.\n'
        '- est_rgb: use the numbered images together with the Estimated RGB.\n'
        '- depth: use the numbered images, the Estimated RGB and an estimated depth map of the Estimated RGB (distance of '
        'each visible surface from the camera).\n'
        'Guidance for this kind of benchmark:\n'
        '- Which object is farthest from another object: distances are hard to judge from color images alone, so prefer depth.\n'
        '- Which object is nearest to another object, or the left-to-right order of objects: the Estimated RGB connects the '
        'views well, so prefer est_rgb.\n'
        'Reply with one short sentence giving your reason, then the choice.')
FMT = {'type': 'json_schema', 'name': 'gate2mv', 'strict': True,
       'schema': {'type': 'object', 'properties': {'reason': {'type': 'string'}, 'choice': {'type': 'string', 'enum': OPTIONS}},
                  'required': ['reason', 'choice'], 'additionalProperties': False}}
_tl = threading.local()


def client():
    if not hasattr(_tl, 'c'):
        _tl.c = OpenAI(api_key=os.environ['OPENAI_LAB_API_KEY'].strip(), timeout=600)
    return _tl.c


def content(t):
    out = []
    for p in t['parts'][:-1]:
        if 'text' in p:
            out.append({'type': 'input_text', 'text': p['text']})
        else:
            b = base64.b64encode(open(f'{P}/images/{p["image"]}', 'rb').read()).decode()
            out.append({'type': 'input_image', 'image_url': f'data:image/png;base64,{b}'})
    q = t['parts'][-1]['text']; q = q[:q.rfind('Answer:')].rstrip()
    out.append({'type': 'input_text', 'text': '\n' + NOTE + '\n\nThe question the reasoner must answer (do not answer it yourself):\n'
                + q[q.find('Question:'):] + '\n\n' + TASK})
    return out


def job(t):
    out = OUT / f"{t['uid']}.json"
    if out.exists():
        return
    for attempt in range(1, 9):
        try:
            r = client().responses.create(model='gpt-6.1-sol', instructions=SYSTEM, input=[{'role': 'user', 'content': content(t)}],
                                          reasoning={'effort': 'high'}, max_output_tokens=8192, store=False,
                                          text={'verbosity': 'low', 'format': FMT})
            x = json.loads(r.output_text)
            break
        except Exception as e:  # noqa: BLE001
            print(f'retry {attempt} {t["uid"]} {type(e).__name__} {getattr(e, "status_code", None)}', flush=True)
            if attempt == 8:
                return
            time.sleep(min(120, 2 ** attempt + random.random()))
    rec = dict(uid=t['uid'], category=t['category'], **x, usage=r.usage.model_dump(mode='json') if r.usage else {})
    OUT.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec)); os.replace(tmp, out)


with ThreadPoolExecutor(W) as ex:
    list(ex.map(job, TASKS))
recs = [json.loads(f.read_text()) for f in OUT.glob('*.json')]
usd = sum((r['usage'].get('input_tokens', 0) * 2 + r['usage'].get('output_tokens', 0) * 10) / 1e6 for r in recs)
print(json.dumps({'n': len(recs), 'usd': round(usd, 2)}))
