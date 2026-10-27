"""Gate 2 for GPT-6.1-sol, estimated single-view setting (2026-10-10, step-wise gating).

Runs only on questions that gate 1 (rule A: the Start-only answer and the Start + estimated FINAL VIEW answer differ)
sends on. Gate 2 picks one option; the final answer is looked up (no re-answering):
  start_only -> the stored Start-only (observed) answer;  <representation> -> the stored start_final_<repr> answer.
Versions:
  S1: neutral option descriptions + the fact that the two answers disagreed.
  S2: S1 + guidance by question kind.
  S2T: S1 + an alternative wording of the guidance.
Usage: python gate2_select.py <payload_dir> --ver S1|S2 --uids f [--workers N]
Output: results/gate2sel_<ver>/<uid>.json. Key: OPENAI_LAB_API_KEY.
"""
import base64, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gate_prompts as GP

P = sys.argv[1]
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
VER, W = arg('--ver'), int(arg('--workers', '12'))
OUT = HERE / 'results' / f'gate2sel_{VER}'
keep = set(json.load(open(arg('--uids'))))
TASKS = [t for t in (json.loads(l) for l in open(f'{P}/tasks.jsonl')) if t['condition'] == 'start_final' and t['uid'] in keep]
OPTIONS = ['start_only', 'topdown', 'text', 'video', 'depth', 'surface_normals', 'panorama']
SYSTEM = 'You decide what evidence to use for a spatial reasoning question. You do not answer the question.'
DESC = {
    'start_only': 'rely on the START image alone and set the generated FINAL VIEW aside (no generated evidence at all).',
    'topdown': 'an overhead map of the generated scene with the numbered positions, the path and an arrow for the final '
               'facing direction; unlabeled grey geometry, no object names or colors.',
    'text': 'a written step-by-step description of the planned movement (distances, turns, final facing direction); no imagery.',
    'video': 'a first-person video through the generated scene from the START view to the FINAL VIEW.',
    'depth': 'a depth map of the generated final view (distance of each visible surface); only what the final view shows.',
    'surface_normals': 'a surface-normal map of the generated final view (which way each visible surface faces); only what '
                       'the final view shows.',
    'panorama': 'a 360-degree image rendered at the final position in the generated scene, showing all directions around you; '
                'directions the START image never showed are filled in by the world model.',
}
CONTEXT = ('Answering from the START image alone and answering from the START image plus the FINAL VIEW gave DIFFERENT '
           'answers to this question, so the generated FINAL VIEW changed the conclusion and may be misleading. Choose ONE '
           'option for the final answer:\n')
GUIDE = ('Guidance for this kind of benchmark:\n'
         '- Open space or the closest obstacle in some direction around you, or which object is nearest to you after moving: '
         'the panorama shows every direction at the final position, so prefer panorama.\n'
         '- Whether something can be seen after moving: the top-down map shows where things are relative to your final '
         'position and facing, so prefer topdown.\n'
         '- Which object is farthest away, or the direction of a named object relative to you: the generated view tends to '
         'mislead these answers, so prefer start_only.\n')
GUIDE_T = ('Guidance for this kind of benchmark:\n'
           '- Open space or the closest obstacle in some direction around you, or which object is farthest away: the movement '
           'video shows how the view turns and what passes by, so prefer video.\n'
           '- Whether something can be seen after moving: prefer depth.\n'
           '- Which object is nearest to you, or the direction of a named object relative to you: prefer surface_normals.\n')
TAIL = ('Reply with one short sentence giving your reason, then the choice.')


def client():
    if not hasattr(_tl, 'c'):
        _tl.c = OpenAI(api_key=os.environ['OPENAI_LAB_API_KEY'].strip(), timeout=600)
    return _tl.c


_tl = threading.local()


def content(t):
    out = []
    for p in t['parts'][:-1]:
        if 'text' in p:
            out.append({'type': 'input_text', 'text': p['text']})
        else:
            b = base64.b64encode(open(f'{P}/images/{p["image"]}', 'rb').read()).decode()
            out.append({'type': 'input_image', 'image_url': f'data:image/png;base64,{b}'})
    q = t['parts'][-1]['text']; q = q[:q.rfind('Answer:')].rstrip()
    opts = '\n'.join(f'- {k}: {DESC[k]}' for k in OPTIONS)
    instr = CONTEXT + opts + '\n' + {'S2': GUIDE, 'S2T': GUIDE_T}.get(VER, '') + TAIL
    out.append({'type': 'input_text', 'text': '\n' + GP.EST_NOTE + '\n\n' + GP.QUESTION_HEADER + q + '\n\n' + instr})
    return out


FMT = {'type': 'json_schema', 'name': 'gate2', 'strict': True,
       'schema': {'type': 'object', 'properties': {'reason': {'type': 'string'}, 'choice': {'type': 'string', 'enum': OPTIONS}},
                  'required': ['reason', 'choice'], 'additionalProperties': False}}


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
    rec = dict(uid=t['uid'], category=t['category'], ver=VER, **x, usage=r.usage.model_dump(mode='json') if r.usage else {})
    OUT.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec)); os.replace(tmp, out)


with ThreadPoolExecutor(W) as ex:
    list(ex.map(job, TASKS))
recs = [json.loads(f.read_text()) for f in OUT.glob('*.json')]
usd = sum((r['usage'].get('input_tokens', 0) * 2 + r['usage'].get('output_tokens', 0) * 10) / 1e6 for r in recs)
print(json.dumps({'ver': VER, 'n': len(recs), 'usd': round(usd, 2)}))
