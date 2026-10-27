"""Question-only camera navigator in a generated HY scene.

Model inputs are restricted BY CONSTRUCTION to: the initial image, the movement text
(question clause removed in code), and renders/top-down views of the generated scene.
Every model call is logged with the sha256 of each image part and the text sent, and
audit() refuses any image that is not the initial image or a render of this session.
No Oracle data, labels, answers or native scans are read in this module.
"""
import base64, hashlib, json, math, os, re, sys, time, urllib.request
from pathlib import Path

import paramiko

REPO = Path(__file__).resolve().parents[2]
TASK = '<external>/hyworld2_nlp26_full_20261001'
MODEL = os.environ.get('NAV_MODEL', 'gemini-3.7-flash')

# Global constants: language convention + development-set statistics (aggregate only).
C = {'eye_m': float(os.environ.get('NAV_EYE_M', '1.75')), 'slight_deg': 35.0, 'turn_deg': 80.0, 'around_deg': 180.0,
     'forward_m': 2.5, 'forward_alone_m': 4.5, 'few_steps_m': 2.0, 'back_m': 1.0, 'const_set': os.environ.get('NAV_CONSTS', 'v1'),
     'stop_m': {'to': 0.8, 'near': 1.0, 'next_to': 0.8, 'beside': 0.8, 'into': 0.0, 'through': 0.0, 'past': 0.0},
     'past_beyond_m': 0.5, 'past_side_m': 0.9, 'through_beyond_m': 2.0,
     'face_low_threshold_deg': -20.0, 'face_low_pitch_deg': -25.0,
     'arrive_facing_target': False, 'leg_heading': os.environ.get('NAV_LEG_HEADING', '1') == '1', 'use_overlap': os.environ.get('NAV_OVERLAP', '0') == '1', 'overlap_min': 0.2, 'use_candidates': os.environ.get('NAV_CANDIDATES', '0') == '1', 'max_candidates': 6, 'refine_shift_m': 0.7, 'view_w': 384, 'view_h': 288}

if C['const_set'] == 'v1':
    C.update(forward_m=2.0, forward_alone_m=2.0, few_steps_m=1.0, through_beyond_m=1.0)

QUESTION_CLAUSE = re.compile(r'[,.]\s*(which|where|would|what|how|is|are|can|will|do|does)\b(?!.*[,.]\s*(which|where|would|what|how)\b)', re.I)


def motion_only(text):
    m = QUESTION_CLAUSE.search(text)
    if not m or not text.lower().startswith('if '):
        raise ValueError('Cannot isolate movement clause: ' + text)
    return text[3:m.start()].strip().rstrip(',.') + '.'


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def sha(b):
    return hashlib.sha256(b).hexdigest()


def gemini_key():
    k = os.environ.get('GEMINI_API_KEY')
    if k:
        return k
    for line in (REPO / '.env').read_text(encoding='utf-8-sig').splitlines():
        if line.strip().startswith('GEMINI_API_KEY='):
            return line.split('=', 1)[1].strip().strip('"\'')
    raise RuntimeError('GEMINI_API_KEY unavailable')


class Remote:
    """Persistent nav_env.py process on a GPU host (via a jump host)."""

    def __init__(self, scene_source, eye_m=None):
        self.eye_m = eye_m or C['eye_m']
        for attempt in range(6):
            try:
                return self._open(scene_source)
            except Exception:
                self.close(); time.sleep(10 + 10 * attempt)
        self._open(scene_source)

    def _open(self, scene_source):
        pw = os.environ['UNC_PW']
        self.j = paramiko.SSHClient(); self.j.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.j.connect(os.environ['JUMP_HOST'], username=os.environ['SSH_USER'], password=pw, timeout=30, look_for_keys=False, allow_agent=False)
        ch = self.j.get_transport().open_channel('direct-tcpip', (os.environ['GPU_HOST'], 22), ('127.0.0.1', 0), timeout=30)
        self.c = paramiko.SSHClient(); self.c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.c.connect(os.environ['GPU_HOST'], username=os.environ['SSH_USER'], password=pw, sock=ch, timeout=30, banner_timeout=60, look_for_keys=False, allow_agent=False)
        cmd = (f'cd {TASK}/code/hyworld2/worldgen && OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= NAV_EYE_M={self.eye_m} '
               f'{TASK}/env/bin/python -u {TASK}/scripts/hy_nav_pilot_20261002/nav_env.py {scene_source} 2>/dev/null')
        self.i, self.o, self.e = self.c.exec_command(cmd)
        self.info = json.loads(self.o.readline())
        self.n = 0

    def __call__(self, **kw):
        self.n += 1; kw['id'] = self.n
        self.i.write(json.dumps(kw) + '\n'); self.i.flush()
        r = json.loads(self.o.readline())
        if r.get('error'):
            raise RuntimeError(f"env {kw['cmd']}: {r['error']}")
        return r

    def close(self):
        try:
            self.i.close(); self.c.close(); self.j.close()
        except Exception:
            pass


class Navigator:
    def __init__(self, qid, initial_image, movement, scene_source, out, prefix=None, eye_m=None):
        self.prefix = prefix
        self.qid = qid; self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        self.initial = Path(initial_image).read_bytes()
        self.motion = motion_only(movement)
        self.allowed = {sha(self.initial)}
        self.calls = []
        self.env = Remote(scene_source, eye_m)
        self.x = 0.0; self.y = 0.0; self.yaw = 0.0; self.pitch = 0.0
        self.route = [[0.0, 0.0]]; self.frames = []; self.events = []
        self.keyframes = [{'kind': 'start', 'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'pitch': 0.0}]
        self.travel_yaw = 0.0
        self.key = gemini_key()

    # ---------- model ----------
    def ask(self, purpose, items, schema_hint):
        parts = []; audit = []
        for kind, value in items:
            if kind == 'text':
                parts.append({'text': value}); audit.append({'text': value})
            else:
                h = sha(value)
                if h not in self.allowed:
                    raise RuntimeError('Isolation violation: unregistered image')
                parts.append({'inlineData': {'mimeType': 'image/png' if value[:4] == b'\x89PNG' else 'image/jpeg', 'data': base64.b64encode(value).decode()}})
                audit.append({'image_sha256': h, 'kind': kind})
        body = {'contents': [{'role': 'user', 'parts': parts}],
                'generationConfig': {'temperature': 0, 'responseMimeType': 'application/json'}}
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent'
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'x-goog-api-key': self.key})
                with urllib.request.urlopen(req, timeout=180) as r:
                    resp = json.loads(r.read())
                text = ''.join(p.get('text', '') for p in resp['candidates'][0]['content']['parts'] if not p.get('thought'))
                result = json.loads(text)
                self.calls.append({'purpose': purpose, 'model': MODEL, 'inputs': audit, 'response': result})
                return result
            except Exception as exc:
                err = exc; time.sleep(5 * (attempt + 1))
        raise RuntimeError(f'model call failed: {err}')

    def view(self, x, y, yaw, pitch=0.0, w=None, h=None, save=None):
        r = self.env(cmd='render', x=x, y=y, yaw=yaw, pitch=pitch, w=w or C['view_w'], h=h or C['view_h'])
        b = base64.b64decode(r['png']); self.allowed.add(sha(b))
        if save:
            (self.out / save).write_bytes(b)
        return b

    # ---------- steps ----------
    def parse(self):
        prompt = ('You will control a camera that starts at the viewpoint of this photo, at standing eye height. '
                  'Convert the movement instruction into primitive steps, in order. Use ONLY what the instruction says.\n'
                  f'Instruction: "{self.motion}"\n'
                  'Primitives (JSON objects):\n'
                  '- {"op":"forward","amount":"default|few_steps"}  walk straight ahead without a named target\n'
                  '- {"op":"back","direction":"left|right|null"}  step backwards (optionally back-left/right)\n'
                  '- {"op":"turn","direction":"left|right","amount":"slight|normal|around"}  turn in place; "turn around" is amount around\n'
                  '- {"op":"go","target":"...","relation":"to|near|next_to|beside|into|through|past","side":"left|right|null"}  walk to/near/past/through a target\n'
                  '- {"op":"face","target":"..."}  turn in place to face an object (also "look at"/"look through")\n'
                  '- {"op":"face_start"}  turn in place to face the starting point\n'
                  '- {"op":"face_motion"}  keep facing the walking direction\n'
                  'Target descriptions must uniquely identify the object as seen in the photo when possible: include color, material, '
                  'relative position in the photo (left/right/center, near/far) exactly as implied by the instruction. '
                  'If the target is not visible in the photo, still describe it from the instruction. '
                  'Return {"steps":[...]}')
        r = self.ask('parse', [('initial', self.initial), ('text', prompt)], 'steps')
        steps = r if isinstance(r, list) else r['steps']
        # Language-only consistency guard: never turn toward the start unless the instruction says so.
        if not re.search(r'starting point|where i (?:come|came) from|the way i (?:come|came) from|look back', self.motion, re.I):
            for st in steps:
                if st.get('op') == 'face_start':
                    st.clear(); st.update({'op': 'face_motion', 'guard': 'face_start not mentioned in instruction'})
        return steps

    def ground(self, target):
        """Return a metre-space surface point of target, or None."""
        prompt = (f'Movement instruction: "{self.motion}"\nTarget: "{target}".\n'
                  'Is this target visible in the photo? If yes give its tight bounding box [ymin, xmin, ymax, xmax] in 0-1000 '
                  'normalized coordinates (only the target object itself). Return {"visible":true|false,"box":[ymin,xmin,ymax,xmax],"confidence":0-1}')
        if True:  # world-fixed target: the initial photo is the sharpest evidence from any position
            r = self.ask('ground_initial', [('initial', self.initial), ('text', prompt)], 'point')
            if r.get('visible') and r.get('box'):
                box = [v / 1000 for v in r['box']]
                p = self.env(cmd='locate_box', x=0, y=0, yaw=0, pitch=0, box=box)['point']
                if p is not None:
                    self.last_ground = {'pose': [0.0, 0.0, 0.0, 0.0], 'box': box}
                    self.events.append({'ground': target, 'method': 'initial_photo', 'box': r['box'], 'point_m': p})
                    return p
        # look around from the current position in the generated scene
        imgs = []; yaws = []
        for k in range(6):
            yw = wrap(self.yaw + math.radians(60 * k)); yaws.append(yw)
            imgs.append(self.view(self.x, self.y, yw, 0.0, save=f'look_{len(self.events):02d}_{k}.png'))
        items = [('initial', self.initial), ('text', 'Above: the original starting photo. Below: six views from the camera\'s CURRENT position '
                                              'in a blurry 3D reconstruction of the same place. View k is rotated 60*k degrees to the right of the current facing direction.')]
        for k, b in enumerate(imgs):
            items += [('text', f'View {k}:'), ('render', b)]
        items.append(('text', f'Movement instruction: "{self.motion}"\nTarget: "{target}".\nWhich view shows this target most clearly? '
                              'Give its tight bounding box [ymin, xmin, ymax, xmax] in 0-1000 normalized coordinates of that view. '
                              'Return {"view":k or -1,"box":[ymin,xmin,ymax,xmax],"confidence":0-1}'))
        r = self.ask('ground_lookaround', items, 'view')
        k = r.get('view', -1)
        if k is None or k < 0 or not r.get('box'):
            return None
        p = self.env(cmd='locate_box', x=self.x, y=self.y, yaw=yaws[k], pitch=0, box=[v / 1000 for v in r['box']])['point']
        self.last_ground = {'pose': [self.x, self.y, yaws[k], 0.0], 'box': [v / 1000 for v in r['box']]}
        self.events.append({'ground': target, 'method': 'lookaround', 'view': k, 'box': r['box'], 'point_m': p})
        return p

    def refine(self, target, p):
        """Close-range re-identification of the target in a render from the arrival pose."""
        dx, dy = p[0] - self.x, p[1] - self.y
        yaw = math.atan2(dy, -dx); pitch = math.radians(-15)
        b = self.view(self.x, self.y, yaw, pitch, save=f'refine_{len(self.events):02d}.png')
        r = self.ask('refine', [('initial', self.initial), ('text', 'Above: the original starting photo. Below: the current view in a blurry 3D '
                                 'reconstruction of the same place, after walking toward the target.'), ('render', b),
                                ('text', f'Movement instruction: "{self.motion}"\nTarget: "{target}".\nIs this target visible in the current view? '
                                         'If yes give its tight bounding box [ymin, xmin, ymax, xmax] in 0-1000 normalized coordinates. '
                                         'Return {"visible":true|false,"box":[ymin,xmin,ymax,xmax],"confidence":0-1}')], 'box')
        bx = r.get('box') or [0, 0, 0, 0]
        if r.get('visible') and (bx[2] - bx[0]) * (bx[3] - bx[1]) > 0.6 * 1000 * 1000:
            self.events.append({'refine': target, 'ignored': 'box covers most of the view (too close)'}); return None
        if not r.get('visible') or not r.get('box') or (r.get('confidence') or 0) < .5:
            self.events.append({'refine': target, 'visible': False}); return None
        q = self.env(cmd='locate_box', x=self.x, y=self.y, yaw=yaw, pitch=pitch, box=[v / 1000 for v in r['box']])['point']
        self.events.append({'refine': target, 'box': r['box'], 'point_m': q, 'previous_m': p})
        return q

    def kf(self, kind):
        k = self.keyframes[-1]
        if kind == 'turn' and abs(wrap(k['yaw'] - self.yaw)) < 1e-4 and abs(k['pitch'] - self.pitch) < 1e-4:
            return
        self.keyframes.append({'kind': kind, 'x': self.x, 'y': self.y, 'yaw': self.yaw, 'pitch': self.pitch})

    def ground_multi(self, target, k=3):
        """Up to k plausible instances of the target, located from the initial photo (most likely first)."""
        prompt = (f'Movement instruction: "{self.motion}"\nTarget: "{target}".\nList up to {k} objects in the photo that could be this target, '
                  'most likely first (the first must be your best answer). For each give a tight bounding box [ymin, xmin, ymax, xmax] in 0-1000 '
                  'normalized coordinates. Return {"candidates":[{"box":[ymin,xmin,ymax,xmax],"confidence":0-1}]} (empty list if none visible).')
        r = self.ask('ground_multi', [('initial', self.initial), ('text', prompt)], 'candidates')
        lst = r if isinstance(r, list) else (r.get('candidates') or [])
        pts = []
        for c in lst[:k]:
            if not isinstance(c, dict):
                continue
            if not c.get('box') or len(c['box']) != 4:
                continue
            p = self.env(cmd='locate_box', x=0, y=0, yaw=0, pitch=0, box=[v / 1000 for v in c['box']])['point']
            if p is not None and all(math.dist(p[:2], q[:2]) > 0.8 for q in pts):
                pts.append(p)
        self.events.append({'ground_multi': target, 'points_m': pts})
        return pts

    def choose_candidate(self, s, instances):
        """Render each candidate stopping pose and let the model pick the one that matches the instruction."""
        rel = s.get('relation') or 'to'
        cands = []
        for ii, p in enumerate(instances):
            r = self.env(cmd='candidates', x=self.x, y=self.y, tx=p[0], ty=p[1], stop_m=C['stop_m'].get(rel, 1.0), keep=3 if ii == 0 else 2)
            for c in r['candidates']:
                c['instance'] = ii; c['target'] = p; cands.append(c)
        cands = cands[:C['max_candidates']]
        if len(cands) <= 1:
            return cands[0] if cands else None
        items = [('initial', self.initial), ('text', 'Above: the original starting photo. Below: candidate stopping positions in a blurry 3D '
                                                    'reconstruction of the same place; each view looks from the candidate position toward the candidate target.')]
        for k, c in enumerate(cands, 1):
            end = c['waypoints'][-1]; yaw = math.atan2(c['target'][1] - end[1], -(c['target'][0] - end[0]))
            items += [('text', f'Candidate {k}:'), ('render', self.view(end[0], end[1], yaw, 0.0, save=f'cand_{len(self.events):02d}_{k}.png'))]
        top = self.env(cmd='topdown', pose=[self.x, self.y, self.yaw],
                       marks=[{'p': c['waypoints'][-1], 'label': str(k), 'color': (220, 0, 0)} for k, c in enumerate(cands, 1)] +
                             [{'p': c['target'], 'color': (0, 160, 0)} for c in cands] + [{'p': [0.0, 0.0], 'label': 'S', 'color': (0, 0, 255)}])
        tb = base64.b64decode(top['png']); self.allowed.add(sha(tb))
        items += [('text', 'Top-down map of the reconstruction (forward from the start is up; S = start, blue arrow = current pose, red numbers = '
                           'candidate stopping positions, green dots = candidate targets):'), ('render', tb),
                  ('text', f'Full movement instruction: "{self.motion}"\nCurrent step: {json.dumps(s)}\n'
                           'Which candidate best completes the current step? Judge (1) whether the viewed object is the intended target '
                           '(compare with the original photo and the wording), (2) whether the position matches the relation (to/near/next to/beside) '
                           'and is a natural place for a person following the instruction to stop, (3) consistency with the rest of the instruction. '
                           'Return {"choice":k,"reason":"..."}')]
        r = self.ask('choose_candidate', items, 'choice')
        k = r.get('choice')
        pick = cands[k - 1] if isinstance(k, int) and 1 <= k <= len(cands) else cands[0]
        self.events.append({'choose': s.get('target'), 'n_candidates': len(cands), 'choice': k, 'reason': r.get('reason', '')[:300],
                            'candidates': [{'end': c['waypoints'][-1], 'instance': c['instance'], 'sector': c['sector'], 'path_m': c['path_m']} for c in cands]})
        return pick

    def walk_to(self, pts, facing=None):
        for p in pts[1:]:
            dx, dy = p[0] - self.x, p[1] - self.y
            if math.hypot(dx, dy) < .05:
                continue
            heading = math.atan2(dy, -dx)
            if facing is None and abs(wrap(heading - self.yaw)) < math.radians(1.0):
                heading = self.yaw  # absorb sub-degree heading noise from grid paths
            self.yaw = heading if facing is None else facing; self.travel_yaw = heading; self.pitch = 0.0
            self.kf('turn')
            self.x, self.y = float(p[0]), float(p[1]); self.route.append([self.x, self.y]); self.kf('walk')

    def straight(self, yaw, dist):
        r = self.env(cmd='move', x=self.x, y=self.y, yaw=yaw, dist=dist)
        self.events.append({'move_yaw_deg': math.degrees(yaw), 'requested_m': dist, 'end': r['end'], 'blocked': r['blocked']})
        self.walk_to([[self.x, self.y], r['end']])
        return r

    def do(self, s):
        op = s.get('op')
        if op in ('forward', 'back', 'go'):
            leg_start = (self.x, self.y)
            self._do(s)
            # Reference heading after a movement = direction of the whole leg, not the last (noisy) path segment.
            dx, dy = self.x - leg_start[0], self.y - leg_start[1]
            if C['leg_heading'] and math.hypot(dx, dy) > 0.3:
                self.travel_yaw = math.atan2(dy, -dx)
                if op != 'back':
                    self.yaw = self.travel_yaw; self.pitch = 0.0
                    self.kf('turn')
            return
        self._do(s)

    def _do(self, s):
        op = s.get('op')
        if op == 'forward':
            # global dev statistics: forward before another movement ~2.5 m; forward as the only movement ~4.5-5 m
            dist = C['few_steps_m'] if s.get('amount') == 'few_steps' else (C['forward_m'] if getattr(self, 'more_moves', True) else C['forward_alone_m'])
            self.straight(self.yaw, dist)
        elif op == 'back':
            off = {'left': -math.radians(30), 'right': math.radians(30)}.get(s.get('direction'), 0)
            keep = self.yaw
            r = self.env(cmd='move', x=self.x, y=self.y, yaw=wrap(self.yaw + math.pi - off), dist=C['back_m'])
            self.events.append({'back': True, 'end': r['end'], 'blocked': r['blocked']})
            self.walk_to([[self.x, self.y], r['end']], facing=keep)
        elif op == 'turn':
            deg = {'slight': C['slight_deg'], 'normal': C['turn_deg'], 'around': C['around_deg']}[s.get('amount', 'normal')]
            sign = -1 if s.get('direction') == 'left' else 1
            if s.get('amount') == 'around':
                sign = 1
            self.yaw = wrap(self.yaw + sign * math.radians(deg)); self.pitch = 0.0
        elif op == 'face_start':
            if math.hypot(self.x, self.y) > .05:
                self.yaw = math.atan2(-self.y, self.x); self.pitch = 0.0
        elif op == 'face_motion':
            self.yaw = self.travel_yaw; self.pitch = 0.0
        elif op == 'face':
            p = self.ground(s['target'])
            if p is None:
                raise RuntimeError('face target not found: ' + s['target'])
            self.face_point(p)
        elif op == 'go':
            p = self.ground(s['target'])
            if p is None:
                self.straight(self.yaw, C['forward_m'])     # explore once, then retry
                p = self.ground(s['target'])
            if p is None:
                raise RuntimeError('go target not found: ' + s['target'])
            rel = s.get('relation') or 'to'
            if rel == 'past':
                # Continue roughly along the current heading until beyond the target,
                # shifting sideways only as needed to keep clearance from it.
                h = [-math.cos(self.yaw), math.sin(self.yaw)]; rgt = [h[1], -h[0]]
                rel_v = [p[0] - self.x, p[1] - self.y]
                along = rel_v[0] * h[0] + rel_v[1] * h[1]; lat = rel_v[0] * rgt[0] + rel_v[1] * rgt[1]
                if along < 0.3:  # target not ahead: head toward it first
                    yaw_to = math.atan2(rel_v[1], -rel_v[0]); h = [-math.cos(yaw_to), math.sin(yaw_to)]; rgt = [h[1], -h[0]]
                    along = math.hypot(*rel_v); lat = 0.0
                want = s.get('side')
                keep_on = ('right' if lat > 0 else 'left') if want not in ('left', 'right') else ('left' if want == 'right' else 'right')
                need = C['past_side_m']
                off = 0.0
                if abs(lat) < need or (keep_on == 'right') != (lat > 0):
                    off = lat - need if keep_on == 'right' else lat + need
                wp = [self.x + h[0] * (along + C['past_beyond_m']) + rgt[0] * off, self.y + h[1] * (along + C['past_beyond_m']) + rgt[1] * off]
                g = self.env(cmd='goto', x=self.x, y=self.y, tx=wp[0], ty=wp[1], stop_m=0.0)
                if not g.get('waypoints'):
                    raise RuntimeError('no path past target')
                self.walk_to(g['waypoints'])
            elif rel in ('into', 'through'):
                g = self.env(cmd='goto', x=self.x, y=self.y, tx=p[0], ty=p[1], stop_m=0.0)
                if not g.get('waypoints'):
                    raise RuntimeError('no path through target')
                self.walk_to(g['waypoints']); self.straight(self.yaw, C['through_beyond_m'])
            elif C['use_overlap'] and getattr(self, 'last_ground', None):
                lg = self.last_ground; px, py, pyaw, ppitch = lg['pose']
                pts = self.env(cmd='target_points', x=px, y=py, yaw=pyaw, pitch=ppitch, box=lg['box'])['points']
                best = None
                if pts:
                    cands = self.env(cmd='scored_candidates', x=self.x, y=self.y, tx=p[0], ty=p[1], stop_m=C['stop_m'].get(rel, 1.0), pts=pts, keep=8)['candidates']
                    if cands:
                        top = max(c.get('visible_frac', 0) for c in cands)
                        if top >= C['overlap_min']:
                            best = min((c for c in cands if c.get('visible_frac', 0) >= top - 0.1), key=lambda c: c['path_m'])
                    self.events.append({'overlap': s.get('target'), 'n_points': len(pts), 'candidates': [
                        {k: c.get(k) for k in ('sector', 'path_m', 'stop_m', 'visible_frac', 'center_deg', 'width_deg', 'distance_m')} for c in cands] if pts else [],
                        'chosen_sector': best['sector'] if best else None})
                if best is not None:
                    self.walk_to(best['waypoints'])
                else:
                    g = self.env(cmd='goto', x=self.x, y=self.y, tx=p[0], ty=p[1], stop_m=C['stop_m'].get(rel, 1.0))
                    if not g.get('waypoints'):
                        raise RuntimeError('no path to target')
                    self.walk_to(g['waypoints'])
            elif C['use_candidates']:
                inst = [p] + [q for q in self.ground_multi(s['target']) if math.dist(q[:2], p[:2]) > 0.8]
                pick = self.choose_candidate(s, inst[:3])
                if pick is not None:
                    self.walk_to(pick['waypoints']); p = pick['target']
                else:   # no sector candidate (e.g. target beyond the reachable floor): fall back to plain goto
                    g = self.env(cmd='goto', x=self.x, y=self.y, tx=p[0], ty=p[1], stop_m=C['stop_m'].get(rel, 1.0))
                    if not g.get('waypoints'):
                        raise RuntimeError('no path to target')
                    self.walk_to(g['waypoints'])
            else:
                g = self.env(cmd='goto', x=self.x, y=self.y, tx=p[0], ty=p[1], stop_m=C['stop_m'].get(rel, 1.0))
                if not g.get('waypoints'):
                    raise RuntimeError('no path to target')
                before = (self.x, self.y, self.yaw, self.travel_yaw, len(self.route), len(self.keyframes))
                self.walk_to(g['waypoints'])
                q = self.refine(s['target'], p)
                if q is not None and math.dist(q[:2], p[:2]) > C['refine_shift_m']:
                    self.x, self.y, self.yaw, self.travel_yaw = before[:4]; del self.route[before[4]:]; del self.keyframes[before[5]:]
                    g2 = self.env(cmd='goto', x=self.x, y=self.y, tx=q[0], ty=q[1], stop_m=C['stop_m'].get(rel, 1.0))
                    if g2.get('waypoints'):
                        self.walk_to(g2['waypoints']); p = q
                    else:
                        self.walk_to(g['waypoints'])
                if C['arrive_facing_target']:
                    dx, dy = p[0] - self.x, p[1] - self.y
                    if math.hypot(dx, dy) > .05:
                        self.yaw = math.atan2(dy, -dx)
            self.last_target = p
        else:
            raise ValueError('unknown op ' + str(op))
        self.kf('turn')

    def face_point(self, p):
        dx, dy = p[0] - self.x, p[1] - self.y
        self.yaw = math.atan2(dy, -dx)
        pitch = math.degrees(math.atan2(p[2], math.hypot(dx, dy)))
        self.pitch = math.radians(C['face_low_pitch_deg'] if pitch < C['face_low_threshold_deg'] else 0.0)

    def run(self):
        t0 = time.time()
        result = {'question_id': self.qid, 'motion_text': self.motion, 'constants': C, 'model': MODEL,
                  'inputs_allowed': ['initial_image', 'motion_text', 'generated_scene_renders'], 'gt_metadata_used': []}
        try:
            self.view(0, 0, 0, save='start_render.png')
            steps = self.parse(); result['steps'] = steps
            moving = [i for i, s in enumerate(steps) if s.get('op') in ('forward', 'back', 'go')]
            start = 0
            if self.prefix and moving:
                # Same walk as the scene's leader question (identical motion text before the final
                # orientation clause): reuse its executed walk, then run only this question's tail.
                P = self.prefix
                self.keyframes = [dict(k) for k in P['keyframes']]; self.route = [list(r) for r in P['route']]
                self.x, self.y, self.yaw, self.pitch, self.travel_yaw = P['x'], P['y'], P['yaw'], P['pitch'], P['travel_yaw']
                start = moving[-1] + 1
                result['shared_walk_from'] = P['question_id']
            rest = steps[start:]
            for i, s in enumerate(rest):
                self.more_moves = any(t.get('op') in ('forward', 'back', 'go') for t in rest[i + 1:]) or any(t.get('op') in ('forward', 'back', 'go') for t in steps[:start])
                self.do(s)
                self.frames.append({'after': s, 'x': self.x, 'y': self.y, 'yaw': self.yaw, 'pitch': self.pitch})
            if moving:
                last = max(i for i, k in enumerate(self.keyframes) if k['kind'] == 'walk')
                result['walk_state'] = {'question_id': self.qid, 'keyframes': self.keyframes[:last + 1], 'route': self.route,
                                        'x': self.keyframes[last]['x'], 'y': self.keyframes[last]['y'],
                                        'yaw': self.travel_yaw if C['leg_heading'] else self.keyframes[last]['yaw'],
                                        'pitch': self.keyframes[last]['pitch'], 'travel_yaw': self.travel_yaw}
            self.view(self.x, self.y, self.yaw, self.pitch, w=640, h=480, save='final_render.png')
            top = self.env(cmd='topdown', path=self.route, pose=[self.x, self.y, self.yaw])
            (self.out / 'topdown_debug.png').write_bytes(base64.b64decode(top['png']))
            result.update(status='complete', route_m=self.route, final={'x': self.x, 'y': self.y, 'yaw_deg': math.degrees(self.yaw), 'pitch_deg': math.degrees(self.pitch)},
                          step_states=self.frames, keyframes=self.keyframes, events=self.events, m_per_unit=self.env.info['m_per_unit'])
        except Exception as exc:
            result.update(status='failed', error=repr(exc), route_m=self.route, events=self.events)
        finally:
            result['seconds'] = time.time() - t0
            (self.out / 'plan.json').write_text(json.dumps(result, indent=2))
            (self.out / 'model_calls.json').write_text(json.dumps(self.calls, indent=2))
            self.env.close()
        return result


if __name__ == '__main__':
    qid, image, movement, source, out = sys.argv[1:6]
    print(json.dumps(Navigator(qid, image, movement, source, out).run(), indent=1)[:3000])
