"""GPT-6.1-sol backend for the MindJourney and AVIC pipelines (2026-10-10). Same interface as gemini_backend.GeminiChatAPI:
get_system_response_with_content(sys_prompt, content, enum=None, role=...), where `content` is the OpenAI-style list
({type: text|image_url}) built by the pipelines' format_gpt_content().

Settings follow our GPT-6.1-sol standard (Responses API, store=False, verbosity low):
  - final answer calls (enum given): reasoning effort high, JSON schema whose `answer` is restricted to the exact choice
    strings; the returned text is that choice string (what the upstream answer parsers expect);
  - intermediate calls (scores, policy plans, verifier): reasoning effort $GPT_EFFORT_INTERMEDIATE (default high);
  - images: detail "low" for the roles listed in $GPT_LOWDETAIL_ROLES (comma separated), otherwise the API default.
Cost-reduction variants only change these two knobs (see run_test.sh).
Keys: primary = env var named by $GPT_KEY_ENV (default OPENAI_COMPANY_API_KEY); fallback = env var named by
$GPT_FALLBACK_KEY_ENV (optional). Every call is logged (no prompt text, no credentials) to $GPT_USAGE_LOG with the key
role ("primary" / "fallback"; older records without it count as primary).
Budget guard (before each call; sums all *.jsonl in $GPT_BUDGET_DIR, cached for 30 s):
  - switch: once the primary key's spend reaches $GPT_BUDGET_DIR/SWITCH_USD (and a fallback key is set), or the primary
    key reports no credit, all later calls of this process use the fallback key;
  - stop: once the total spend reaches $GPT_BUDGET_DIR/BUDGET_USD (else $GPT_BUDGET_USD) the process writes BUDGET_STOP
    and exits with code 2 (run scripts do not restart on 2); also when no usable key is left.
Price: $2 / $10 per 1M input / output tokens.
"""
import glob
import json
import os
import random
import threading
import time

PRICE_IN, PRICE_OUT = 2.0, 10.0
_lock = threading.Lock()
_budget = {'t': 0.0, 'usd': 0.0, 'primary': 0.0}
_state = {'key': 'primary'}


def _log(rec):
    path = os.environ.get('GPT_USAGE_LOG')
    if path:
        with _lock, open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')


def spent_usd():
    """(total, primary-key) spend in USD over all usage logs in $GPT_BUDGET_DIR. The primary-key spend counts only records
    logged with key_env == $GPT_KEY_ENV, so earlier runs with another key do not count toward the switch threshold."""
    d = os.environ.get('GPT_BUDGET_DIR')
    if not d:
        return 0.0, 0.0
    prim_env = os.environ.get('GPT_KEY_ENV', 'OPENAI_COMPANY_API_KEY')
    tot = prim = 0.0
    for f in glob.glob(os.path.join(d, '*.jsonl')):
        for line in open(f, encoding='utf-8'):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            tot += r.get('usd', 0.0)
            if r.get('key_env') == prim_env:
                prim += r.get('usd', 0.0)
    return tot, prim


def _read(d, name):
    try:
        return float(open(os.path.join(d, name)).read().strip())
    except (OSError, ValueError):
        return None


def _key_env():
    """Name of the env var holding the key in use (logged so spend can be split by key; never the key itself)."""
    return os.environ.get('GPT_KEY_ENV', 'OPENAI_COMPANY_API_KEY') if _state['key'] == 'primary' else os.environ.get('GPT_FALLBACK_KEY_ENV', '')


def _switch(reason):
    if _state['key'] == 'primary' and os.environ.get(os.environ.get('GPT_FALLBACK_KEY_ENV', ''), '').strip():
        _state['key'] = 'fallback'
        print(f'[gpt_backend] switching to the fallback key ({reason})', flush=True)
        return True
    return False


def _check_budget():
    d = os.environ.get('GPT_BUDGET_DIR')
    if not d:
        return
    if os.path.exists(os.path.join(d, 'BUDGET_STOP')):
        os._exit(2)
    if time.time() - _budget['t'] > 30:
        (_budget['usd'], _budget['primary']), _budget['t'] = spent_usd(), time.time()
    sw = _read(d, 'SWITCH_USD')
    if sw is not None and _budget['primary'] >= sw:
        _switch(f'primary-key spend ${_budget["primary"]:.2f} >= ${sw}')
    cap = _read(d, 'BUDGET_USD') or float(os.environ.get('GPT_BUDGET_USD', '0') or 0)
    if cap and _budget['usd'] >= cap:
        open(os.path.join(d, 'BUDGET_STOP'), 'w').write(f"{_budget['usd']:.2f}\n")
        print(f'[gpt_backend] budget ${cap} reached (${_budget["usd"]:.2f}); stopping', flush=True)
        os._exit(2)


class GPTChatAPI:
    def __init__(self, model='gpt-6.1-sol', api_price=0.0):
        from openai import OpenAI
        self._OpenAI = OpenAI; self._client_key = None
        if not os.environ.get(os.environ.get('GPT_KEY_ENV', 'OPENAI_COMPANY_API_KEY'), '').strip():
            raise RuntimeError('OpenAI key env var is not set')
        self.model = model
        self.price = api_price
        self.messages, self.history = [], []
        self.total_usage_tokens = 0
        self.cost = 0.0
        self.effort_mid = os.environ.get('GPT_EFFORT_INTERMEDIATE', 'high')
        self.lowdetail = {r for r in os.environ.get('GPT_LOWDETAIL_ROLES', '').split(',') if r}

    @property
    def client(self):
        if self._client_key != _state['key']:
            env = os.environ.get('GPT_KEY_ENV', 'OPENAI_COMPANY_API_KEY') if _state['key'] == 'primary' else os.environ['GPT_FALLBACK_KEY_ENV']
            self._client = self._OpenAI(api_key=os.environ[env].strip(), timeout=int(os.environ.get('GPT_TIMEOUT_S', '600')))
            self._client_key = _state['key']
        return self._client

    def _input(self, content, role):
        out = []
        for item in content:
            if item.get('type') == 'text':
                out.append({'type': 'input_text', 'text': item.get('text', '')})
            elif item.get('type') == 'image_url':
                x = {'type': 'input_image', 'image_url': item['image_url']['url']}
                if role in self.lowdetail:
                    x['detail'] = 'low'
                out.append(x)
        return out

    def get_system_response_with_content(self, sys_prompt, content, enum=None, role='call'):
        _check_budget()
        effort = 'high' if enum else self.effort_mid
        kw = dict(model=self.model, instructions=sys_prompt, reasoning={'effort': effort}, store=False,
                  max_output_tokens=16384, input=[{'role': 'user', 'content': self._input(content, role)}])
        if enum:
            kw['text'] = {'verbosity': 'low', 'format': {
                'type': 'json_schema', 'name': 'answer_choice', 'strict': True,
                'schema': {'type': 'object', 'properties': {'answer': {'type': 'string', 'enum': list(enum)}},
                           'required': ['answer'], 'additionalProperties': False}}}
        else:
            kw['text'] = {'verbosity': 'low'}
        n_images = sum(1 for x in content if x.get('type') == 'image_url')
        t0 = time.monotonic()
        for attempt in range(1, 9):
            try:
                r = self.client.responses.create(**kw)
                break
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, 'status_code', None); s = str(exc)
                out_of_credit = code == 429 and ('insufficient_quota' in s or 'credit_balance_exhausted' in s)
                retryable = not out_of_credit and (code in (408, 409, 429, 500, 502, 503, 504) or code is None)
                _log(dict(role=role, ok=False, attempt=attempt, error=type(exc).__name__, code=code, key=_state['key'], key_env=_key_env(),
                          pid=os.getpid(), t=time.time(), usd=0.0))
                if out_of_credit:
                    if _switch('primary key has no credit'):
                        continue
                    d = os.environ.get('GPT_BUDGET_DIR')
                    if d:   # marker for the monitor: the key in use has no credit left
                        open(os.path.join(d, 'OUT_OF_CREDIT'), 'a').write(f'{_key_env()} {time.time():.0f}\n')
                    os._exit(2)
                if not retryable or attempt == 8:
                    raise
                time.sleep(min(120.0, 2 ** attempt + random.uniform(0, 1.5)))
        u = r.usage
        inp = getattr(u, 'input_tokens', 0) or 0; out = getattr(u, 'output_tokens', 0) or 0
        reas = getattr(getattr(u, 'output_tokens_details', None), 'reasoning_tokens', 0) or 0
        usd = (inp * PRICE_IN + out * PRICE_OUT) / 1e6
        text = r.output_text or ''
        if enum and text:
            try:
                text = json.loads(text)['answer']
            except (ValueError, KeyError):
                pass
        _log(dict(role=role, ok=True, attempt=attempt, latency_s=round(time.monotonic() - t0, 2), n_images=n_images,
                  enum=bool(enum), effort=effort, low_detail=role in self.lowdetail, status=getattr(r, 'status', None),
                  prompt_tokens=inp, output_tokens=out, thoughts_tokens=reas, usd=round(usd, 6), key=_state['key'], key_env=_key_env(), pid=os.getpid(), t=time.time()))
        with _lock:
            _budget['usd'] += usd
            if _state['key'] == 'primary':
                _budget['primary'] += usd
        self.total_usage_tokens += inp + out
        self.cost += usd
        return text

    def clear(self):
        self.messages, self.history = [], []
