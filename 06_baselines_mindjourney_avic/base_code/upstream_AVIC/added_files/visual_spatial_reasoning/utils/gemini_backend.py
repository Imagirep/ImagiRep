"""Shared Gemini 3.7 Flash backend for the MindJourney and AVIC baselines (2026-10-06).

Exposes the ChatAPI interface both upstream pipelines call: get_system_response_with_content(sys_prompt, content),
where `content` is the OpenAI-style list produced by their format_gpt_content() ({type: text|image_url}).

Standard settings everywhere: temperature 0, HIGH thinking, 8192 max output tokens. Intermediate calls (policy JSON,
scores) are unconstrained text, as in the upstream pipelines. The final answer call passes `enum` = the exact answer
choice strings, so the reply is constrained to one choice (our A-D enum standard, expressed as the choice text that
the upstream answer parsers expect).

Per-request HTTP timeout 300 s; retries 429/5xx/timeouts with exponential backoff. Every call is appended to the
JSONL file named by $GEMINI_USAGE_LOG (role, latency, token counts, attempts; never prompt text or credentials).
"""
import base64
import json
import os
import random
import threading
import time

MODEL_DEFAULT = "gemini-3.7-flash"
TIMEOUT_S = int(os.environ.get("GEMINI_TIMEOUT_S", "300"))
MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "8"))
_log_lock = threading.Lock()


def _log(rec):
    path = os.environ.get("GEMINI_USAGE_LOG")
    if not path:
        return
    with _log_lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


class GeminiChatAPI:
    def __init__(self, model=MODEL_DEFAULT, api_price=0.0):
        from google import genai
        from google.genai import types
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.types = types
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000))
        self.model = model
        self.price = api_price
        self.messages, self.history = [], []
        self.total_usage_tokens = 0
        self.cost = 0.0

    def _parts(self, content):
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(self.types.Part.from_text(text=item.get("text", "")))
            elif item.get("type") == "image_url":
                url = item["image_url"]["url"]
                header, payload = url.split(",", 1)
                mime = header[5:].split(";", 1)[0] or "image/png"
                parts.append(self.types.Part.from_bytes(data=base64.b64decode(payload), mime_type=mime))
        return parts

    def get_system_response_with_content(self, sys_prompt, content, enum=None, role="call"):
        t = self.types
        kw = dict(system_instruction=sys_prompt, temperature=0, max_output_tokens=8192,
                  thinking_config=t.ThinkingConfig(thinking_level=t.ThinkingLevel.HIGH))
        if enum:
            kw.update(response_mime_type="text/x.enum", response_schema={"type": "STRING", "enum": list(enum)})
        config = t.GenerateContentConfig(**kw)
        parts = self._parts(content)
        n_images = sum(1 for x in content if x.get("type") == "image_url")
        t0 = time.monotonic()
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=[t.Content(role="user", parts=parts)], config=config)
                break
            except Exception as exc:  # noqa: BLE001
                s = f"{type(exc).__name__} {getattr(exc, 'code', '')} {str(exc)[:300]}"
                retryable = any(m in s for m in ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                                                 "INTERNAL", "Deadline", "Timeout", "timed out", "ReadTimeout"))
                _log(dict(role=role, ok=False, attempt=attempt, error=type(exc).__name__, retryable=retryable,
                          pid=os.getpid(), t=time.time()))
                if not retryable or attempt == MAX_RETRIES:
                    raise
                time.sleep(min(120.0, 2 ** attempt + random.uniform(0, 1.5)))
        u = getattr(resp, "usage_metadata", None)
        text = resp.text or ""
        rec = dict(role=role, ok=True, attempt=attempt, latency_s=round(time.monotonic() - t0, 2), n_images=n_images,
                   enum=bool(enum), prompt_tokens=getattr(u, "prompt_token_count", None),
                   output_tokens=getattr(u, "candidates_token_count", None),
                   thoughts_tokens=getattr(u, "thoughts_token_count", None), pid=os.getpid(), t=time.time())
        self.total_usage_tokens += int(getattr(u, "total_token_count", 0) or 0)
        _log(rec)
        return text

    def clear(self):
        self.messages, self.history = [], []

# [gpt61-patch 2026-10-10] gpt-* model names are served by utils.gpt_backend.GPTChatAPI; gemini-* behaviour is unchanged.
_GeminiChatAPI_impl = GeminiChatAPI


def GeminiChatAPI(model=MODEL_DEFAULT, api_price=0.0):  # noqa: F811
    if str(model).startswith("gpt-"):
        from utils.gpt_backend import GPTChatAPI
        return GPTChatAPI(model, api_price)
    return _GeminiChatAPI_impl(model, api_price)
