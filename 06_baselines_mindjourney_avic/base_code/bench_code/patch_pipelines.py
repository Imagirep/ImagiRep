"""Apply the 2026-10-06 benchmark adaptations to clean upstream checkouts (run on the server).

  python patch_pipelines.py mj   <MindJourney checkout @aeed003>
  python patch_pipelines.py avic <AVIC checkout @88d55e2 with our 2026-08 macro/Gemini overlay copied in>

Changes (algorithms, prompts and search/control logic are untouched):
  1. Gemini 3.7 Flash backend (code/gemini_backend.py) for every VLM call; final-answer calls are enum-constrained to
     the exact answer-choice strings.
  2. All observed images are passed to the VLM (Image 1 = primary/WM-conditioning view, Images 2..n = helpers). Upstream
     passed at most one helper image.
  3. Final answers are scored by exact match of the (enum-constrained) last line. Upstream used substring matching,
     which mis-scores choices that contain one another (e.g. "chair" / "armchair").
  4. MJ only: --inference_step is a real CLI arg (upstream script set it but never passed it, silently using 50), the
     SWM checkpoint must load (upstream silently fell back to base weights with random camera encoders), and the
     vlm model-name `choices` restriction is removed so gemini-* names are accepted.
Motion ranges are changed only through existing CLI arguments (see run_*.sh).
Idempotent: refuses to patch a file twice (marker comment).
"""
import re
import shutil
import sys
from pathlib import Path

MARK = "# [bench-patch 2026-10-06]"
NL = "\n"
HERE = Path(__file__).resolve().parent


def edit(path, subs):
    p = Path(path)
    s = p.read_text(encoding="utf-8").replace("\r\n", "\n")
    if MARK in s:
        print(f"skip (already patched): {p}")
        return
    for old, new, count in subs:
        if isinstance(old, re.Pattern):
            s, n = old.subn(lambda _m, _new=new: _new, s)  # literal replacement (no backslash processing)
        else:
            n = s.count(old)
            s = s.replace(old, new)
        assert n == count, f"{p}: expected {count} match(es) for {str(old)[:80]!r}, got {n}"
    p.write_text(s.rstrip("\n") + f"\n{MARK}\n", encoding="utf-8")
    print(f"patched: {p}")


HELPERS_LIST = "[primary_img_path] + self._helper_imgs"

PROCESS_ANSWER = '''    def _process_answer(self, response: str, question: dict, fwd=0.075, turn=3):
        # [bench-patch] exact match on the enum-constrained final line (upstream: substring match)
        last = (response or "").strip().split("\\n")[-1].strip().lower()
        if last not in [c.lower() for c in question["answer_choices"]]:
            return "out of control"
        return "correct" if last == question["correct_answer"].lower() else "wrong"
'''
PROCESS_ANSWER_RE = re.compile(
    r"    def _process_answer\(self, response: str, question: dict, fwd=0\.075, turn=3\):\n.*?return \"out of control\"\n",
    re.S)


def helper_block(indent, resize_else):
    i = " " * indent
    return (f'{i}helper_img_path = None\n'
            f'{i}self._helper_imgs = []  # [bench-patch] all observed images after Image 1\n'
            f'{i}for _hi, _hp in enumerate(question["img_paths"][1:], start=1):\n'
            f'{i}    _dst = os.path.join(save_dir, "step_0", "helper_img.png" if _hi == 1 else f"helper_img_{{_hi}}.png")\n'
            f'{i}    helper = cv2.imread(_hp)\n'
            + (f'{i}    helper = resize_to_short_side(helper, target_short=512)\n' if resize_else else '')
            + f'{i}    cv2.imwrite(_dst, helper)\n'
            f'{i}    self._helper_imgs.append(_dst)\n'
            f'{i}helper_img_path = self._helper_imgs[0] if self._helper_imgs else None\n')


HELPER_RE = re.compile(r"            helper_img_path = None\n            if len\(question\[\"img_paths\"\]\) > 1:\n.*?cv2\.imwrite\(helper_img_path, helper\)\n", re.S)

WRAPPER_GEMINI_RUN = '''            if getattr(self, "is_gemini", False):  # [bench-patch]
                enum = list(self.curr_choices) if prompt_type[:7] == "answer_" and self.curr_choices else None
                return self.model.get_system_response_with_content(system_prompt, content, enum=enum, role=prompt_type)
'''


def patch_mj(root):
    root = Path(root)
    shutil.copy(HERE / "gemini_backend.py", root / "utils/gemini_backend.py")
    edit(root / "utils/vlm_wrapper.py", [
        ("from utils.api import ChatAPI, AzureConfig\n",
         "from utils.api import ChatAPI, AzureConfig\nfrom utils.gemini_backend import GeminiChatAPI\n", 1),
        ("    def __init__(self, model_name, qa_model_name=None):\n",
         "    def __init__(self, model_name, qa_model_name=None):\n"
         "        self.is_gemini = str(model_name).startswith('gemini')  # [bench-patch]\n"
         "        self.curr_choices = None\n"
         "        if self.is_gemini:\n"
         "            self.model = GeminiChatAPI(model_name); self.qa_model = None\n"
         "            self.prompt_style = 'gpt'; self.curr_prompt = None\n"
         "            return\n", 1),
        ("    def format_prompt(self, prompt_type, question, answer_choices, images, action_consequences=None, sys_prompt=None):\n",
         "    def format_prompt(self, prompt_type, question, answer_choices, images, action_consequences=None, sys_prompt=None):\n"
         "        self.curr_choices = answer_choices  # [bench-patch]\n", 1),
        ("            content = format_gpt_content(content)\n",
         "            content = format_gpt_content(content)\n" + WRAPPER_GEMINI_RUN, 1),
    ])
    edit(root / "utils/args.py", [
        ('        choices=["gpt-4o", "gpt-4.1", "o4-mini", "o1", "OpenGVLab/InternVL3-8B", "OpenGVLab/InternVL3-14B"],\n', "", 1),
        ('        choices=[None, "None", "gpt-4o", "gpt-4.1", "o4-mini", "o1", "OpenGVLab/InternVL3-8B", "OpenGVLab/InternVL3-14B"],\n', "", 1),
        ('    parser.add_argument(\n        "--num_question_chunks",',
         '    parser.add_argument("--inference_step", type=int, default=50)  # [bench-patch] upstream default path used 50\n'
         '    parser.add_argument("--helpful_topk", type=int, default=0)  # [bench-patch] rank-based helpful selection\n'
         '    parser.add_argument("--helpful_floor", type=float, default=99)\n'
         '    parser.add_argument("--exploration_floor", type=float, default=99)  # 99 = upstream (threshold only)\n'
         '    parser.add_argument(\n        "--num_question_chunks",', 1),
    ])
    edit(root / "pipelines/pipeline_wan_scaling_beam_search_double_rank.py", [
        (HELPER_RE, helper_block(12, resize_else=False), 1),
        ("[primary_img_path, helper_img_path] if len(question['img_paths']) > 1 else [primary_img_path]", HELPERS_LIST, 2),
        (PROCESS_ANSWER_RE, PROCESS_ANSWER, 1),
        # Rank-based selection (2026-10-06): Gemini rarely scores >= 8, so also keep the per-step top-k views that clear
        # a quality floor. Upstream absolute thresholds still apply; defaults (topk 0, floors 99) = upstream behaviour.
        ("                        for score, index in top_scores_all[rank_i]:" + NL,
         "                        for _rank, (score, index) in enumerate(top_scores_all[rank_i]):  # [bench-patch]" + NL, 1),
        ("                            if score < self.model_args.helpful_score_threshold:" + NL,
         "                            if score < self.model_args.helpful_score_threshold and not (_rank < self.model_args.helpful_topk and score >= self.model_args.helpful_floor):  # [bench-patch]" + NL, 1),
        ("                    if score < self.model_args.exploration_score_threshold:" + NL,
         "                    if score < min(self.model_args.exploration_score_threshold, self.model_args.exploration_floor):  # [bench-patch]" + NL, 1),
        ("            theta = np.array([np.radians(-10.0), 0.0, 0.0], dtype=float)" + NL,
         "            theta = np.array([np.radians(getattr(self.model_args, 'camera_pitch_deg', -10.0)), 0.0, 0.0], dtype=float)  # [bench-patch] true start pitch" + NL, 1),
        ('            qid = question["database_idx"]' + NL,
         '            qid = question["database_idx"]' + NL + '            self.model_args.camera_pitch_deg = float(question.get("camera_pitch_deg", -10.0))  # [bench-patch]' + NL, 1),
        # turn-only mode (IMAG_TURN_ONLY=1): multi-view sets are taken from one fixed position, so drop move-forward
        ('                possible_actions = [f"move-forward {self.model_args.fixed_forward_magnitudes}", f"turn-left {self.model_args.fixed_rotation_magnitudes}", f"turn-right {self.model_args.fixed_rotation_magnitudes}"]\n',
         '                possible_actions = [f"move-forward {self.model_args.fixed_forward_magnitudes}", f"turn-left {self.model_args.fixed_rotation_magnitudes}", f"turn-right {self.model_args.fixed_rotation_magnitudes}"]\n'
         '                if os.environ.get("IMAG_TURN_ONLY") == "1":  # [bench-patch]\n'
         '                    possible_actions = [a for a in possible_actions if not a.startswith("move-forward")]\n', 1),
        ('        ckpt_path = os.environ.get("WAN_CKPT_PATH", None)\n',
         '        ckpt_path = os.environ.get("WAN_CKPT_PATH", None)\n'
         '        assert ckpt_path and os.path.isfile(ckpt_path), f"[bench-patch] WAN_CKPT_PATH not a file: {ckpt_path}"\n', 1),
        ('                print(f"[WARN] Failed to load finetuned ckpt from {ckpt_path}: {e}. Proceeding with base weights.")\n',
         '                raise  # [bench-patch] never fall back to base weights\n', 1),
    ])


def patch_avic(root):
    root = Path(root) / "visual_spatial_reasoning"
    shutil.copy(HERE / "gemini_backend.py", root / "utils/gemini_backend.py")
    edit(root / "utils/api.py", [
        ('\nif __name__ == "__main__":\n',
         '\nfrom utils.gemini_backend import GeminiChatAPI  # noqa: E402,F811 [bench-patch] standard 3.7 backend overrides the 2026-08 adapter\n\n'
         'if __name__ == "__main__":\n', 1),
    ])
    edit(root / "utils/args.py", [
        ('        choices=["gpt-4o", "gpt-4.1", "o4-mini", "o1", "gemini-3.5-flash", "gemini-2.5-flash", "OpenGVLab/InternVL3-8B", "OpenGVLab/InternVL3-14B"],\n', "", 1),
        ('        choices=[None, "None", "gpt-4o", "gpt-4.1", "o4-mini", "o1", "gemini-3.5-flash", "gemini-2.5-flash", "OpenGVLab/InternVL3-8B", "OpenGVLab/InternVL3-14B"],\n', "", 1),
    ])
    edit(root / "utils/vlm_wrapper.py", [
        ('        if provider == "gemini":\n            self.model = GeminiChatAPI(model_name)\n            self.qa_model = None\n',
         '        self.is_gemini = provider == "gemini"  # [bench-patch]\n        self.curr_choices = None\n'
         '        if provider == "gemini":\n            self.model = GeminiChatAPI(model_name)\n            self.qa_model = None\n', 1),
        ("                      sys_prompt=None, policy_text=None, **kwargs):\n",
         "                      sys_prompt=None, policy_text=None, **kwargs):\n        self.curr_choices = answer_choices  # [bench-patch]\n", 1),
        ("            content = format_gpt_content(content)\n",
         "            content = format_gpt_content(content)\n" + WRAPPER_GEMINI_RUN, 1),
    ])
    edit(root / "stable_virtual_camera/seva/eval.py", [
        # optional fixed metric translation scale (SVC units per metre) instead of per-trajectory normalisation
        ("    w2c[:, :3, 3] *= translation_scaling_factor\n",
         "    if os.environ.get(\"SVC_METRIC_SCALE\"):  # [bench-patch]\n"
         "        translation_scaling_factor = float(os.environ[\"SVC_METRIC_SCALE\"])\n"
         "    w2c[:, :3, 3] *= translation_scaling_factor\n", 1),
    ])
    edit(root / "utils/prompt_formatting.py", [
        # turn-only mode (IMAG_TURN_ONLY=1): remove move-forward from the macro action space and say the position is fixed
        ("\ndef format_spatial_vqa_prompt_policy_plan(\n", "\ndef _orig_format_spatial_vqa_prompt_policy_plan(\n", 1),
    ])
    pf = root / "utils/prompt_formatting.py"
    if "turn-only wrapper" not in pf.read_text(encoding="utf-8"):
        with open(pf, "a", encoding="utf-8") as f:
            f.write('''

import os  # noqa: E402


def format_spatial_vqa_prompt_policy_plan(*args, **kwargs):  # [bench-patch] turn-only wrapper
    sys_prompt, content = _orig_format_spatial_vqa_prompt_policy_plan(*args, **kwargs)
    if os.environ.get("IMAG_TURN_ONLY") == "1":
        sys_prompt = "\\n".join(l for l in sys_prompt.split("\\n") if not l.startswith("- move-forward"))
        sys_prompt = sys_prompt.replace('"move-forward"|"turn-left"|"turn-right"', '"turn-left"|"turn-right"')
        sys_prompt = sys_prompt.replace("Constraints:\\n", "Constraints:\\n- All images were taken from ONE fixed position: you can only turn in place, never move.\\n")
    if os.environ.get("AVIC_ROUTE_HINT") == "1":
        sys_prompt = sys_prompt.replace("Constraints:\\n", "Constraints:\\n- If the question describes a hypothetical movement (walking, turning, facing something), plan macro actions that reproduce that movement as closely as possible, in the same order, and end facing the direction the question describes.\\n")
    return sys_prompt, content
''')
    edit(root / "pipelines/pipeline_avic.py", [
        # the --policy_majority_threshold arg existed but was ignored (hard-coded call_n > skip_n); honour it.
        # At the default 0.5 the decision is identical to upstream.
        ('        decision = "call_wm" if call_n > skip_n else "skip"\n',
         '        decision = "call_wm" if call_n / max(1, call_n + skip_n) > self.model_args.policy_majority_threshold else "skip"  # [bench-patch]\n', 1),
        ('            if t not in ("move-forward", "turn-left", "turn-right"):\n                return None\n',
         '            if t not in ("move-forward", "turn-left", "turn-right"):\n                return None\n'
         '            if t == "move-forward" and os.environ.get("IMAG_TURN_ONLY") == "1":  # [bench-patch]\n                return None\n', 1),
        # SVC img2trajvid_s-prob renders only num_targets+1 poses, taking the FIRST ones; macro plans expand to up to
        # ~120 atomic poses, so everything after the first 8 atomic steps was silently dropped while keyframes were still
        # sampled proportionally over the full plan. Subsample the poses evenly over the whole trajectory instead.
        ('    c2ws = traj["camera_extrinsic"]\n    T = len(c2ws)\n',
         '    c2ws = traj["camera_extrinsic"]\n    T = len(c2ws)\n'
         '    _nt = max(1, min(model_args.num_targets, T - 1))  # [bench-patch] even subsample over the full trajectory\n'
         '    _svc_c2ws = [c2ws[int(round(i))] for i in np.linspace(0, T - 1, _nt + 1)] if T - 1 > _nt else list(c2ws)\n', 1),
        ("        c2ws=traj['camera_extrinsic'],\n", "        c2ws=_svc_c2ws,  # [bench-patch]\n", 1),
        (HELPER_RE, helper_block(12, resize_else=True), 1),
        ("[primary_img_path, helper_img_path] if helper_img_path else [primary_img_path]", HELPERS_LIST, 3),
        ("[primary_img_path, helper_img_path] if len(question['img_paths']) > 1 else [primary_img_path]", HELPERS_LIST, 2),
        ("[primary_img_path] + ([helper_img_path] if helper_img_path else [])", HELPERS_LIST, 1),
        (PROCESS_ANSWER_RE, PROCESS_ANSWER, 1),
        ("    theta = np.array([np.radians(-20.0), 0.0, 0.0], dtype=float)" + NL,
         "    theta = np.array([np.radians(getattr(model_args, 'camera_pitch_deg', -20.0)), 0.0, 0.0], dtype=float)  # [bench-patch] true start pitch" + NL, 1),
        ('            qid = question["database_idx"]' + NL,
         '            qid = question["database_idx"]' + NL + '            self.model_args.camera_pitch_deg = float(question.get("camera_pitch_deg", -20.0))  # [bench-patch]' + NL, 1),
    ])


if __name__ == "__main__":
    {"mj": patch_mj, "avic": patch_avic}[sys.argv[1]](sys.argv[2])
