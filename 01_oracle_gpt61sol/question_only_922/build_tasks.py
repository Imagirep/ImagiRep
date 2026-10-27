"""GPT-6.1-sol question-only (text_only) requests for the 922 benchmark questions, 2026-10-09.

Matches the Gemini 3.7 Flash question-only runs exactly:
  FO 622 (fo_622_question_only_trajectory_20261006, runner_snapshot text_only): system = FO SYSTEM_PROMPT; the only
     user part is question_text(row) = "Question: ...\n<options>\nAnswer:" (no images).
  MVC 300 (cohort run.py text_only): system = prompts/text_system.txt; the only user part is
     f"{prompts/text_only.txt}\n\nQuestion: {q}\n{options}\nAnswer:" (identical prompt files in all three cohorts).
Built from the observed rows of the GPT-6.1-sol payload (gpt61sol_observed_oracle_922_20261007/tasks.jsonl): their last
part is exactly the FO question_text / the MVC "<original_only.txt>\n\nQuestion: ..." block, so only the images and
the MVC condition prompt are removed/replaced.

Usage: python build_tasks.py <gpt_payload_dir> <out_dir> <mvc_text_system.txt> <mvc_text_only.txt>
"""
import collections, json, sys
from pathlib import Path

G, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MSYS = Path(sys.argv[3]).read_text(encoding='utf-8').strip(); MTXT = Path(sys.argv[4]).read_text(encoding='utf-8').strip()
(OUT / 'images').mkdir(parents=True, exist_ok=True)  # run_gpt.py expects the folder; no images are used
out = []
for t in map(json.loads, open(G / 'tasks.jsonl', encoding='utf-8')):
    if t['condition'] != 'observed':
        continue
    q = t['parts'][-1]['text']
    if t['benchmark'] == 'FO622':
        assert q.startswith('Question: ') and q.endswith('Answer:'), t['uid']
        parts, system = [{'text': q}], t['system']
    else:
        i = q.index('\n\nQuestion: ')
        parts, system = [{'text': MTXT + q[i:]}], MSYS
    out.append({**t, 'condition': 'question_only', 'system': system, 'parts': parts})
with open(OUT / 'tasks.jsonl', 'w', encoding='utf-8') as f:
    for t in out:
        f.write(json.dumps(t, ensure_ascii=False) + '\n')
print(len(out), dict(collections.Counter(t['benchmark'] for t in out)))
