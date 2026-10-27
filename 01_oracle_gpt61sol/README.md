# GPT-6.1-sol with oracle representations

Paper tables: `oracle_imagination` (GPT columns), `task_breakdown_gpt`.

- `observed_oracle_8conditions_922/`: all 922 questions under 8 conditions.
  - `tasks.jsonl` (`observed`, `oracle`) and `tasks_mod.jsonl` (`oracle_<m>` for the six representations): the exact
    requests of the paper runs. Image names refer to `benchmark/benchmark922/images/` (same files). For `oracle_video`,
    the video was sent as 16 frames sampled uniformly in time; those frame images are not included and can be sampled
    from `benchmark/benchmark922/videos/`.
  - `run_gpt.py`, `full.sh`, `full_mod.sh`: run scripts (Responses API, reasoning effort high).
  - `results/<condition>/<uid>.json`: one answer per question and condition; `summary.json`, `summary_mod_positive.json`.
- `question_only_922/`: the question-only baseline (question and options, no images).
