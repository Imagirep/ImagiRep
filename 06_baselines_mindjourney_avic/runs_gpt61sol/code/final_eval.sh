#!/usr/bin/env bash
# Final evaluation, single view only (2026-10-11). Waits until generation
# slots and any running QA replay have exited, then completes the FINAL-config replays on fo622 (MJ top3+label, AVIC
# label; resume-safe, finished questions are skipped) and writes summary.md (multi-view rows there are raw pipeline answers).
T=06_baselines_mindjourney_avic/runs_gpt61sol
R=<external>/imagination_baselines_mj_avic_20261006
while pgrep -u "$USER" -f "bash code/slot" > /dev/null; do sleep 120; done
while pgrep -u "$USER" -f "qa_variants_gpt.py" > /dev/null; do sleep 30; done
echo "[$(date -Is)] generation and running replays finished"
set -a; source <secrets_dir>/openai_company.env; set +a
export GPT_TEST_ROOT=$T GPT_KEY_ENV=OPENAI_COMPANY_API_KEY GPT_BUDGET_DIR=$T/usage PYTHONNOUSERSITE=1
I=$R/bench_code/fo622_ids.json
GPT_USAGE_LOG=$T/usage/qa_fo622_mj_top3+label.jsonl $R/envs/mj/bin/python $T/code/qa_variants_gpt.py mj $I top3+label \
  --src_abs "$T/results/A/mj_fo622*" --tag fo622 --workers 12
GPT_USAGE_LOG=$T/usage/qa_fo622_avic_label.jsonl $R/envs/avic/bin/python $T/code/qa_variants_gpt.py avic $I label \
  --src_abs "$T/results/A/avic_fo622*" --tag fo622 --workers 12
python3 $T/code/summarize.py > $T/summary.md
echo "[$(date -Is)] final eval done"
