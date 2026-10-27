#!/usr/bin/env bash
# 2026-10-11: MindJourney final evaluation. Waits until all generation slots have exited, then
# runs the MindJourney FINAL-config QA replay (top3+label) on fo622 and mvc300 and writes summary.md (AVIC parts of the
# summary are partial until AVIC is resumed).
T=06_baselines_mindjourney_avic/runs_gpt61sol
R=<external>/imagination_baselines_mj_avic_20261006
sleep 300
while pgrep -u "$USER" -f "bash code/slot" > /dev/null; do sleep 300; done
echo "[$(date -Is)] generation slots finished"
set -a; source <secrets_dir>/openai_personal.env; source <secrets_dir>/openai_lab.env; set +a
export GPT_TEST_ROOT=$T GPT_KEY_ENV=OPENAI_PERSONAL_API_KEY GPT_FALLBACK_KEY_ENV=OPENAI_LAB_API_KEY GPT_BUDGET_DIR=$T/usage PYTHONNOUSERSITE=1
for SET in fo622 mvc300; do
  I=$R/bench_code/${SET}_ids.json
  GPT_USAGE_LOG=$T/usage/qa_${SET}_mj_top3+label.jsonl $R/envs/mj/bin/python $T/code/qa_variants_gpt.py mj $I top3+label \
    --src_abs "$T/results/A/mj_${SET}*" --tag $SET --workers 12
done
python3 $T/code/summarize.py > $T/summary.md
echo "[$(date -Is)] final eval (MJ) done"
