#!/usr/bin/env bash
# 2026-10-11: finish the CV (mvc300) FINAL-config answer replays for MJ (top3+label) and AVIC (label),
# so that CV baselines use the same protocol as VI. Lab key only, no fallback; separate usage dir with a $15 hard stop.
# Resume-safe: questions already replayed (MJ 47) are skipped. Regenerates summary.md afterwards.
T=06_baselines_mindjourney_avic/runs_gpt61sol
R=<external>/imagination_baselines_mj_avic_20261006
U=$T/usage_cvreplay_lab_20261011
mkdir -p $U; echo 15 > $U/BUDGET_USD
set -a; source <secrets_dir>/openai_lab.env; set +a
export GPT_TEST_ROOT=$T GPT_KEY_ENV=OPENAI_LAB_API_KEY GPT_BUDGET_DIR=$U PYTHONNOUSERSITE=1
unset GPT_FALLBACK_KEY_ENV
I=$R/bench_code/mvc300_ids.json
echo "[$(date -Is)] start"
GPT_USAGE_LOG=$U/qa_mvc300_mj_top3+label.jsonl $R/envs/mj/bin/python $T/code/qa_variants_gpt.py mj $I top3+label \
  --src_abs "$T/results/A/mj_mvc300*" --tag mvc300 --workers 12 > $U/mj.log 2>&1 &
GPT_USAGE_LOG=$U/qa_mvc300_avic_label.jsonl $R/envs/avic/bin/python $T/code/qa_variants_gpt.py avic $I label \
  --src_abs "$T/results/A/avic_mvc300*" --tag mvc300 --workers 12 > $U/avic.log 2>&1 &
wait
echo "[$(date -Is)] replays done"
python3 $T/code/summarize.py > $T/summary.md
echo "[$(date -Is)] summary written"
