#!/usr/bin/env bash
# Final evaluation of the tuned configs on the full sets (2026-10-07). Re-uses existing QA replays (skip if present).
#   MV300: MJ turn-only (+ top3+label), AVIC turn-only thr0.3 (+ label), each vs its own no-imagination base
#   SV622: MJ bench (+ top3+label), AVIC bench (+ label), each vs base; original pipeline results reported too
R=<external>/imagination_baselines_mj_avic_20261006; cd $R
set -a; source <(sed '1s/^\xEF\xBB\xBF//' <secrets_dir>/gemini_api_key.env); set +a; export PYTHONNOUSERSITE=1
QA(){ m=$1; shift; envs/$m/bin/python bench_code/qa_variants.py $m "$@" 2>&1 | grep -E "^$m "; }

# ---------- MV300 ----------
while [ "$(ls results/avic_bench_turn_t0.3/mvc*/*/*/step_0/gpt.json 2>/dev/null | wc -l)" -lt 300 ]; do sleep 60; done
python3 -c "import json;json.dump([x['database_idx'] for x in json.load(open('inputs/mvc300/test.json'))],open('bench_code/mvc300_ids.json','w'))"
I=bench_code/mvc300_ids.json
( for v in base top3+label; do QA mj $I $v --src 'mj_bench_turn_r2f6e6/mvc*_*' --tag _mv150turn --workers 10; done ) &
( for v in base label; do QA avic $I $v --src 'avic_bench_turn_t0.3/mvc*_*' --tag _mv150t03 --workers 10; done ) &
wait
python3 bench_code/summarize_runs.py $I mj_turn_raw=mj_bench_turn_r2f6e6/mvc*_*:mj mj_base=qa:qa_variants_mv150turn/mj/base mj_FINAL_turn_top3L=qa:qa_variants_mv150turn/mj/top3+label \
  avic_t03_raw=avic_bench_turn_t0.3/mvc*_*:avic avic_base=qa:qa_variants_mv150t03/avic/base avic_FINAL_t03_label=qa:qa_variants_mv150t03/avic/label \
  --dump queue/final_mv300_perq.json > queue/final_mv300.txt 2>&1

# ---------- SV622 ----------
while [ "$(ls results/mj_bench_r2f6e6/fo622_*/*/*/gpt.json | wc -l)" -lt 622 ] || [ "$(ls results/avic_bench/fo622_*/*/*/step_0/gpt.json | wc -l)" -lt 622 ]; do sleep 60; done
python3 -c "import json;json.dump([x['database_idx'] for x in json.load(open('inputs/fo622/test.json'))],open('bench_code/fo622_ids.json','w'))"
I=bench_code/fo622_ids.json
( for v in base top3+label; do QA mj $I $v --workers 10; done ) &
( for v in base label; do QA avic $I $v --workers 10; done ) &
wait
python3 bench_code/summarize_runs.py $I mj_orig=mj_bench_r2f6e6/fo622_*:mj mj_base=qa:qa_variants/mj/base mj_FINAL_top3L=qa:qa_variants/mj/top3+label \
  avic_orig=avic_bench/fo622_*:avic avic_base=qa:qa_variants/avic/base avic_FINAL_label=qa:qa_variants/avic/label \
  --dump queue/final_sv622_perq.json > queue/final_sv622.txt 2>&1
echo "[$(date -Is)] final_eval done" >> queue/final_sv622.txt
