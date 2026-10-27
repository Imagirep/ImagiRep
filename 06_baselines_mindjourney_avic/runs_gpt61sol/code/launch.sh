#!/usr/bin/env bash
# Launch the variant-A full runs on GPUs 4-7: per GPU 2 MindJourney slots + 1 AVIC slot (same layout as the
# 2026-10-06 Gemini full run), then the final-eval watcher. Chunking as before: MJ fo622 22 / mvc300 10; AVIC 11 / 5.
T=06_baselines_mindjourney_avic/runs_gpt61sol
R=<external>/imagination_baselines_mj_avic_20261006
cd $T; mkdir -p queue/claims logs usage results inputs/fo622 inputs/mvc300
cp -n $R/inputs/fo622/test.json inputs/fo622/; cp -n $R/inputs/mvc300/test.json inputs/mvc300/
[ -f usage/BUDGET_USD ] || echo 90 > usage/BUDGET_USD
[ -f queue/jobs_mj.txt ] || { for i in $(seq 0 21); do echo "fo622 22 $i"; done; for i in $(seq 0 9); do echo "mvc300 10 $i"; done; } > queue/jobs_mj.txt
[ -f queue/jobs_avic.txt ] || { for i in $(seq 0 10); do echo "fo622 11 $i"; done; for i in $(seq 0 4); do echo "mvc300 5 $i"; done; } > queue/jobs_avic.txt
for g in 4 5 6 7; do
  nohup bash code/slot.sh avic $g avic_g$g > /dev/null 2>&1 < /dev/null & sleep 20
done
for g in 4 5 6 7; do for s in a b; do
  nohup bash code/slot.sh mj $g mj_${s}_g$g > /dev/null 2>&1 < /dev/null & sleep 15
done; done
nohup bash code/final_eval.sh > logs/final_eval.log 2>&1 < /dev/null &
echo launched
