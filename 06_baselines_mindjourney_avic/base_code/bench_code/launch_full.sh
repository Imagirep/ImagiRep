#!/usr/bin/env bash
# Full run: FO622 + MVC300 for MindJourney (bench, rank 2/6/6) and AVIC (bench) on GPUs 0-3.
#   bash launch_full.sh <stage>   stage: avic | mj | monitor
# Per GPU: 1 AVIC slot + 2 MJ slots. Jobs are pulled from queue/jobs_<method>.txt (dynamic, atomic claims).
R=<external>/imagination_baselines_mj_avic_20261006
cd $R; mkdir -p queue/claims
[ -f queue/jobs_mj.txt ] || { for i in $(seq 0 21); do echo "fo622 22 $i"; done; for i in $(seq 0 9); do echo "mvc300 10 $i"; done; } > queue/jobs_mj.txt
[ -f queue/jobs_avic.txt ] || { for i in $(seq 0 10); do echo "fo622 11 $i"; done; for i in $(seq 0 4); do echo "mvc300 5 $i"; done; } > queue/jobs_avic.txt

release() {  # reserve GPU $1 for our runs, then stop its park holder
  touch park/RESERVE_GPU$1 park/STOP_GPU$1
  for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -lt 2000 ] && return 0; sleep 2; done
  echo "GPU$1 did not free" >&2; return 1
}

case $1 in
  avic)
    for g in 0 1 2 3; do release $g || exit 1; done
    tmux kill-session -t smoke_mj 2>/dev/null; tmux kill-session -t smoke_avic 2>/dev/null
    for g in 0 1 2 3; do tmux kill-session -t imag_park_gpu$g 2>/dev/null; done
    for g in 0 1 2 3; do tmux new-session -d -s full_avic_g$g "bash bench_code/queue_slot.sh avic $g avic_g$g"; sleep 20; done ;;
  mj)
    for g in 0 1 2 3; do for s in a b; do tmux new-session -d -s full_mj_${s}_g$g "bash bench_code/queue_slot.sh mj $g mj_${s}_g$g"; sleep 15; done; done ;;
  monitor)
    tmux has-session -t full_monitor 2>/dev/null || tmux new-session -d -s full_monitor "bash bench_code/queue_monitor.sh" ;;
esac
tmux ls | grep -E "full_|park"
