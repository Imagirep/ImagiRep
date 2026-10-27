#!/usr/bin/env bash
# One shard of the MindJourney (mj) or AVIC (avic) baseline on one GPU, with crash-resume.
#   bash run_shard.sh <mj|avic> <set: smoke10|fo622|mvc325> <num_chunks> <chunk_idx> <gpu> [variant]
# variant: bench (default; motion ranges adapted to the benchmark) | orig (upstream motion settings)
# Both pipelines resume from <out>/results.json and skip finished questions, so a crash just restarts the process.
# Requires RUN_CONFIRMED=1 (model calls + GPU).
set -uo pipefail
METHOD=$1 SET=$2 NCH=$3 IDX=$4 GPU=$5 VARIANT=${6:-bench}
[[ "${RUN_CONFIRMED:-0}" == 1 ]] || { echo "set RUN_CONFIRMED=1" >&2; exit 3; }
R=<external>/imagination_baselines_mj_avic_20261006
Z=<external>
TAG=${VARIANT}${MJ_HELPFUL:+_h$MJ_HELPFUL}   # MJ_HELPFUL overrides the helpfulness threshold (default 8)
[[ "${IMAG_TURN_ONLY:-0}" == 1 ]] && TAG=${TAG}_turn   # turn-only action space (multi-view)
[[ $METHOD == avic && -n "${AVIC_THR:-}" ]] && TAG=${TAG}_t${AVIC_THR}   # AVIC policy majority threshold
[[ $METHOD == avic && "${AVIC_ROUTE_HINT:-0}" == 1 ]] && TAG=${TAG}_route   # route-following hint in the policy prompt
[[ -n "${SVC_METRIC_SCALE:-}" ]] && TAG=${TAG}_m${SVC_METRIC_SCALE}_n${AVIC_TARGETS:-8}   # fixed SVC metric scale
# MJ rank-based selection "topk,helpful_floor,exploration_floor" (bench default 2,6,6; "off" = upstream thresholds only)
MJ_RANK=${MJ_RANK:-$([[ $VARIANT == bench ]] && echo 2,6,6 || echo off)}
if [[ $METHOD == mj && $MJ_RANK != off ]]; then IFS=, read -r RK HF EF <<< "$MJ_RANK"; TAG=${TAG}_r${RK}f${HF}e${EF}
  RANKARGS="--helpful_topk $RK --helpful_floor $HF --exploration_floor $EF"; else RANKARGS=""; fi
OUT=$R/results/${METHOD}_${TAG}/${SET}
mkdir -p $OUT $R/logs $R/usage $Z/tmp/imag_run_${METHOD}_${GPU}_${IDX}
set -a; source <(sed '1s/^\xEF\xBB\xBF//' $Z/secrets/gemini_api_key.env); set +a
export CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HOME=$Z/huggingface HUGGINGFACE_HUB_CACHE=$Z/huggingface/hub TORCH_HOME=$Z/cache/torch
export XDG_CACHE_HOME=$Z/cache/xdg TMPDIR=$Z/tmp/imag_run_${METHOD}_${GPU}_${IDX}
export GEMINI_USAGE_LOG=$R/usage/${METHOD}_${TAG}_${SET}_c${IDX}of${NCH}.jsonl
COMMON="--num_questions 100000 --input_dir $R/inputs/$SET --split test --question_type None
        --num_question_chunks $NCH --question_chunk_idx $IDX --max_images 3 --max_tries_gpt 5"

if [[ $METHOD == mj ]]; then
  cd $R/code/MindJourney; PY=$R/envs/mj/bin/python
  export PYTHONPATH=$R/code/MindJourney WORLD_MODEL_TYPE=wan2.2 QUESTION_DATASET_TYPE=SAT_test
  export WAN_CKPT_PATH=$(ls $R/weights/MindJourney-World-Model/*.ckpt $R/weights/MindJourney-World-Model/*.pt $R/weights/MindJourney-World-Model/*.pth 2>/dev/null | head -1)
  if [[ $VARIANT == orig ]]; then   # upstream SWM script: 27 deg / 0.75 m, frame_interval 1, n=2, caps 60 deg / 1.5 m
    MOTION="--sampling_interval_angle 9 --sampling_interval_meter 0.25 --fixed_rotation_magnitudes 27 --fixed_forward_magnitudes 0.75
            --max_steps_per_question 2 --max_turn_angle 60 --max_forward_distance 1.5 --frame_interval 1"
  else                              # bench: 9 deg / 0.25 m per frame -> 24 frames span 216 deg / 6 m; n=4 reaches 180 deg / 6 m
    MOTION="--sampling_interval_angle 15 --sampling_interval_meter 0.5 --fixed_rotation_magnitudes 45 --fixed_forward_magnitudes 1.5
            --max_steps_per_question 4 --max_turn_angle 180 --max_forward_distance 6 --frame_interval 3"
  fi
  CMD="$PY pipelines/pipeline_wan_scaling_beam_search_double_rank.py $COMMON $MOTION
       --vlm_model_name gemini-3.7-flash --vlm_qa_model_name None --output_dir $OUT
       --scaling_strategy beam_search_double_rank --helpful_score_threshold ${MJ_HELPFUL:-8} --exploration_score_threshold 8
       --num_top_candidates 18 --num_beams 2 --num_frames 25 --max_inference_batch_size ${MJ_BS:-3} --inference_step ${MJ_STEPS:-20} $RANKARGS"
else
  V=$R/code/AVIC/visual_spatial_reasoning; cd $V; PY=$R/envs/avic/bin/python
  export PYTHONPATH=$V WORLD_MODEL_TYPE=svc HF_HUB_OFFLINE=1
  if [[ $VARIANT == orig ]]; then   # upstream AVIC: atomic 0.25 m / 9 deg, <= 6 atomic actions
    MOTION="--policy_action_mode atomic --sampling_interval_angle 9 --sampling_interval_meter 0.25 --max_action_ids_cap 6"
  else                              # bench: 2026-08 macro planner (0.5-1.5 m, 15-180 deg macro actions, <= 6 macros)
    MOTION="--policy_action_mode macro --max_macro_actions 6 --macro_forward_values 0.5,1.0,1.5
            --macro_turn_values 15,30,45,60,90,135,180 --sampling_interval_angle 15 --sampling_interval_meter 0.25
            --max_action_ids_cap 120"
  fi
  CMD="$PY pipelines/pipeline_avic.py $COMMON $MOTION --provider gemini --vlm_model_name gemini-3.7-flash
       --output_dir $OUT --scaling_strategy spatial_beam_search --max_imagined_keyframes 5 --wm_condition_source question_image
       --num_policy_samples ${AVIC_POLICY_SAMPLES:-5} --max_wm_candidates 5 --policy_majority_threshold ${AVIC_THR:-0.5}
       --fixed_rotation_magnitudes 27 --fixed_forward_magnitudes 0.75 --max_steps_per_question 3 --num_top_candidates 6
       --num_beams 3 --num_frames 9 --frame_interval 3 --max_inference_batch_size 1
       --task img2trajvid_s-prob --replace_or_include_input True --cfg 4.0 --guider 1 --L_short 576 --num_targets ${AVIC_TARGETS:-8}
       --use_traj_prior True --chunk_strategy interp"
fi

LOG=$R/logs/${METHOD}_${TAG}_${SET}_c${IDX}of${NCH}_gpu${GPU}.log
echo "[$(date -Is)] start $METHOD/$VARIANT $SET chunk $IDX/$NCH gpu $GPU" | tee -a $LOG
for attempt in $(seq 1 ${MAX_RESTARTS:-20}); do
  $CMD >> $LOG 2>&1; rc=$?
  [[ $rc == 0 ]] && { echo "[$(date -Is)] done (attempt $attempt)" | tee -a $LOG; exit 0; }
  [[ $rc == 2 ]] && { echo "[$(date -Is)] usage error (rc 2); not restarting" | tee -a $LOG; exit 2; }
  echo "[$(date -Is)] exit $rc -> restart $attempt" | tee -a $LOG; sleep 30
done
echo "[$(date -Is)] FAILED after restarts" | tee -a $LOG; exit 1
