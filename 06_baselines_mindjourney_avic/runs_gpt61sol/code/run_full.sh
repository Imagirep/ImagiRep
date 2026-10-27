#!/usr/bin/env bash
# GPT-6.1-sol MindJourney (mj) / AVIC (avic) full runs, variant A (2026-10-10); one chunk per call.
#   bash run_full.sh <mj|avic> <fo622|mvc300> <A|B> <gpu> [num_chunks chunk_idx]
# Generation settings = the FINAL configs of imagination_baselines_mj_avic_20261006 (run_shard.sh):
#   SV (fo*): bench motion; MV (mvc*): turn-only, AVIC policy threshold 0.3. MJ rank selection 2/6/6.
# Variants (company OpenAI key, $90 budget guard in gpt_backend.py):
# API key: read from the environment variable named in GPT_KEY_ENV.
#   A = 1+2: intermediate calls (MJ scores, AVIC policy/verifier) at reasoning effort medium; MJ scoring images detail low
#   B = 1+2+3+4: A + AVIC policy samples 5 -> 3 + MJ beams 2 -> 1
# Final-answer calls always use effort high. QA replays (MJ top3+label, AVIC label) are run afterwards by qa_variants_gpt.py.
set -uo pipefail
METHOD=$1 SET=$2 VAR=$3 GPU=$4 NCH=${5:-1} IDX=${6:-0}
R=<external>/imagination_baselines_mj_avic_20261006
T=06_baselines_mindjourney_avic/runs_gpt61sol
Z=<external>
OUT=$T/results/$VAR/${METHOD}_${SET}
mkdir -p $OUT $T/logs $T/usage $Z/tmp/gpt61t_${METHOD}_${GPU}_${IDX}
set -a; source $Z/secrets/openai_company.env; set +a
export GPT_KEY_ENV=OPENAI_COMPANY_API_KEY GPT_BUDGET_DIR=$T/usage   # usage/BUDGET_USD caps the spend
export GPT_USAGE_LOG=$T/usage/${VAR}_${METHOD}_${SET}_c${IDX}.jsonl
export GPT_EFFORT_INTERMEDIATE=medium GPT_LOWDETAIL_ROLES=prompt_scores
BEAMS=2; SAMPLES=5
[[ $VAR == B ]] && { BEAMS=1; SAMPLES=3; }
export CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HOME=$Z/huggingface HUGGINGFACE_HUB_CACHE=$Z/huggingface/hub TORCH_HOME=$Z/cache/torch
export XDG_CACHE_HOME=$Z/cache/xdg TMPDIR=$Z/tmp/gpt61t_${METHOD}_${GPU}_${IDX}
COMMON="--num_questions 100000 --input_dir $T/inputs/$SET --split test --question_type None
        --num_question_chunks $NCH --question_chunk_idx $IDX --max_images 3 --max_tries_gpt 5"
TURN=0; [[ $SET == mvc* ]] && TURN=1
export IMAG_TURN_ONLY=$TURN

if [[ $METHOD == mj ]]; then
  cd $R/code/MindJourney; PY=$R/envs/mj/bin/python
  export PYTHONPATH=$R/code/MindJourney WORLD_MODEL_TYPE=wan2.2 QUESTION_DATASET_TYPE=SAT_test
  export WAN_CKPT_PATH=$(ls $R/weights/MindJourney-World-Model/*.ckpt 2>/dev/null | head -1)
  MOTION="--sampling_interval_angle 15 --sampling_interval_meter 0.5 --fixed_rotation_magnitudes 45 --fixed_forward_magnitudes 1.5
          --max_steps_per_question 4 --max_turn_angle 180 --max_forward_distance 6 --frame_interval 3"
  CMD="$PY pipelines/pipeline_wan_scaling_beam_search_double_rank.py $COMMON $MOTION
       --vlm_model_name gpt-6.1-sol --vlm_qa_model_name None --output_dir $OUT
       --scaling_strategy beam_search_double_rank --helpful_score_threshold 8 --exploration_score_threshold 8
       --num_top_candidates 18 --num_beams $BEAMS --num_frames 25 --max_inference_batch_size 3 --inference_step 20
       --helpful_topk 2 --helpful_floor 6 --exploration_floor 6"
else
  V=$R/code/AVIC/visual_spatial_reasoning; cd $V; PY=$R/envs/avic/bin/python
  export PYTHONPATH=$V WORLD_MODEL_TYPE=svc HF_HUB_OFFLINE=1
  THR=0.5; [[ $TURN == 1 ]] && THR=0.3
  MOTION="--policy_action_mode macro --max_macro_actions 6 --macro_forward_values 0.5,1.0,1.5
          --macro_turn_values 15,30,45,60,90,135,180 --sampling_interval_angle 15 --sampling_interval_meter 0.25
          --max_action_ids_cap 120"
  CMD="$PY pipelines/pipeline_avic.py $COMMON $MOTION --provider gemini --vlm_model_name gpt-6.1-sol
       --output_dir $OUT --scaling_strategy spatial_beam_search --max_imagined_keyframes 5 --wm_condition_source question_image
       --num_policy_samples $SAMPLES --max_wm_candidates 5 --policy_majority_threshold $THR
       --fixed_rotation_magnitudes 27 --fixed_forward_magnitudes 0.75 --max_steps_per_question 3 --num_top_candidates 6
       --num_beams 3 --num_frames 9 --frame_interval 3 --max_inference_batch_size 1
       --task img2trajvid_s-prob --replace_or_include_input True --cfg 4.0 --guider 1 --L_short 576 --num_targets 8
       --use_traj_prior True --chunk_strategy interp"
fi

LOG=$T/logs/${VAR}_${METHOD}_${SET}_c${IDX}_gpu${GPU}.log
echo "[$(date -Is)] start $METHOD $SET $VAR chunk $IDX/$NCH gpu $GPU (beams $BEAMS, policy samples $SAMPLES)" | tee -a $LOG
for attempt in $(seq 1 ${MAX_RESTARTS:-10}); do
  $CMD >> $LOG 2>&1; rc=$?
  [[ $rc == 0 ]] && { echo "[$(date -Is)] done (attempt $attempt)" | tee -a $LOG; exit 0; }
  [[ $rc == 2 ]] && { echo "[$(date -Is)] stop (rc 2: budget/usage)" | tee -a $LOG; exit 2; }
  [[ -f $T/usage/BUDGET_STOP ]] && { echo "[$(date -Is)] budget stop" | tee -a $LOG; exit 2; }
  echo "[$(date -Is)] exit $rc -> restart $attempt" | tee -a $LOG; sleep 30
done
echo "[$(date -Is)] FAILED after restarts" | tee -a $LOG; exit 1
