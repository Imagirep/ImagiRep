cd 01_oracle_gpt61sol/question_only_922; set -a; source <secrets_dir>/openai_lab.env; set +a; export PYTHONNOUSERSITE=1
<external>/imagination_baselines_mj_avic_20261006/envs/mj/bin/python run_gpt.py 01_oracle_gpt61sol/question_only_922 --workers 32 > full.log 2>&1
