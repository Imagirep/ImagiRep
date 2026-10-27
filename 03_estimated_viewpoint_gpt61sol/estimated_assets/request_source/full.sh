cd 03_estimated_viewpoint_gpt61sol/estimated_assets/request_source; set -a; source <(sed "1s/^\xEF\xBB\xBF//" <secrets_dir>/gemini_api_key.env); set +a; export PYTHONNOUSERSITE=1
<external>/imagination_baselines_mj_avic_20261006/envs/mj/bin/python run_rounds.py 03_estimated_viewpoint_gpt61sol/estimated_assets/request_source --rounds 1,2 --workers 48 > full.log 2>&1
