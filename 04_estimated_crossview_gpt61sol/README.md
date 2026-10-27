# Estimated representations, Cross-view Integration (SEVA + MoGe-2)

Paper table: `task_breakdown_estimated_cv`; also the inputs of GEARS on Cross-view Integration.

- `generation_code/`: how the estimated view is produced from the observed images (VGGT-Ω camera poses, Stable Virtual
  Camera view synthesis, MoGe-2 depth); see `generation_code/README.md`.
- `gpt_runs/`: GPT-6.1-sol requests (`tasks.jsonl`, 300 questions x 2 conditions: `est_rgb`, `est_rgb_depth`),
  request builder and run script, and per-question results (`results/<condition>/<uid>.json`).
  `inputs/` (estimated RGB and depth per view group) and `images/` (the exact request images) are in the Hugging Face
  dataset.
- `compiled_predictions/mvc_preds.json`: per-question answers (`observed`, `est_rgb`, `est_rgb_depth`) for the 300
  questions, read by `05_gears/ablation/`.
