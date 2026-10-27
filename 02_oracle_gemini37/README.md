# Gemini 3.7 Flash with oracle representations

Paper tables: `oracle_imagination` (Gemini columns), `task_breakdown_gemini`, `representation_breakdown`,
`per_type_selection`.

Settings for every call: Gemini 3.7 Flash, temperature 0, thinking level HIGH, answers constrained to A-D. Videos are
sent as mp4.

- `viewpoint_imagination/`: the 622 Viewpoint Imagination questions were evaluated in several runs with the same model,
  prompts and settings (`fo_batch*`, `fo_622_supplement_*`, `future201_four_condition_20260913/runs/*`).
  `single_view_modality_tables_b1_4_20261005/sv622.py` assembles the per-question answers of all runs into the reported
  tables; `fo_622_question_only_trajectory_20261006/` is the question-only baseline.
- `crossview_integration/`: the 300 Cross-view Integration questions come from three cohorts
  (`mvc_ready100_*`, `mvc_batch1_last50_*`, `mvc_mvc131_supplement_*`; observed, oracle RGB and question-only), and the
  six representations were run in `mvc_oracle_modalities_325_20261006/` and `mvc_mvc131_supplement_20261006/`.
  `mvc_oracle_modalities_325_20261006/results/summary_mvc300.json` lists the 300 questions.
- `tables/`: the table outputs.

Each run folder keeps its runner script, prompts and per-question results; folder names are run names ending in the
run date (`YYYYMMDD`).
`07_verification/scripts/verify_from_raw.py` shows exactly which files make up each reported number.
