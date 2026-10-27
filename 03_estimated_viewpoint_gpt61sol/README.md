# Estimated representations, Viewpoint Imagination (HY-World 2.0)

Paper table: `task_breakdown_estimated`; also the inputs of GEARS on Viewpoint Imagination.

- `generation_code/`: how the estimated imagined state is produced from the observed view and the movement instruction.
  - `trajectory_planner/`: plans the endpoint camera inside the HY-World 2.0 scene from the instruction.
  - `hyworld2_upstream/`: upstream repository, exact commit and our patch (the upstream code is not redistributed).
  - `reference_geometry/`: ground-truth geometry used only to evaluate the planner, never as model input.
- `estimated_assets/`
  - `hyworld_v5_packages/<scene>/<view>/`: rendered representations per endpoint (RGB, depth, normals, panorama,
    top-down, video, text) and `manifest.json`. Images and videos are in the Hugging Face dataset.
  - `request_source/`: the requests built from these packages (`tasks.jsonl`) and their assets.
  - `topdown_arrow_rerender/`: re-renders the top-down maps with an enlarged final-facing arrow (`old2new.json` maps
    old to new image names).
- `gpt_runs/`: GPT-6.1-sol requests (`tasks.jsonl`, 622 questions x 7 conditions; `images/` from the Hugging Face
  dataset), run scripts and per-question results (`results/<condition>/<uid>.json`). The top-down map in these requests
  is the version with an enlarged final-facing arrow. Videos were sent as 16 sampled frames.
- `canonical_622_rows/`: the 622 questions with their evidence paths.
