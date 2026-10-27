# GEARS: gated evidence and adaptive representation selection

Paper tables: `wm_baselines` (GEARS column), `gating_ablation`. Model: GPT-6.1-sol, reasoning effort high.

GEARS answers each question in two stages, using the estimated evidence of `03_*` and `04_*`:

1. **Gate.** Compare the answer from the observed view(s) alone with the answer from the observed view(s) plus the
   estimated endpoint RGB. If they agree, that answer is final.
2. **Routing.** Otherwise the model is shown the question and the available options and picks the evidence to use:
   the observed view(s) only, or one estimated representation. The final answer is that condition's answer.

All answers are read from the stored per-question runs (`01_*` for observed, `03_*` / `04_*` for estimated);
only the routing step makes new model calls.

- `viewpoint_gating_routing/`: routing prompts (`gate_prompts.py`), routing script (`gate2_select_7opt.py`) and the
  routing choices (`results/gate2sel_S2/<uid>.json`, field `choice`).
- `crossview_gating_routing/`: the same for Cross-view Integration (`gate2_mvc.py`, `results_gate2/S2MV/`); options are
  observed only, estimated RGB, or estimated RGB + depth.
- `ablation/`: `ablation_eval.py` scores No Gate, +Gating, +Routing and Gating+Routing
  (`python ablation_eval.py eval --all`); `build_allreps.py` builds the No Gate requests (all estimated representations
  in one request).
