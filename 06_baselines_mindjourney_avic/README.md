# MindJourney and AVIC baselines

Paper table: `wm_baselines`. Both methods use GPT-6.1-sol as the final reasoner.

- `base_code/upstream_MindJourney/`, `base_code/upstream_AVIC/`: upstream repository, exact commit and our changes as a
  patch (the upstream code is not redistributed). `base_code/bench_code/`: adapters that turn ImagiRep questions into the
  methods' inputs.
- `runs_gpt61sol/code/`: run scripts (`run_full.sh`), the GPT backend and the final question-answering replay
  (`qa_variants_gpt.py`, `final_eval*.sh`).
- `runs_gpt61sol/results/A/`: per-run records of the imagination stage. The imagined views themselves (about 18 GB) are
  not included.
- `runs_gpt61sol/qa/`: the final answers that the paper reports (`fo622`, `mvc300`; MindJourney answers from its three
  most helpful imagined views, AVIC from its selected trajectory).
