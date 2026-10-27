# ImagiRep: Representing the Unseen

Code, model outputs and evaluation for the paper
**Representing the Unseen: Rethinking Visual Imagination for Spatial Reasoning**
(Zengqi Zhao\*, Zun Wang\*, Shoubin Yu, Mohit Bansal, Yue Zhang; UNC Chapel Hill; \* equal contribution).

[Project page](https://imagirep.github.io/) · [Paper (arXiv)](https://arxiv.org/abs/XXXX.XXXXX) ·
[Benchmark (Hugging Face)](https://huggingface.co/datasets/Imagirep/ImagiRep)

ImagiRep has 922 four-option spatial reasoning questions about unobserved viewpoints in Matterport3D scenes
(Viewpoint Imagination: 622; Cross-view Integration: 300). For every question the imagined state is given as endpoint
RGB and six alternative representations: top-down geometry, spatial text, video, depth, surface normals and panorama.
GEARS is a test-time strategy that decides when to use imagined evidence and which representation to give the model.

This repository holds code and per-question results. The benchmark itself (questions, images, videos) and the
estimated representations are on Hugging Face behind a Matterport3D Terms of Use agreement.

## Setup

```bash
git clone https://github.com/Imagirep/ImagiRep.git && cd ImagiRep
pip install -r requirements.txt
# request access on the Hugging Face page first, then download into the repository root:
huggingface-cli download Imagirep/ImagiRep --repo-type dataset --local-dir .
```

The dataset uses the same folder layout as this repository, so the download puts every image and video where the
scripts expect it (`benchmark/`, and the estimated representations under `03_*/` and `04_*/`).

## Reproduce the paper's numbers

```bash
python 07_verification/scripts/verify_from_raw.py
```

recomputes every reported number from the stored per-question outputs (no model calls; needs the `benchmark/`
download for the question index). `07_verification/README.md` maps each number to its files.

Input conditions: `observed` (observed view(s) only), `oracle` (+ endpoint RGB) and `oracle_<m>` (+ representation `m`:
`topdown`, `text`, `video`, `depth`, `surface_normals`, `panorama`). For each added representation the paper reports a
recovery score: a question counts as correct if it is answered correctly with endpoint RGB alone or with endpoint RGB
plus that representation.

| folder | content | paper tables (LaTeX labels) |
|---|---|---|
| `01_oracle_gpt61sol/` | GPT-6.1-sol with oracle representations: request builder, run script, per-question answers | `oracle_imagination`, `task_breakdown_gpt` |
| `02_oracle_gemini37/` | Gemini 3.7 Flash with oracle representations | `oracle_imagination`, `task_breakdown_gemini`, `representation_breakdown`, `per_type_selection` |
| `03_estimated_viewpoint_gpt61sol/` | HY-World 2.0 estimation (trajectory planner, patch to the upstream code) and GPT-6.1-sol answers | `task_breakdown_estimated` |
| `04_estimated_crossview_gpt61sol/` | SEVA view synthesis + MoGe-2 depth pipeline and GPT-6.1-sol answers | `task_breakdown_estimated_cv` |
| `05_gears/` | GEARS gate-2 routing prompts (`viewpoint_gating_routing/gate_prompts.py`), routing scripts and choices, ablation | `wm_baselines` (GEARS), `gating_ablation` |
| `06_baselines_mindjourney_avic/` | MindJourney and AVIC: our changes to the upstream code, run scripts, final answers | `wm_baselines` |
| `07_verification/` | number-by-number verification | all |

Notes:
- The requests in `01_oracle_gpt61sol/` use the same image files as `benchmark/benchmark922/images/`; the 16 video
  frames per `oracle_video` request are not included (sample them from `benchmark/benchmark922/videos/`).
- The MindJourney / AVIC imagined views (about 18 GB) are not included; the final answers in
  `06_baselines_mindjourney_avic/runs_gpt61sol/qa/` are, and they are all `verify_from_raw.py` needs.
- GEARS gate 1 compares two existing answers (observed only vs. observed + estimated endpoint RGB); gate 2 asks the
  model which representation to add, and the final answer is read from that condition's run.
- Some conditions were run in several batches with the same model, prompts and settings; `07_verification/README.md`
  lists the files behind each number.
- `07_verification/scripts/verify_from_raw.py` runs as released. The build, generation and run scripts in `01_*` to
  `06_*` are kept as the record of how the results were produced; several of them read intermediate folders that are
  not part of the release.
- Paper runs: GPT-6.1-sol (Responses API, reasoning effort high) and Gemini 3.7 Flash (thinking HIGH, temperature 0).
  API keys come from environment variables; scripts that used SSH to a GPU host read `JUMP_HOST`, `GPU_HOST`,
  `SSH_USER` and `SSH_PASSWORD` from the environment.

## Third-party code

Not redistributed. Each folder gives the upstream repository, the exact commit and our changes as a patch:
MindJourney (`06_*/base_code/upstream_*`), AVIC (same), HY-World 2.0 (`03_*/generation_code/hyworld2_upstream/`);
VGGT-Ω, Stable Virtual Camera and MoGe-2 for Cross-view Integration (`04_*/generation_code/README.md`).

## License

Code: MIT (`LICENSE`). Questions, model outputs and other result files: CC BY-NC 4.0. The benchmark data on Hugging
Face is derived from Matterport3D and subject to the Matterport3D Terms of Use.

## Citation

```bibtex
@article{zhao2026imagirep,
  title   = {Representing the Unseen: Rethinking Visual Imagination for Spatial Reasoning},
  author  = {Zhao, Zengqi and Wang, Zun and Yu, Shoubin and Bansal, Mohit and Zhang, Yue},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
