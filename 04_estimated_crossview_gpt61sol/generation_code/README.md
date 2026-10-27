# Cross-view Integration estimation code

Code that produced the estimated representations for Cross-view Integration
(Table `tab:task_breakdown_estimated_cv`, Appendix `app:estimation`):

1. **Camera poses:** VGGT-Ω (`VGGT-Omega-1B-512`) estimates the cameras of the two or three observed images.
2. **Target view:** with two images, the target camera keeps the shared centre and takes the orientation midway
   between Image 1 and Image 2; with three images, a short camera path is planned from the estimated cameras.
3. **Estimated RGB:** Stable Virtual Camera (SEVA) v1.1 synthesizes the target view from the observed images and
   their estimated poses (`img2trajvid`, 21 frames, short side 384).
4. **Estimated depth:** MoGe-2 (`Ruicheng/moge-2-vitl-normal`, ViT-L) predicts depth on the synthesized RGB; the
   depth image uses reversed Turbo colours (red near, blue far).

The pipeline and its options are documented in `multiview_pipeline/README.md`
(entry point `multiview_pipeline/run_case.py`).

## Outputs used in the paper

`../gpt_runs/inputs/Dylan_MultiView_Generated_RGB_Depth_143_Scenes.zip`, unpacked as
`../gpt_runs/inputs/Dylan_MultiView_Generated_RGB_Depth/<view group>/{rgb.png, depth.png}`:
143 view groups (138 with two observed images, 5 with three), 512 x 384 RGB and depth.
All questions of a view group share the same observed images, so one estimated RGB/depth pair serves all of them.
`../gpt_runs/build_tasks.py` builds the GPT-6.1-sol requests from these images.

## Third-party code and weights

Not redistributed. Install the upstream repositories and download the weights from their original sources:

| component | repository | known compatible revision | license |
|---|---|---|---|
| VGGT-Ω | https://github.com/facebookresearch/vggt-omega | `39a0cb8af88554f15ddcb5354cd52bde588fa014` | FAIR Noncommercial Research License |
| Stable Virtual Camera | https://github.com/Stability-AI/stable-virtual-camera | `fe19948e9b7bea261ab2db780a59656131404a83` | Stability AI Non-Commercial License |
| MoGe-2 | https://github.com/microsoft/MoGe | (model `Ruicheng/moge-2-vitl-normal`) | MIT (some components Apache 2.0) |

`multiview_pipeline/enable_local_seva_weights.py` is an optional two-line change to the SEVA checkout that lets it
load its VAE and CLIP weights from local paths instead of downloading them.

## Tests

`cd multiview_pipeline && python -m unittest discover -s tests -v` (needs NumPy and SciPy) checks the two-view and
three-view camera paths and the SEVA call without running any model.
