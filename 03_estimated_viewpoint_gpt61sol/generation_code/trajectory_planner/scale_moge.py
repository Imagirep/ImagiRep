"""Per-scene metric scale from the INITIAL IMAGE ONLY (MoGe-2 metric depth) vs the generated scene's start-view depth.

Allowed inputs: the initial image and the generated scene. No Oracle/benchmark data.
Output JSON: metres per original HY scene unit (moge), the eye-height-prior value for comparison, and robustness stats.
Usage: scale_moge.py <scene_source> <initial_image> <out.json>
"""
import json, math, sys, time
from pathlib import Path
import numpy as np, cv2, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nav_env  # noqa: E402

VFOV = 60.0


def main():
    src, img_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    t0 = time.time()
    torch.set_num_threads(16)
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal').eval()
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    fov_x = math.degrees(2 * math.atan(math.tan(math.radians(VFOV / 2)) * W / H))
    with torch.inference_mode():
        res = model.infer(torch.from_numpy(img).float().permute(2, 0, 1) / 255.0, fov_x=fov_x)
    depth_m = res['depth'].cpu().numpy(); mask = res['mask'].cpu().numpy().astype(bool)
    env = nav_env.Env(src)
    f = H / (2 * math.tan(math.radians(VFOV) / 2))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
    _, dep_export, alpha = env.splat.render(env.pose(0, 0, 0, 0), K, W, H)
    dep_units = dep_export / env.scale                      # original HY units
    valid = mask & (alpha > .5) & np.isfinite(depth_m) & np.isfinite(dep_units) & (dep_units > 1e-4) & (depth_m > .3) & (depth_m < 15)
    ratio = depth_m[valid] / dep_units[valid]
    q = np.percentile(ratio, [10, 25, 50, 75, 90])
    result = {'m_per_unit_moge': float(q[2]), 'ratio_percentiles_10_25_50_75_90': [float(x) for x in q],
              'valid_fraction': float(valid.mean()), 'eye_units': env.eye,
              'm_per_unit_eye_1p6': 1.6 / env.eye, 'implied_eye_height_m': float(q[2] * env.eye),
              'moge_median_depth_m': float(np.median(depth_m[valid])), 'fov_x_deg': fov_x, 'seconds': time.time() - t0,
              'inputs': ['initial_image', 'generated_scene']}
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
