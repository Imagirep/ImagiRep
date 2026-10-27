"""Render one clear mid-route frame of an estimated (navigator) trajectory at higher resolution.

Usage: render_midframe.py PLAN SCENE OUT_DIR [--lo 0.4 --hi 0.6 --n 7 --w 1920 --h 1440]
Same camera model as render_package.py (vertical FOV 60 deg, NAV_EYE_M eye height); only the image size changes.
Candidates along the middle of the walking frames are scored at 640x480 by splat coverage x Laplacian variance
(a clarity measure only; no sharpening is applied), then the best one is re-rendered at W x H. CPU only.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nav_env  # noqa: E402
import render_package as rp  # noqa: E402


def intrinsics(w, h):
    f = h / (2 * math.tan(math.radians(rp.VFOV) / 2))
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('plan'); ap.add_argument('scene'); ap.add_argument('out')
    ap.add_argument('--lo', type=float, default=.4); ap.add_argument('--hi', type=float, default=.6); ap.add_argument('--n', type=int, default=7)
    ap.add_argument('--w', type=int, default=1920); ap.add_argument('--h', type=int, default=1440)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rp.ENV = nav_env.Env(a.scene)
    plan = json.loads(Path(a.plan).read_text()); fr = rp.frames(plan['keyframes'])
    # walking frames = frames whose position differs from the previous frame
    walk = [i for i in range(1, len(fr)) if math.dist([fr[i]['x'], fr[i]['y']], [fr[i - 1]['x'], fr[i - 1]['y']]) > 1e-6]
    picks = sorted({walk[int(round(p * (len(walk) - 1)))] for p in np.linspace(a.lo, a.hi, a.n)})
    scores = []
    for i in picks:
        rgb, _, alpha = rp.render_rgb(fr[i])
        g = cv2.cvtColor(np.uint8(np.clip(rgb, 0, 1) * 255), cv2.COLOR_RGB2GRAY)
        s = float((alpha > .5).mean()) * float(cv2.Laplacian(g, cv2.CV_64F).var())
        scores.append(dict(frame=i, walk_fraction=walk.index(i) / (len(walk) - 1), coverage=float((alpha > .5).mean()), score=s))
        rp.write_rgb(out / f'candidate_{i:04d}.png', rgb)
    best = max(scores, key=lambda r: r['score'])
    rgb, _, _ = rp.render_rgb(fr[best['frame']], intrinsics(a.w, a.h), a.w, a.h)
    name = f"midroute_frame_{best['frame']:04d}_{a.w}x{a.h}.png"; rp.write_rgb(out / name, rgb)
    meta = dict(plan=a.plan, scene=a.scene, total_frames=len(fr), walking_frames=len(walk), candidates=scores, chosen=best,
                pose=fr[best['frame']], width=a.w, height=a.h, vfov_degrees=rp.VFOV, eye_m=nav_env.EYE_M, sharpening=False, image=name)
    (out / 'midframe_manifest.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(dict(image=name, chosen=best)))


if __name__ == '__main__':
    main()
