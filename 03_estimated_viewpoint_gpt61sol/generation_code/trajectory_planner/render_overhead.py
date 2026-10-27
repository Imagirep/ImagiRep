"""Roof-removed overhead render of a generated scene around a navigator route (CPU Gaussian splatting).

Usage: render_overhead.py PLAN SCENE OUT_DIR [--w 640 --h 480] [--views top,oblique60,oblique45] [--cut_below_ceiling 0.3]
Gaussians whose centre is above (measured ceiling - cut) are removed; the ceiling height is the median upward
mesh ray hit over the route. Cameras look at the route region from above. Plain render: no sharpening.
Also writes a copy with the route (start -> end) drawn for reference.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nav_env  # noqa: E402


def K(w, h, vfov):
    f = h / (2 * math.tan(math.radians(vfov) / 2))
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)


def project(env, c2w, k, pts_m):
    """Nav-frame metres -> pixel coords for the drawn route."""
    world = np.array([env.scale * (env.R @ env.u(p)) + env.t for p in pts_m])
    cam = (world - c2w[:3, 3]) @ c2w[:3, :3]
    ok = cam[:, 2] > .01
    return np.stack([k[0, 0] * cam[:, 0] / cam[:, 2] + k[0, 2], k[1, 1] * cam[:, 1] / cam[:, 2] + k[1, 2]], -1), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('plan'); ap.add_argument('scene'); ap.add_argument('out')
    ap.add_argument('--w', type=int, default=640); ap.add_argument('--h', type=int, default=480); ap.add_argument('--vfov', type=float, default=60.0)
    ap.add_argument('--views', default='top,oblique60,oblique45'); ap.add_argument('--cut_below_ceiling', type=float, default=0.3)
    ap.add_argument('--margin', type=float, default=3.0)
    ap.add_argument('--back', type=float, default=None, help='metres kept behind the start (unobserved side); default = margin')
    ap.add_argument('--tag', default='')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = nav_env.Env(a.scene); sp = env.splat
    kfs = json.loads(Path(a.plan).read_text())['keyframes']
    route = np.array([[k['x'], k['y']] for k in kfs])
    # ceiling: upward ray hits (scene units -> metres above eye) along the route
    rays = np.array([[*env.u([x, y, 0.0]), 0, 0, 1] for x, y in route], np.float32)
    up = env.cast(rays) * env.mpu; up = up[np.isfinite(up)]
    ceiling = float(np.median(up)) if len(up) else 1.2
    cut = ceiling - a.cut_below_ceiling
    # gaussian heights in metres relative to eye level (nav frame z)
    nav = ((sp.means - env.t) @ env.R) / env.scale * env.mpu
    keep = nav[:, 2] < cut
    for name in ('means', 'cov', 'opacity', 'color'):
        setattr(sp, name, getattr(sp, name)[keep])
    # region of interest: route bbox + margin, clipped to walkable-grid extent
    lo = route.min(0) - a.margin; hi = route.max(0) + a.margin
    if a.back is not None: hi[0] = route[:, 0].max() + a.back
    c = (lo + hi) / 2; span = hi - lo
    k = K(a.w, a.h, a.vfov); tv = math.tan(math.radians(a.vfov / 2)); th = tv * a.w / a.h
    floor = -nav_env.EYE_M
    meta = dict(ceiling_above_eye_m=ceiling, cut_above_eye_m=cut, eye_m=nav_env.EYE_M, removed_gaussians=int((~keep).sum()), kept_gaussians=int(keep.sum()),
                roi_center=c.tolist(), roi_span_m=span.tolist(), views={})
    for v in a.views.split(','):
        if v == 'top':
            # image right = nav +Y (right of start), image up = nav -X (forward of start): yaw 0, pitch straight down
            hgt = max(span[1] / 2 / th, span[0] / 2 / tv) + 0.0
            x, y, z, yaw, pitch = c[0], c[1], floor + hgt, 0.0, -math.radians(89.9)
        else:
            p = math.radians(float(v.removeprefix('oblique')))
            # stand behind the region (start side, +X) and look forward (-X) and down
            dist = max(span[0], span[1] / (a.w / a.h)) * 0.95
            x, y, z, yaw, pitch = c[0] + dist * math.cos(p), c[1], floor + dist * math.sin(p), 0.0, -p
        c2w = env.pose(x, y, yaw, pitch, z_m=z)
        rgb, dep, alpha = sp.render(c2w, k, a.w, a.h)
        img = cv2.cvtColor(np.uint8(np.clip(rgb, 0, 1) * 255 + .5), cv2.COLOR_RGB2BGR)
        name = f'overhead{a.tag}_{v}_{a.w}x{a.h}.png'; cv2.imwrite(str(out / name), img)
        pts = [[k_['x'], k_['y'], floor + 0.05] for k_ in kfs]
        uv, ok = project(env, c2w, k, pts); ov = img.copy(); t = max(2, a.w // 320)
        for i in range(len(uv) - 1):
            if ok[i] and ok[i + 1]:
                cv2.line(ov, tuple(np.int32(uv[i])), tuple(np.int32(uv[i + 1])), (0, 220, 255), t, cv2.LINE_AA)
        if ok[0]: cv2.circle(ov, tuple(np.int32(uv[0])), 4 * t, (0, 200, 0), -1, cv2.LINE_AA)
        if ok[-1]: cv2.circle(ov, tuple(np.int32(uv[-1])), 4 * t, (0, 0, 230), -1, cv2.LINE_AA)
        cv2.imwrite(str(out / name.replace('.png', '_route.png')), ov)
        meta['views'][v] = dict(camera_nav_m=[x, y, z], yaw=yaw, pitch_deg=math.degrees(pitch), coverage=float((alpha > .5).mean()), image=name)
    (out / f'overhead{a.tag}_manifest_{a.w}x{a.h}.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({k_: v_ for k_, v_ in meta.items() if k_ != 'views'}), json.dumps({v: round(m['coverage'], 3) for v, m in meta['views'].items()}))


if __name__ == '__main__':
    main()
