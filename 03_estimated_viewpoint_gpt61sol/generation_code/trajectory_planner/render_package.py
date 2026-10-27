"""Render the six estimated modalities (+RGB) for one navigator plan, desktop FO-0003_Data format.

Inputs: plan.json from navigator.py (keyframes in metres, initial-camera frame) and the
generated scene. CPU Gaussian splatting only; no GPU, no Oracle/labels.
Outputs in OUT: Topdown.png Text.txt Video.mp4 Depth.png Normals.png Panorama.png RGB.png manifest.json
"""
import json, math, multiprocessing as mp, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nav_env  # noqa: E402

W, H, VFOV = 640, 480, 60.0
FOCAL = H / (2 * math.tan(math.radians(VFOV) / 2))
K = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1]], np.float32)
ENV = None


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------- text (oracle-style template) ----------------
def pitch_state(p):
    d = math.degrees(p)
    return 'level' if abs(d) < .05 else f'{abs(d):.1f}° {"above" if d > 0 else "below"} level'


def rotation(y0, p0, y1, p1):
    yaw = math.degrees(wrap(y1 - y0)); pit = math.degrees(p1 - p0); c = []
    if abs(yaw) >= 179.95:
        c.append('turn around 180.0° in place')
    elif abs(yaw) >= .05:
        c.append(f'turn {"right" if yaw > 0 else "left"} {abs(yaw):.1f}° in place')
    if abs(pit) >= .05:
        c.append(f'{"raise" if pit > 0 else "lower"} your view by {abs(pit):.1f}° so it is {pitch_state(p1)}')
    return ' and '.join(c)


def facing(a):
    d = math.degrees(wrap(a))
    if abs(d) < .05:
        return 'in the same direction as your starting view'
    if abs(d) >= 179.95:
        return 'opposite your starting direction (180.0°)'
    return f'{abs(d):.1f}° to the {"right" if d > 0 else "left"} of your starting direction'


def text(keyframes, lookback):
    """Steps = walk keyframes; turns between walks attach as 'After arriving' of the previous step."""
    lines = ['Begin at the position and viewing direction shown in the starting image. Use that starting direction as 0°. '
             'All left and right turns below are relative to your current view.', 'Your initial view is level.']
    walks = [i for i, k in enumerate(keyframes) if k['kind'] == 'walk']
    if not walks:
        raise ValueError('plan has no movement')
    # the lookback turn = final in-place turn after the last walk, if the motion ends facing the start
    last_walk = walks[-1]
    tail = keyframes[last_walk + 1:]
    look = tail[-1] if lookback and tail else None
    tail_end = tail[-2] if look is not None and len(tail) >= 2 else (keyframes[last_walk] if look is not None else (tail[-1] if tail else keyframes[last_walk]))
    total = 0.0; prev = keyframes[0]
    for n, wi in enumerate(walks, 1):
        before = [k for k in keyframes[(walks[n - 2] + 1 if n > 1 else 1):wi] if k['kind'] == 'turn']
        start = keyframes[walks[n - 2]] if n > 1 else keyframes[0]
        walk = keyframes[wi]
        heading_pose = before[-1] if before else start
        clauses = []
        if n == 1:
            t = rotation(start['yaw'], start['pitch'], heading_pose['yaw'], heading_pose['pitch'])
            clauses.append(t[0].upper() + t[1:] + '.' if t else 'Keep your current viewing direction.')
        else:
            clauses.append('Keep your current viewing direction.')
        d = math.dist([heading_pose['x'], heading_pose['y']], [walk['x'], walk['y']]); total += d
        heading = math.atan2(walk['y'] - heading_pose['y'], -(walk['x'] - heading_pose['x']))
        verb = 'backward' if abs(wrap(heading - walk['yaw'])) > math.pi / 2 else 'forward'
        clauses.append(f'Move {verb} {d:.2f} m to the next position, keeping that viewing direction during the movement.')
        nxt = walks[n] if n < len(walks) else None
        if nxt is not None:
            after = [k for k in keyframes[wi + 1:nxt] if k['kind'] == 'turn']
            target = after[-1] if after else walk
        else:
            target = tail_end
        t = rotation(walk['yaw'], walk['pitch'], target['yaw'], target['pitch'])
        clauses.append('After arriving, ' + t + '.' if t else 'After arriving, keep facing the same direction.')
        lines.append(f'Step {n}: ' + ' '.join(clauses))
        prev = target
    final = prev
    if look is not None:
        t = rotation(prev['yaw'], prev['pitch'], look['yaw'], look['pitch'])
        lines.append('At the endpoint, ' + (t if t else 'keep your current viewing direction') + '. Stay at the same position; do not walk any farther.')
        final = look
    lines.append(f'Stop here. You have traveled {total:.2f} m in total across {len(walks)} movement segment(s). '
                 f'You are now facing {facing(final["yaw"])}, with your view {pitch_state(final["pitch"])}.')
    return '\n\n'.join(lines) + '\n', total


# ---------------- frames ----------------
def frames(keyframes):
    out = [dict(keyframes[0]) for _ in range(12)]
    for a, b in zip(keyframes, keyframes[1:]):
        if b['kind'] == 'walk':
            d = math.dist([a['x'], a['y']], [b['x'], b['y']]); n = max(6, int(math.ceil(d / .9 * 12)))
            for s in np.linspace(0, 1, n + 1)[1:]:
                out.append({'x': a['x'] + (b['x'] - a['x']) * s, 'y': a['y'] + (b['y'] - a['y']) * s, 'yaw': b['yaw'], 'pitch': b['pitch']})
        else:
            dy = wrap(b['yaw'] - a['yaw']); n = max(6, int(math.ceil(abs(math.degrees(dy)) / 7.5)))
            for s in np.linspace(0, 1, n + 1)[1:]:
                out.append({'x': b['x'], 'y': b['y'], 'yaw': a['yaw'] + dy * s, 'pitch': a['pitch'] + (b['pitch'] - a['pitch']) * s})
    out += [dict(out[-1]) for _ in range(12)]
    return out


def render_rgb(f, k=K, w=W, h=H):
    rgb, dep, alpha = ENV.splat.render(ENV.pose(f['x'], f['y'], f['yaw'], f['pitch']), k, w, h)
    return rgb, dep, alpha


def frame_job(args):
    i, f, d = args
    rgb, _, _ = render_rgb(f)
    cv2.imwrite(os.path.join(d, f'{i:05d}.png'), cv2.cvtColor(np.uint8(np.clip(rgb, 0, 1) * 255 + .5), cv2.COLOR_RGB2BGR))
    return i


def write_rgb(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(np.uint8(np.clip(rgb, 0, 1) * 255 + .5), cv2.COLOR_RGB2BGR))


def depth_color(d, mask):
    lo, hi = np.percentile(d[mask], [2, 98]); g = np.uint8(np.clip((d - lo) / (hi - lo + 1e-8), 0, 1) * 255)
    im = cv2.applyColorMap(255 - g, cv2.COLORMAP_TURBO); im[~mask] = 0
    return im, [float(lo), float(hi)]


def normals_png(depth, valid):
    yy, xx = np.mgrid[:H, :W]; xyz = np.stack([(xx - W / 2) / FOCAL * depth, (yy - H / 2) / FOCAL * depth, depth], -1)
    n = np.cross(np.gradient(xyz, axis=1), np.gradient(xyz, axis=0)); n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    n[np.sum(n * xyz, axis=-1) > 0] *= -1; n[..., 1] *= -1
    edge = np.maximum(np.abs(np.gradient(depth, axis=0)), np.abs(np.gradient(depth, axis=1)))
    nv = valid & (edge < .08 * np.maximum(depth, .01)); n[~nv] = 0
    im = np.uint8(np.clip((n + 1) / 2 * 255, 0, 255)); im[~nv] = 0
    return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)


def panorama(f):
    c2w = ENV.pose(f['x'], f['y'], f['yaw'], f['pitch']); cube = []; bases = []
    k = np.array([[256, 0, 256], [0, 256, 256], [0, 0, 1]], np.float32)
    cf, cr, cu = c2w[:3, 2], c2w[:3, 0], -c2w[:3, 1]
    for fd, up in [(cf, cu), (cr, cu), (-cf, cu), (-cr, cu), (cu, -cf), (-cu, cf)]:
        rt = np.cross(fd, up); rt /= np.linalg.norm(rt); up = np.cross(rt, fd)
        b = np.column_stack([rt, -up, fd]); c = c2w.copy(); c[:3, :3] = b
        face, _, _ = ENV.splat.render(c, k, 512, 512); cube.append(face); bases.append(b)
    ph, pw = 512, 1024; v, a = np.mgrid[:ph, :pw]; yaw = ((a + .5) / pw - .5) * 2 * np.pi; lat = (.5 - (v + .5) / ph) * np.pi
    ray = cf * (np.cos(lat) * np.cos(yaw))[..., None] + cr * (np.cos(lat) * np.sin(yaw))[..., None] + cu * np.sin(lat)[..., None]
    idx = np.stack([ray @ b[:, 2] for b in bases], -1).argmax(-1); pano = np.zeros((ph, pw, 3), np.float32)
    for j, (face, b) in enumerate(zip(cube, bases)):
        z = ray @ b[:, 2]
        mx = ((ray @ b[:, 0]) / np.maximum(z, 1e-8) * 256 + 255.5).astype(np.float32); my = ((ray @ b[:, 1]) / np.maximum(z, 1e-8) * 256 + 255.5).astype(np.float32)
        warped = cv2.remap(face, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); m = idx == j; pano[m] = warped[m]
    return pano


def topdown(keyframes, final):
    """Generated-mesh geometry near floor/eye height, full route, numbered waypoints 0..N, final heading arrow."""
    v = ENV.mesh_vertices * ENV.mpu; keep = (v[:, 2] >= -nav_env.EYE_M - .15) & (v[:, 2] <= .5)
    pts = v[keep, :2]; route = [[k['x'], k['y']] for k in keyframes if k['kind'] in ('start', 'walk')]
    allp = np.vstack([pts, np.array(route)]); lo = np.percentile(allp, .5, axis=0) - .3; hi = np.percentile(allp, 99.5, axis=0) + .3
    span = max(hi - lo); scale = 856 / span; canvas = np.full((1000, 1000, 3), 255, np.uint8)
    # image up = forward (-X), right = +Y
    P = lambda q: (int((q[1] - lo[1]) * scale + 72), int((q[0] - lo[0]) * scale + 72))
    uv = np.array([P(q) for q in pts]); m = (uv[:, 0] >= 0) & (uv[:, 0] < 1000) & (uv[:, 1] >= 0) & (uv[:, 1] < 1000); uv = uv[m]
    canvas[uv[:, 1], uv[:, 0]] = (195, 185, 170); canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    cv2.polylines(canvas, [np.array([P(q) for q in route], np.int32)], False, (20, 20, 20), 6)
    end = route[-1]; arrow = [end[0] - .6 * math.cos(final['yaw']), end[1] + .6 * math.sin(final['yaw'])]
    cv2.arrowedLine(canvas, P(end), P(arrow), (0, 0, 0), 5, tipLength=.25)
    for j, q in enumerate(route):
        a = P(q); cv2.circle(canvas, a, 22, (255, 255, 255), -1); cv2.circle(canvas, a, 22, (0, 0, 0), 4)
        cv2.putText(canvas, str(j), (a[0] - 9, a[1] + 9), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
    return canvas


def main():
    global ENV
    plan_path, source, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    lookback = '--lookback' in sys.argv
    plan = json.loads(plan_path.read_text()); t0 = time.time()
    if out.exists():
        raise SystemExit('refuse to overwrite ' + str(out))
    out.mkdir(parents=True)
    ENV = nav_env.Env(source)
    kfs = plan['keyframes']; final = kfs[-1]
    txt, total = text(kfs, lookback); (out / 'Text.txt').write_text(txt, encoding='utf-8')
    rgb, depth, alpha = render_rgb(final); write_rgb(out / 'RGB.png', rgb)
    valid = (alpha > .5) & np.isfinite(depth) & (depth > 0)
    im, rng = depth_color(depth, valid); cv2.imwrite(str(out / 'Depth.png'), im)
    cv2.imwrite(str(out / 'Normals.png'), normals_png(depth, valid))
    write_rgb(out / 'Panorama.png', panorama(final))
    cv2.imwrite(str(out / 'Topdown.png'), topdown(kfs, final))
    fr = frames(kfs); tmp = tempfile.mkdtemp(dir=str(out))
    with mp.get_context('fork').Pool(int(os.environ.get('RENDER_PROCS', '16'))) as pool:
        list(pool.imap_unordered(frame_job, [(i, f, tmp) for i, f in enumerate(fr)], chunksize=2))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '12', '-i', os.path.join(tmp, '%05d.png'), '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'Video.mp4')], check=True)
    shutil.rmtree(tmp)
    manifest = {'schema': 'hy-estimated-modalities/desktop-v1', 'question_id': plan['question_id'], 'motion_text': plan['motion_text'],
                'view_variant': 'lookback' if lookback else 'endpoint', 'frames': len(fr), 'fps': 12, 'total_distance_m': total,
                'depth_display_range_scene_units': rng, 'depth_color_scale_m': {'near_m': rng[0] / ENV.scale * ENV.mpu, 'far_m': rng[1] / ENV.scale * ENV.mpu}, 'depth_source': 'CPU GS expected depth', 'normals_source': 'GS expected depth gradients',
                'scale': f'eye height prior {nav_env.EYE_M} m over generated floor distance', 'm_per_unit': ENV.mpu,
                'render_backend': 'CPU anisotropic Gaussian splatting (degree-0 SH)', 'oracle_used': False,
                'planner_inputs': plan.get('inputs_allowed'), 'seconds': time.time() - t0}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
