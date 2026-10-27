"""Re-render estimated Topdown.png with a larger, visible final-heading arrow (2026-10-09).

Same map, route and numbered waypoints as render_package.topdown(); only the final arrow changes: it is drawn after
the waypoint circles (so they no longer cover it), starts at the edge of the last circle and is longer and thicker.
For every package the original drawing is first reproduced from the same plan + generated mesh and must be
pixel-identical to the existing Topdown.png; only then is the new image written (in place).

Inputs per job: plan JSON (keyframes, metres), scene dir (result/ply/fuse_post.ply + position_meta_info.json),
eye height NAV_EYE_M used for that render (scale/<sid>.json implied_eye_height_m, or 1.75 for non-MoGe variant plans).
Usage: python rerender_topdown_arrow.py jobs.json [--dry]   (jobs: [{"plan","scene","eye","out"}])
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2, open3d as o3d

ARROW_PX, ARROW_W, TIP = 120, 9, .35


def load_mesh_vertices(source):
    source = Path(source)
    meta = json.loads((source / 'result/ply/position_meta_info.json').read_text())
    f = np.array(meta['facing_direction'], float); f /= np.linalg.norm(f)
    u = np.array(meta['up_direction'], float); u /= np.linalg.norm(u)
    r = np.cross(f, u); r /= np.linalg.norm(r)
    R = np.column_stack([-f, r, u]); T = np.eye(4); T[:3, :3] = meta['scale'] * R; T[:3, 3] = np.array(meta['center_point'])
    mesh = o3d.io.read_triangle_mesh(str(source / 'result/ply/fuse_post.ply')); mesh.transform(np.linalg.inv(T))
    rays = o3d.t.geometry.RaycastingScene(); rays.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    down = rays.cast_rays(o3d.core.Tensor(np.array([[0, 0, 0, 0, 0, -1]], np.float32)))['t_hit'].numpy()[0]
    return np.asarray(mesh.vertices), float(down)


def draw(verts, eye_units, eye_m, keyframes, final, new):
    mpu = eye_m / eye_units
    v = verts * mpu; keep = (v[:, 2] >= -eye_m - .15) & (v[:, 2] <= .5)
    pts = v[keep, :2]; route = [[k['x'], k['y']] for k in keyframes if k['kind'] in ('start', 'walk')]
    allp = np.vstack([pts, np.array(route)]); lo = np.percentile(allp, .5, axis=0) - .3; hi = np.percentile(allp, 99.5, axis=0) + .3
    span = max(hi - lo); scale = 856 / span; canvas = np.full((1000, 1000, 3), 255, np.uint8)
    P = lambda q: (int((q[1] - lo[1]) * scale + 72), int((q[0] - lo[0]) * scale + 72))
    uv = np.array([P(q) for q in pts]); m = (uv[:, 0] >= 0) & (uv[:, 0] < 1000) & (uv[:, 1] >= 0) & (uv[:, 1] < 1000); uv = uv[m]
    canvas[uv[:, 1], uv[:, 0]] = (195, 185, 170); canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    cv2.polylines(canvas, [np.array([P(q) for q in route], np.int32)], False, (20, 20, 20), 6)
    end = route[-1]
    if not new:   # original arrow (0.6 m, under the circles)
        arrow = [end[0] - .6 * math.cos(final['yaw']), end[1] + .6 * math.sin(final['yaw'])]
        cv2.arrowedLine(canvas, P(end), P(arrow), (0, 0, 0), 5, tipLength=.25)
    for j, q in enumerate(route):
        a = P(q); cv2.circle(canvas, a, 22, (255, 255, 255), -1); cv2.circle(canvas, a, 22, (0, 0, 0), 4)
        cv2.putText(canvas, str(j), (a[0] - 9, a[1] + 9), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
    if new:       # image up = forward (-X), right = +Y  ->  heading yaw points to (sin yaw, -cos yaw) in image
        c = np.array(P(end), float); d = np.array([math.sin(final['yaw']), -math.cos(final['yaw'])])
        a0 = c + 24 * d; a1 = c + (24 + ARROW_PX) * d
        cv2.arrowedLine(canvas, tuple(int(round(x)) for x in a0), tuple(int(round(x)) for x in a1), (0, 0, 0), ARROW_W,
                        line_type=cv2.LINE_AA, tipLength=TIP)
    return canvas


def main():
    jobs = json.load(open(sys.argv[1])); dry = '--dry' in sys.argv
    by_scene = defaultdict(list)
    for j in jobs:
        by_scene[j['scene']].append(j)
    ok = bad = 0
    for scene, js in by_scene.items():
        verts, eye_units = load_mesh_vertices(scene)
        for j in js:
            plan = json.load(open(j['plan'])); kfs = plan['keyframes']; final = kfs[-1]
            old = draw(verts, eye_units, float(j['eye']), kfs, final, new=False)
            cur = cv2.imread(j['out'])
            if cur is None or cur.shape != old.shape or not np.array_equal(cur, old):
                bad += 1; print('MISMATCH', j['out'], None if cur is None else int(np.abs(cur.astype(int) - old).sum()), flush=True)
                continue
            ok += 1
            if not dry:
                cv2.imwrite(j['out'], draw(verts, eye_units, float(j['eye']), kfs, final, new=True))
    print(json.dumps({'reproduced_and_replaced' if not dry else 'reproduced': ok, 'mismatch': bad}), flush=True)


if __name__ == '__main__':
    main()
