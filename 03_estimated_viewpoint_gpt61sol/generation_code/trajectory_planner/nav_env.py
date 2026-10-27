"""Generated-scene navigation environment (CPU only). JSON lines on stdin/stdout.

Reads ONLY the generated scene (GS point cloud, fused mesh, export transform).
No benchmark labels, Oracle trajectories or native scans are accessible here.
Coordinates exposed to the controller are metres in the initial-camera frame:
forward=-X, right=+Y, up=+Z; yaw positive to the right, pitch positive up.
Scale: assumed 1.6 m eye height over the measured generated floor distance.
"""
import base64, heapq, json, math, os, sys, time
from pathlib import Path
import numpy as np, cv2, open3d as o3d

sys.path.insert(0, '<external>/hyworld2_nlp26_full_20261001/scripts/hy_manual_fo0003_20261002')
from cpu_splat import Splat  # noqa: E402

EYE_M = float(os.environ.get('NAV_EYE_M', '1.6'))  # calibrated global eye-height prior
CELL_M = 0.10
BODY_M = 0.25


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Env:
    def __init__(self, source):
        source = Path(source)
        meta = json.loads((source / 'result/ply/position_meta_info.json').read_text())
        f = np.array(meta['facing_direction'], float); f /= np.linalg.norm(f)
        u = np.array(meta['up_direction'], float); u /= np.linalg.norm(u)
        r = np.cross(f, u); r /= np.linalg.norm(r)
        self.R = np.column_stack([-f, r, u]); self.scale = meta['scale']; self.t = np.array(meta['center_point'])
        T = np.eye(4); T[:3, :3] = self.scale * self.R; T[:3, 3] = self.t
        mesh = o3d.io.read_triangle_mesh(str(source / 'result/ply/fuse_post.ply')); mesh.transform(np.linalg.inv(T))
        self.mesh_vertices = np.asarray(mesh.vertices)
        self.rays = o3d.t.geometry.RaycastingScene()
        self.rays.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        down = self.cast(np.array([[0, 0, 0, 0, 0, -1]]))[0]
        if not np.isfinite(down) or down < .1:
            raise RuntimeError('No generated floor below start')
        self.eye = float(down)              # scene units
        self.mpu = EYE_M / self.eye         # metres per scene unit
        self.splat = Splat(source / 'result/ply/point_cloud_1999.ply')
        self.build_grid()
        log('ENV ready', 'eye_units', self.eye, 'm_per_unit', self.mpu, 'grid', self.free.shape)

    # ---------- geometry ----------
    def cast(self, rays):
        hit = self.rays.cast_rays(o3d.core.Tensor(np.asarray(rays, np.float32)))
        return hit['t_hit'].numpy()

    def u(self, m):
        return np.asarray(m, float) / self.mpu

    def build_grid(self):
        v = self.mesh_vertices * self.mpu   # metres
        floor = -EYE_M
        lo = np.maximum(np.percentile(v[:, :2], 1, axis=0) - .5, -15); hi = np.minimum(np.percentile(v[:, :2], 99, axis=0) + .5, 15)
        lo = np.minimum(lo, -1); hi = np.maximum(hi, 1)
        self.lo = lo; n = np.ceil((hi - lo) / CELL_M).astype(int); self.n = n
        xs = lo[0] + (np.arange(n[0]) + .5) * CELL_M; ys = lo[1] + (np.arange(n[1]) + .5) * CELL_M
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        rays = np.zeros((X.size, 6), np.float32)
        rays[:, 0] = X.ravel() / self.mpu; rays[:, 1] = Y.ravel() / self.mpu; rays[:, 5] = -1
        hit = self.rays.cast_rays(o3d.core.Tensor(rays))
        t = hit['t_hit'].numpy().reshape(X.shape) * self.mpu      # metres from eye level down
        nz = np.abs(hit['primitive_normals'].numpy()[:, 2]).reshape(X.shape)
        ij_all = np.floor((v[:, :2] - lo) / CELL_M).astype(int)
        ok = (ij_all[:, 0] >= 0) & (ij_all[:, 0] < n[0]) & (ij_all[:, 1] >= 0) & (ij_all[:, 1] < n[1])
        ij_all = ij_all[ok]; z_all = v[ok, 2]

        def occupancy(floor_ok, blocked_top, height):
            rel = z_all - height[ij_all[:, 0], ij_all[:, 1]]
            ij = ij_all[(rel > .15) & (rel < 1.75)]
            counts = np.zeros(X.shape, int); np.add.at(counts, (ij[:, 0], ij[:, 1]), 1)
            return (counts >= 3) | blocked_top
        # A (primary): level floor at the eye-height prior
        floor_ok = np.isfinite(t) & (t > EYE_M * .8) & (t < EYE_M * 1.25)
        blocked_top = np.isfinite(t) & (t <= EYE_M * .8) & (t > 0.0)
        occ = occupancy(floor_ok, blocked_top, np.full(X.shape, floor))
        # B (fallback): tilted/raised generated floors, obstacles relative to local floor height
        from scipy import ndimage
        floor_b = np.isfinite(t) & (t > EYE_M * .6) & (t < EYE_M * 1.5) & (nz > .85)
        hf = np.where(floor_b, -t, floor)
        if floor_b.any():
            _, idx = ndimage.distance_transform_edt(~floor_b, return_indices=True); hf = hf[idx[0], idx[1]]
        occ_b = occupancy(floor_b, np.isfinite(t) & (t <= EYE_M * .6) & (t > 0.0), hf)
        k = int(math.ceil(BODY_M / CELL_M)); ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        occ_d = cv2.dilate(occ.astype(np.uint8), ker) > 0
        free = floor_ok & ~occ_d
        s = self.cell([0, 0]); free[max(0, s[0] - k):s[0] + k + 1, max(0, s[1] - k):s[1] + k + 1] |= floor_ok[max(0, s[0] - k):s[0] + k + 1, max(0, s[1] - k):s[1] + k + 1]
        occ_bd = cv2.dilate(occ_b.astype(np.uint8), ker) > 0
        self.free_local = floor_b & ~occ_bd; self.free_local[max(0, s[0] - k):s[0] + k + 1, max(0, s[1] - k):s[1] + k + 1] = True
        self.free = free; self.occ = occ; self.floor_ok = floor_ok
        # relaxed: not blocked, floor may be a reconstruction hole (used only if strict planning fails)
        self.relaxed = ~occ_d; self.relaxed[max(0, s[0] - k):s[0] + k + 1, max(0, s[1] - k):s[1] + k + 1] = True
        self.clear_m = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * CELL_M
        self.grid = self.free

    def cell(self, p):
        return tuple(np.clip(np.floor((np.asarray(p[:2]) - self.lo) / CELL_M).astype(int), 0, self.n - 1))

    def center(self, c):
        return self.lo + (np.asarray(c) + .5) * CELL_M

    def is_free(self, p):
        return bool(self.grid[self.cell(p)])

    def segment_clear(self, a, b):
        a = np.asarray(a[:2], float); b = np.asarray(b[:2], float); d = np.linalg.norm(b - a)
        for s in np.linspace(0, 1, max(2, int(d / (CELL_M / 2)) + 1)):
            if not self.is_free(a + (b - a) * s):
                return False
        return True

    def move(self, x, y, yaw, dist):
        a = np.array([x, y], float); d = np.array([-math.cos(yaw), math.sin(yaw)])
        last = a; blocked = False
        for s in np.arange(CELL_M / 2, dist + 1e-9, CELL_M / 2):
            p = a + d * s
            if not self.is_free(p):
                blocked = True; break
            last = p
        return last, blocked

    def astar(self, start, goals):
        """goals: set of cells. Returns cell path or None."""
        s = self.cell(start); free = self.grid.copy(); free[s] = True
        h = lambda c: min(math.hypot(c[0] - g[0], c[1] - g[1]) for g in goals) if len(goals) < 50 else 0
        openq = [(0, 0, s)]; came = {s: None}; cost = {s: 0}
        nb = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while openq:
            _, g, c = heapq.heappop(openq)
            if c in goals:
                path = []
                while c is not None:
                    path.append(c); c = came[c]
                return path[::-1]
            if g > cost.get(c, 1e18):
                continue
            for dx, dy, w in nb:
                q = (c[0] + dx, c[1] + dy)
                if 0 <= q[0] < self.n[0] and 0 <= q[1] < self.n[1] and free[q]:
                    ng = g + w
                    if ng < cost.get(q, 1e18):
                        cost[q] = ng; came[q] = c; heapq.heappush(openq, (ng + h(q), ng, q))
        return None

    def smooth(self, pts):
        out = [pts[0]]; i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.segment_clear(pts[i], pts[j]):
                j -= 1
            out.append(pts[j]); i = j
        return out

    def goto(self, x, y, tx, ty, stop_m, band=0.25):
        best = None
        for mode, grid in (('strict', self.free), ('local_floor', self.free_local), ('relaxed', self.relaxed)):
            self.grid = grid
            try:
                r = self._goto(x, y, tx, ty, stop_m, band)
            finally:
                self.grid = self.free
            if r.get('waypoints'):
                r['grid'] = mode
                if best is None or r['reached_stop_m'] < best['reached_stop_m'] - 0.5:
                    best = r
                if r['reached_stop_m'] <= stop_m + 1.0:
                    break
        return best or {'waypoints': None, 'error': 'no reachable free cell near target'}

    def _goto(self, x, y, tx, ty, stop_m, band=0.25):
        start = np.array([x, y]); target = np.array([tx, ty])
        ci = np.argwhere(self.grid)
        centers = self.lo + (ci + .5) * CELL_M
        dist = np.linalg.norm(centers - target, axis=1)
        clear = self.clear_m[ci[:, 0], ci[:, 1]]
        for widen in [0, .25, .5, 1.0]:
            ring = (dist >= stop_m - band - widen) & (dist <= stop_m + band + widen)
            roomy = ring & (clear >= 0.5)
            for sel in (ci[roomy], ci[ring]):
                if len(sel):
                    path = self.astar(start, set(map(tuple, sel)))
                    if path:
                        pts = self.smooth([start] + [self.center(c) for c in path[1:]])
                        return {'waypoints': [list(map(float, p)) for p in pts], 'reached_stop_m': float(np.linalg.norm(pts[-1] - target)), 'widened': widen}
        reach = self.reachable(start)
        if reach is not None and len(reach):
            cen = self.lo + (reach + .5) * CELL_M; k = int(np.argmin(np.linalg.norm(cen - target, axis=1)))
            path = self.astar(start, {tuple(reach[k])})
            if path:
                pts = self.smooth([start] + [self.center(c) for c in path[1:]])
                return {'waypoints': [list(map(float, p)) for p in pts], 'reached_stop_m': float(np.linalg.norm(pts[-1] - target)), 'widened': 'nearest_reachable'}
        return {'waypoints': None, 'error': 'no reachable free cell near target'}

    def candidates(self, x, y, tx, ty, stop_m, band=0.3, sectors=8, keep=4):
        """Reachable stopping cells around a target, one per approach sector (shortest path within each).
        Returns up to `keep` candidates ordered by path length, each with smoothed waypoints."""
        from collections import deque
        start = np.array([x, y]); target = np.array([tx, ty]); out = []
        for grid in (self.free, self.free_local, self.relaxed):
            self.grid = grid
            try:
                s0 = self.cell(start); free = grid.copy(); free[s0] = True
                dist = np.full(free.shape, -1, np.int32); dist[s0] = 0; q = deque([s0])
                while q:
                    c = q.popleft()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        n = (c[0] + dx, c[1] + dy)
                        if 0 <= n[0] < self.n[0] and 0 <= n[1] < self.n[1] and free[n] and dist[n] < 0:
                            dist[n] = dist[c] + 1; q.append(n)
                ci = np.argwhere(dist >= 0); cen = self.lo + (ci + .5) * CELL_M
                d = np.linalg.norm(cen - target, axis=1)
                for widen in (0, .3, .7):
                    ring = (d >= stop_m - band - widen) & (d <= stop_m + band + widen)
                    if ring.sum():
                        break
                if not ring.sum():
                    continue
                ang = np.arctan2(cen[ring, 1] - ty, cen[ring, 0] - tx)
                sec = ((ang + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
                cells = ci[ring]; pd = dist[cells[:, 0], cells[:, 1]]
                clear = self.clear_m[cells[:, 0], cells[:, 1]]
                for k in range(sectors):
                    m = np.flatnonzero(sec == k)
                    if not len(m):
                        continue
                    roomy = m[clear[m] >= .5]
                    pick = (roomy if len(roomy) else m)[np.argmin(pd[roomy if len(roomy) else m])]
                    path = self.astar(start, {tuple(cells[pick])})
                    if path:
                        pts = self.smooth([start] + [self.center(c) for c in path[1:]])
                        out.append({'waypoints': [list(map(float, p)) for p in pts], 'path_m': float(pd[pick] * CELL_M),
                                    'stop_m': float(d[ring][pick]), 'sector': int(k), 'grid': 'strict' if grid is self.free else 'fallback'})
            finally:
                self.grid = self.free
            if out:
                break
        out.sort(key=lambda c: c['path_m'])
        return {'candidates': out[:keep]}

    def scored_candidates(self, x, y, tx, ty, stop_m, pts, keep=8):
        r = self.candidates(x, y, tx, ty, stop_m, keep=keep)
        P = np.asarray(pts, float); c = P[:, :2].mean(0) if len(P) else np.array([tx, ty])
        for cand in r['candidates']:
            ex, ey = cand['waypoints'][-1]
            yaw = math.atan2(c[1] - ey, -(c[0] - ex))
            cand['face_yaw'] = yaw
            cand.update(self.overlap(ex, ey, yaw, 0.0, pts))
        return r

    def reachable(self, start):
        s = self.cell(start); free = self.grid.copy(); free[s] = True
        seen = np.zeros_like(free); seen[s] = True; q = [s]
        while q:
            c = q.pop()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                n = (c[0] + dx, c[1] + dy)
                if 0 <= n[0] < self.n[0] and 0 <= n[1] < self.n[1] and free[n] and not seen[n]:
                    seen[n] = True; q.append(n)
        return np.argwhere(seen)

    # ---------- rendering ----------
    def pose(self, x, y, yaw, pitch, z_m=0.0):
        fwd = np.array([-math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
        cr = np.cross(fwd, [0, 0, 1]); cr /= np.linalg.norm(cr); cu = np.cross(cr, fwd)
        c = np.eye(4); c[:3, :3] = self.R @ np.column_stack([cr, -cu, fwd])
        c[:3, 3] = self.scale * (self.R @ self.u([x, y, z_m])) + self.t
        return c.astype(np.float32)

    def render(self, x, y, yaw, pitch=0.0, w=384, h=288, vfov=60.0):
        f = h / (2 * math.tan(math.radians(vfov) / 2))
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)
        rgb, dep, alpha = self.splat.render(self.pose(x, y, yaw, pitch), K, w, h)
        return np.uint8(np.clip(rgb, 0, 1) * 255 + .5)

    def locate(self, x, y, yaw, pitch, px, py, w=384, h=288, vfov=60.0):
        """Pixel (normalised 0..1) in a rendered view -> first mesh hit, metres."""
        f = h / (2 * math.tan(math.radians(vfov) / 2))
        d_cam = np.array([(px * w - w / 2) / f, (py * h - h / 2) / f, 1.0])
        fwd = np.array([-math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
        cr = np.cross(fwd, [0, 0, 1]); cr /= np.linalg.norm(cr); cu = np.cross(cr, fwd)
        d = np.column_stack([cr, -cu, fwd]) @ d_cam; d /= np.linalg.norm(d)
        o = self.u([x, y, 0])
        t = self.cast(np.concatenate([o, d])[None])[0]
        if not np.isfinite(t):
            return None
        p = (o + d * t) * self.mpu
        return [float(v) for v in p]

    def locate_box(self, x, y, yaw, pitch, box, w=384, h=288, vfov=60.0):
        """box = [ymin,xmin,ymax,xmax] normalised 0..1. Robust surface point of the boxed object."""
        y0, x0, y1, x1 = box; pts = []
        for py in np.linspace(y0 + .1 * (y1 - y0), y1 - .1 * (y1 - y0), 9):
            for px in np.linspace(x0 + .15 * (x1 - x0), x1 - .15 * (x1 - x0), 9):
                p = self.locate(x, y, yaw, pitch, px, py, w, h, vfov)
                if p is not None:
                    pts.append(p)
        if len(pts) < 5:
            return None
        P = np.array(pts); d = np.linalg.norm(P[:, :2] - [x, y], axis=1)
        # object surface = dense near cluster; drop far background seen around the object
        cut = np.percentile(d, 60); sel = P[d <= cut]
        c = np.median(sel, axis=0)
        return [float(v) for v in c]

    def target_points(self, x, y, yaw, pitch, box, w=384, h=288, vfov=60.0, grid=15):
        """Lift a target box (normalised [ymin,xmin,ymax,xmax]) seen from a pose into a 3D point set (metres):
        rays through the box interior, keeping the near surface cluster (the boxed object, not the background)."""
        y0, x0, y1, x1 = box; pts = []
        for py in np.linspace(y0 + .08 * (y1 - y0), y1 - .08 * (y1 - y0), grid):
            for px in np.linspace(x0 + .1 * (x1 - x0), x1 - .1 * (x1 - x0), grid):
                p = self.locate(x, y, yaw, pitch, px, py, w, h, vfov)
                if p is not None:
                    pts.append(p)
        if len(pts) < 8:
            return []
        P = np.array(pts); d = np.linalg.norm(P[:, :2] - [x, y], axis=1)
        P = P[d <= np.percentile(d, 60) * 1.15]
        return P.tolist()

    def overlap(self, x, y, yaw, pitch, pts, vfov=60.0, aspect=4 / 3):
        """Pixel-overlap style check of a target point set from a pose: visible fraction (in frustum and not occluded),
        horizontal offset of the visible part (deg, + = right), angular width (deg) and distance (m)."""
        P = np.asarray(pts, float)
        if not len(P):
            return {'visible_frac': 0.0}
        fwd = np.array([-math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
        cr = np.cross(fwd, [0, 0, 1]); cr /= np.linalg.norm(cr); cu = np.cross(cr, fwd)
        o = np.array([x, y, 0.0]); v = P - o
        zc = v @ fwd; xc = v @ cr; yc = v @ cu
        tv = math.tan(math.radians(vfov) / 2); th = tv * aspect
        # horizontal field of view as rendered; vertically allow a natural head tilt (+-60 deg) so low/high targets near you still count
        inf = (zc > .1) & (np.abs(xc / np.maximum(zc, 1e-6)) < th) & (np.abs(yc / np.maximum(zc, 1e-6)) < math.tan(math.radians(60)))
        vis = np.zeros(len(P), bool)
        if inf.any():
            idx = np.flatnonzero(inf); dist = np.linalg.norm(v[idx], axis=1)
            dirs = v[idx] / dist[:, None]
            rays = np.concatenate([np.broadcast_to(self.u(o), (len(idx), 3)), dirs], 1)
            t = self.cast(rays) * self.mpu
            vis[idx] = ~np.isfinite(t) | (t >= dist - 0.25)
        if not vis.any():
            return {'visible_frac': 0.0, 'in_frustum_frac': float(inf.mean())}
        ang = np.degrees(np.arctan2(xc[vis], zc[vis]))
        return {'visible_frac': float(vis.mean()), 'in_frustum_frac': float(inf.mean()), 'center_deg': float(np.median(ang)),
                'width_deg': float(np.percentile(ang, 95) - np.percentile(ang, 5)), 'distance_m': float(np.median(np.linalg.norm(v[vis][:, :2], axis=1)))}

    def topdown(self, path=None, pose=None, marks=None, size=512):
        # rows = x (forward=-X at top), cols = y (right=+Y to the right)
        img = np.full(self.free.shape + (3,), 255, np.uint8)
        img[self.floor_ok] = (225, 225, 225); img[self.occ] = (90, 90, 90); img[self.free & self.floor_ok] = (250, 250, 250)
        sc = size / max(self.n)
        img = cv2.resize(img, (int(self.n[1] * sc), int(self.n[0] * sc)), interpolation=cv2.INTER_NEAREST)
        P = lambda p: (int((p[1] - self.lo[1]) / CELL_M * sc), int((p[0] - self.lo[0]) / CELL_M * sc))
        if path:
            cv2.polylines(img, [np.array([P(p) for p in path], np.int32)], False, (0, 0, 0), 2)
        for m in marks or []:
            cv2.circle(img, P(m['p']), 9 if m.get('label') else 6, tuple(m.get('color', (220, 0, 0))), -1)
            if m.get('label'):
                cv2.putText(img, str(m['label']), (P(m['p'])[0] + 10, P(m['p'])[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 2)
        if pose:
            b = [pose[0] - .8 * math.cos(pose[2]), pose[1] + .8 * math.sin(pose[2])]
            cv2.arrowedLine(img, P(pose), P(b), (0, 0, 255), 2, tipLength=.3)
        return img


def png(img):
    ok, buf = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


def main():
    env = Env(sys.argv[1])
    print(json.dumps({'ready': True, 'm_per_unit': env.mpu, 'eye_units': env.eye}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line); t0 = time.time(); out = {'id': req.get('id')}
        try:
            c = req['cmd']
            if c == 'render':
                out['png'] = png(env.render(req['x'], req['y'], req['yaw'], req.get('pitch', 0.), req.get('w', 384), req.get('h', 288)))
            elif c == 'locate':
                out['point'] = env.locate(req['x'], req['y'], req['yaw'], req.get('pitch', 0.), req['px'], req['py'], req.get('w', 384), req.get('h', 288))
            elif c == 'locate_box':
                out['point'] = env.locate_box(req['x'], req['y'], req['yaw'], req.get('pitch', 0.), req['box'], req.get('w', 384), req.get('h', 288))
            elif c == 'scored_candidates':
                out.update(env.scored_candidates(req['x'], req['y'], req['tx'], req['ty'], req['stop_m'], req['pts'], keep=req.get('keep', 8)))
            elif c == 'candidates':
                out.update(env.candidates(req['x'], req['y'], req['tx'], req['ty'], req['stop_m'], keep=req.get('keep', 4)))
            elif c == 'target_points':
                out['points'] = env.target_points(req['x'], req['y'], req['yaw'], req.get('pitch', 0.), req['box'])
            elif c == 'overlap':
                out.update(env.overlap(req['x'], req['y'], req['yaw'], req.get('pitch', 0.), req['pts']))
            elif c == 'move':
                p, b = env.move(req['x'], req['y'], req['yaw'], req['dist']); out.update(end=list(map(float, p)), blocked=b)
            elif c == 'goto':
                out.update(env.goto(req['x'], req['y'], req['tx'], req['ty'], req['stop_m']))
            elif c == 'free':
                out['free'] = env.is_free([req['x'], req['y']])
            elif c == 'topdown':
                out['png'] = png(env.topdown(req.get('path'), req.get('pose'), req.get('marks')))
            elif c == 'clear':
                out['clear'] = env.segment_clear(req['a'], req['b'])
            else:
                raise ValueError('unknown cmd')
        except Exception as exc:
            out['error'] = repr(exc)
        out['seconds'] = time.time() - t0
        print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
