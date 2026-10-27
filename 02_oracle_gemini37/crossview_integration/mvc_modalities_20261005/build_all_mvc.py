"""Oracle-anchored modalities for every MVC resource of the frozen multi-view benchmark (Ready100 + Last50).

Same standard as the MVC-217 prototype (build_mvc_modalities.py):
  single-frame modalities at the ORACLE camera (1280x960, HFOV 96.418, VFOV 80, native_pose_erp_v1 model)
    oracle_rgb.png                 labeler oracle, copied unchanged (native ERP re-render must reproduce it)
    oracle_depth.png / _raw_uint16 z-depth from the official MP3D textured mesh (uint16 = m*4000, 0 = none)
    oracle_surface_normals.png     mesh normals in oracle-camera axes (R right, G up, B toward camera)
    oracle_panorama.png            native skybox ERP (2048x1024), oracle heading at the image centre
  multi-frame modalities
    view_transition.txt / .mp4     in-place rotation through the observed views; two-view sets pass through the
                                   oracle heading (it lies between them); three-view sets rotate Image 1->2->3 and the
                                   text states where the separately captured oracle view was taken
    topdown_geometry.png           labeler top-down, copied unchanged
Three schemas: low-overlap two-view (MVC-*), MVP-* candidates, three-view (MVC3I-*, oracle at another viewpoint).
Usage: python build_all_mvc.py [RID ...]   (default: all resources in scope.json; existing outputs are skipped)
No model calls; source assets read-only.
"""
import glob, hashlib, json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, trimesh, moderngl
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'outputs/mvc_panorama_farthest_ordering300_20260927/scripts'))
import panorama_render as pr  # noqa: E402

W, H = 1280, 960
TX, TY = math.tan(math.radians(pr.HFOV) / 2), math.tan(math.radians(pr.VFOV) / 2)
LOW = ROOT / 'outputs/mvc_labeler_low_overlap_panorama_20260929'
MVP = ROOT / 'outputs/mvc_panorama_farthest_ordering300_20260927/candidates'
MESHES = ROOT / 'outputs/mp3d_dollhouse_20261005'
OUT = HERE / 'resources'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def resources():
    res = {}
    for d in ['outputs/mvc_ready100_three_settings_20260930', 'outputs/mvc_batch1_last50_scenes_three_settings_20261001']:
        R = json.load(open(ROOT / d / 'resources.json', encoding='utf-8')); R = R['items'] if isinstance(R, dict) and 'items' in R else R
        for x in (R if isinstance(R, list) else R.values()):
            res[x['id']] = x
    return res


def spec(rid, r):
    """Camera specification for one resource (observation point, views, oracle)."""
    img = {i['label']: ROOT / i['path'] for i in r['images']}
    if (LOW / rid / 'render_manifest.json').exists():
        m = json.loads((LOW / rid / 'render_manifest.json').read_text())
        views = [(v['image_number'], v['world_heading_degrees'], v['world_pitch_degrees'], ROOT / v['path']) for v in m['views']]
        if m.get('oracle_viewpoint_id'):          # three-view: oracle kept at its original capture point
            ovp = m['oracle_viewpoint_id']; opose = np.asarray(m['oracle_pose_row_major'], float)
        else:
            ovp = m['job']['viewpoint_id']; opose = np.asarray(m['pose_row_major'], float)
        return dict(scan=m['job']['scan_id'], vp=m['job']['viewpoint_id'], pose=np.asarray(m['pose_row_major'], float), views=views,
                    ovp=ovp, opose=opose, oh=m['oracle_heading_degrees'], op=m['oracle_pitch_degrees'],
                    oracle=img['Oracle RGB'], topdown=img[next(k for k in img if k.startswith('Top-down'))], schema='low_overlap')  # labeler-linked files
    m = json.loads((MVP / rid / 'render_manifest.json').read_text())
    v = {x['view']: x for x in m['views']}
    pose = np.asarray(m['pose_row_major'], float)
    # resource Image 1 / Image 2 paths decide which physical view is which
    by_file = {x['filename']: x for x in m['views']}
    views = []
    for n, lab in ((1, 'Image 1'), (2, 'Image 2')):
        x = by_file[img[lab].name]; views.append((n, x['world_heading_degrees'], x['world_pitch_degrees'], img[lab]))
    return dict(scan=r['scan_id'], vp=r['viewpoint_id'], pose=pose, views=views, ovp=r['viewpoint_id'], opose=pose,
                oh=v['oracle']['world_heading_degrees'], op=v['oracle']['world_pitch_degrees'], oracle=img['Oracle RGB'],
                topdown=img[next(k for k in img if k.startswith('Top-down'))], schema='mvp_candidate')


class MeshRenderer:
    """Textured MP3D mesh, loaded once per scan."""
    def __init__(self, scan, center=None, radius=25.0):
        """center: optional camera position; then only mesh parts within `radius` m are uploaded (very large scans
        otherwise exceed GPU texture memory and render black)."""
        obj = glob.glob(str(MESHES / scan / scan / 'matterport_mesh' / '*' / '*.obj'))
        if not obj:
            raise FileNotFoundError(f'mesh for {scan} missing')
        scene = trimesh.load(obj[0], process=False)
        self.ctx = ctx = moderngl.create_standalone_context()
        self.prog = ctx.program(vertex_shader='''#version 330
            uniform mat4 view; uniform mat4 proj; in vec3 pos; in vec2 uv; out vec2 v_uv; out vec3 v_cam;
            void main(){ vec4 c = view * vec4(pos, 1.0); v_cam = c.xyz; v_uv = uv; gl_Position = proj * c; }''',
            fragment_shader='''#version 330
            uniform sampler2D tex; in vec2 v_uv; in vec3 v_cam;
            layout(location=0) out vec4 color; layout(location=1) out vec4 geo;
            void main(){ vec3 n = normalize(cross(dFdx(v_cam), dFdy(v_cam)));
                         if (dot(n, v_cam) > 0.0) n = -n;
                         color = vec4(texture(tex, v_uv).rgb, 1.0); geo = vec4(n, -v_cam.z); }''')
        self.draws = []
        for g in (scene.geometry.values() if isinstance(scene, trimesh.Scene) else [scene]):
            if getattr(g.visual, 'uv', None) is None or getattr(g.visual.material, 'image', None) is None:
                continue
            if center is not None:
                lo, hi = g.vertices.min(0), g.vertices.max(0)
                if np.linalg.norm(np.clip(np.asarray(center), lo, hi) - np.asarray(center)) > radius:
                    continue
            idx = g.faces.reshape(-1)
            buf = ctx.buffer(np.hstack([np.asarray(g.vertices, 'f4')[idx], np.asarray(g.visual.uv, 'f4')[idx]]).tobytes())
            im = g.visual.material.image.convert('RGB').transpose(Image.FLIP_TOP_BOTTOM)
            tex = ctx.texture(im.size, 3, im.tobytes()); tex.build_mipmaps(); tex.anisotropy = 16.0
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self.draws.append((ctx.vertex_array(self.prog, [(buf, '3f 2f', 'pos', 'uv')]), tex))
        self.col, self.geo = ctx.texture((W, H), 4), ctx.texture((W, H), 4, dtype='f4')
        self.fbo = ctx.framebuffer([self.col, self.geo], ctx.depth_renderbuffer((W, H)))

    def render(self, cam_pos, heading, pitch):
        B = -pr.world_basis(heading, pitch)            # columns: camera right / up / forward in world
        view = np.eye(4); view[:3, :3] = np.diag([1, 1, -1]) @ B.T; view[:3, 3] = -view[:3, :3] @ np.asarray(cam_pos)
        n, f = 0.05, 60.0
        proj = np.array([[1 / TX, 0, 0, 0], [0, 1 / TY, 0, 0], [0, 0, (f + n) / (n - f), 2 * f * n / (n - f)], [0, 0, -1, 0]])
        self.prog['view'].write(view.T.astype('f4').tobytes()); self.prog['proj'].write(proj.T.astype('f4').tobytes())
        self.fbo.use(); self.ctx.enable(moderngl.DEPTH_TEST); self.fbo.clear(0, 0, 0, 0)
        for vao, tex in self.draws:
            tex.use(0); self.prog['tex'].value = 0; vao.render(moderngl.TRIANGLES)
        rgb = np.frombuffer(self.col.read(), 'u1').reshape(H, W, 4)[::-1, :, :3]
        g = np.frombuffer(self.geo.read(), 'f4').reshape(H, W, 4)[::-1]
        return rgb.copy(), g[..., 3].copy(), g[..., :3].copy()

    def release(self):
        self.ctx.release()


def depth_png(z):
    valid = z > 0
    raw = np.where(valid, np.clip(np.round(z * 4000), 0, 65535), 0).astype(np.uint16)
    near, far = (np.percentile(z[valid], [2, 98]) if valid.any() else (0, 1))
    t = np.clip((z - near) / max(far - near, 1e-6), 0, 1)
    color = cv2.applyColorMap(np.uint8(255 - t * 255), cv2.COLORMAP_TURBO); color[~valid] = 0
    return raw, color, float(near), float(far), float(valid.mean())


def edge_corr(a, b):
    e = [cv2.filter2D(cv2.Canny(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), 60, 150).astype(float), -1, np.ones((7, 7))) for x in (a, b)]
    return float(np.corrcoef(e[0].ravel(), e[1].ravel())[0, 1])


def turn(a, b):
    d = (b - a + 180) % 360 - 180
    return ('right' if d > 0 else 'left'), abs(d), d


def build(rid, r, mesh):
    s = spec(rid, r); out = OUT / rid
    if (out / 'modalities_manifest.json').exists():
        return 'skip'
    out.mkdir(parents=True, exist_ok=True)
    for n, *_rest, p in s['views']:
        shutil.copy2(p, out / f'image_{n}.png')
    shutil.copy2(s['oracle'], out / 'oracle_rgb.png'); shutil.copy2(s['topdown'], out / 'topdown_geometry.png')
    oerp, *_ = pr.panorama(s['scan'], s['ovp'])
    official = cv2.cvtColor(cv2.imread(str(s['oracle'])), cv2.COLOR_BGR2RGB)
    rep_mae = float(np.abs(pr.extract(oerp, s['opose'], s['oh'], s['op'], W, H)[0].astype(float) - official.astype(float)).mean())
    # depth / normals at the oracle camera
    mrgb, z, nrm = mesh.render(s['opose'][:3, 3], s['oh'], s['op'])
    raw, dcol, near, far, valid = depth_png(z)
    cv2.imwrite(str(out / 'oracle_depth_raw_uint16.png'), raw); cv2.imwrite(str(out / 'oracle_depth.png'), dcol)
    nimg = np.uint8(np.clip((nrm + 1) / 2 * 255 + .5, 0, 255)); nimg[z <= 0] = 0
    cv2.imwrite(str(out / 'oracle_surface_normals.png'), cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / 'mesh_alignment_check.jpg'), cv2.cvtColor(cv2.resize(np.hstack([official, mrgb]), (1280, 480)), cv2.COLOR_RGB2BGR))
    # oracle-centred panorama
    fwd = pr.local_basis(s['opose'], s['oh'], s['op']) @ np.array([0, 0, 1.0])
    col = ((math.atan2(fwd[0], fwd[2]) / (2 * math.pi) + .5) * pr.ERP_W - .5) % pr.ERP_W
    pano = np.roll(oerp, int(round(pr.ERP_W / 2 - .5 - col)), axis=1)
    cv2.imwrite(str(out / 'oracle_panorama.png'), cv2.cvtColor(cv2.resize(pano, (2048, 1024), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
    # transition text + video
    heads = {n: h for n, h, _, _ in s['views']}; pitch = s['views'][0][2]
    two = len(s['views']) == 2
    if two:
        stops = [(heads[1], 'Image 1'), (s['oh'], 'the oracle view (between the two images)'), (heads[2], 'Image 2')]
    else:
        stops = [(heads[n], f'Image {n}') for n in sorted(heads)]
    lines = ['All observed images were taken from the same position; between them there is no movement, only rotation in place.',
             f'Your view is tilted {abs(pitch):.1f}° below level throughout.', '', f'Start facing as in {stops[0][1]}.']
    total = 0.0
    for (a, _), (b, name) in zip(stops, stops[1:]):
        t, ang, d = turn(a, b); total += d
        lines.append(f'Turn {t} {ang:.1f}° in place to face as in {name}.')
    if not two:
        dp = s['opose'][:3, 3] - s['pose'][:3, 3]; dist = float(np.linalg.norm(dp[:2]))
        bearing = math.degrees(math.atan2(float(dp[0]), float(dp[1])))            # MatterSim heading convention
        t, ang, _ = turn(heads[1], bearing)
        lines += ['', f'The oracle view was taken from a different position, {dist:.2f} m away, located {ang:.1f}° to the {t} of the Image 1 viewing direction.']
    (out / 'view_transition.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    erp, *_ = pr.panorama(s['scan'], s['vp'])
    seq = [stops[0][0]] * 12
    for (a, _), (b, _) in zip(stops, stops[1:]):
        d = (b - a + 180) % 360 - 180; n = max(12, int(math.ceil(abs(d) / 3)))
        seq += [a + d * (3 * x * x - 2 * x ** 3) for x in np.linspace(0, 1, n + 1)[1:]] + [b] * 12
    fr = out / '_frames'; fr.mkdir(exist_ok=True)
    for i, h in enumerate(seq):
        cv2.imwrite(str(fr / f'{i:04d}.jpg'), cv2.cvtColor(pr.extract(erp, s['pose'], h, pitch, W, H)[0], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '12', '-i', str(fr / '%04d.jpg'), '-c:v', 'libx264', '-crf', '20',
                    '-pix_fmt', 'yuv420p', str(out / 'view_transition.mp4')], check=True)
    shutil.rmtree(fr)
    first = pr.extract(erp, s['pose'], heads[1], pitch, W, H)[0]
    img1 = cv2.cvtColor(cv2.imread(str(out / 'image_1.png')), cv2.COLOR_BGR2RGB)
    meta = dict(resource_id=rid, schema=s['schema'], scan_id=s['scan'], observation_viewpoint=s['vp'], oracle_viewpoint=s['ovp'],
                image_headings_degrees=heads, pitch_degrees=pitch, oracle_heading_degrees=s['oh'], oracle_pitch_degrees=s['op'],
                resolution=[W, H], hfov_degrees=pr.HFOV, vfov_degrees=pr.VFOV,
                checks=dict(oracle_native_reproduction_mae=rep_mae, image1_vs_first_video_frame_mae=float(np.abs(first.astype(float) - img1.astype(float)).mean()),
                            mesh_vs_oracle_edge_correlation=edge_corr(official, mrgb)),
                depth=dict(source='official MP3D textured mesh at the oracle camera', near_m=round(near, 3), far_m=round(far, 3), valid_fraction=valid,
                           encoding='raw uint16 = metres*4000, 0 = no surface; colour reversed TURBO (red near, blue far, black none)'),
                surface_normals='mesh normals in oracle-camera axes: R right, G up, B toward camera; -1..1 -> 0..255',
                panorama='native skybox ERP at the oracle capture point, 2048x1024, oracle heading at the centre',
                view_transition=dict(stops=[n for _, n in stops], frames=len(seq), fps=12),
                topdown='labeler top-down, unchanged', model_calls=0)
    (out / 'modalities_manifest.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    return meta['checks']


def main():
    scope = json.load(open(HERE / 'scope.json'))['resources']
    ids = [a for a in sys.argv[1:] if not a.startswith('--')] or list(scope)
    res = resources(); by_scan = {}
    for rid in ids:
        by_scan.setdefault(res[rid]['scan_id'], []).append(rid)
    log = open(HERE / 'build_all.log', 'a', encoding='utf-8')
    for scan, rids in by_scan.items():
        if all((OUT / r / 'modalities_manifest.json').exists() for r in rids):
            continue
        t0 = time.time()
        local = len(rids) == 1 and '--local' in sys.argv
        mesh = MeshRenderer(scan, center=spec(rids[0], res[rids[0]])['opose'][:3, 3] if local else None)
        for rid in rids:
            try:
                c = build(rid, res[rid], mesh)
                msg = f'{rid} {scan} {c}'
            except Exception as e:
                msg = f'{rid} {scan} FAILED {type(e).__name__}: {e}'
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
        mesh.release(); print(f'scan {scan} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
