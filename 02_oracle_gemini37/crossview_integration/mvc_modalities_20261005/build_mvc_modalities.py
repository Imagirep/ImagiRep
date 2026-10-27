"""Oracle-anchored modalities for one MVC (multi-view) resource, matching the current low-overlap images.

Single-frame modalities use the ORACLE camera (same viewpoint, oracle heading, -15 deg pitch, 1280x960,
HFOV 96.418 / VFOV 80, the exact native_pose_erp_v1 camera of the current images):
  oracle_rgb.png              current labeler oracle (copied, unchanged)
  oracle_depth.png / _raw     z-depth rendered from the official MP3D textured mesh at the oracle camera
                              (uint16 = metres*4000, 0 = no surface; colour = reversed TURBO p02/p98)
  oracle_surface_normals.png  mesh geometric normals in oracle-camera axes (R=right, G=up, B=toward camera)
  oracle_panorama.png         native skybox -> ERP, rolled so the oracle heading is the image centre
Multi-frame modalities describe Image 1 -> Oracle -> Image 2 (pure rotation at one point):
  view_transition.txt / .mp4  12 fps, native ERP perspectives, smoothstep yaw, 1 s holds at each image
  topdown_geometry.png        current labeler top-down (View 1 / View 2), copied unchanged
Alignment check: mesh-texture RGB at the oracle camera vs the official oracle image (edge correlation).
No model calls; source assets read-only. Usage: python build_mvc_modalities.py MVC-217
"""
import glob, hashlib, json, math, shutil, subprocess, sys
from pathlib import Path
import numpy as np, cv2, trimesh, moderngl, py360convert
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'outputs/mvc_panorama_farthest_ordering300_20260927/scripts'))
import panorama_render as pr  # noqa: E402  (native_pose_erp_v1 camera model)

W, H = 1280, 960
TX, TY = math.tan(math.radians(pr.HFOV) / 2), math.tan(math.radians(pr.VFOV) / 2)
LOW = ROOT / 'outputs/mvc_labeler_low_overlap_panorama_20260929'
MESHES = ROOT / 'outputs/mp3d_dollhouse_20261005'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mesh_render(scan, cam_pos, basis):
    """Textured-mesh RGB, camera z-depth (m) and camera-frame normals at a native_pose_erp camera."""
    obj = glob.glob(str(MESHES / scan / scan / 'matterport_mesh' / '*' / '*.obj'))
    if not obj:
        raise FileNotFoundError(f'download {scan}/matterport_mesh.zip into {MESHES / scan}')
    scene = trimesh.load(obj[0], process=False)
    ctx = moderngl.create_standalone_context()
    prog = ctx.program(vertex_shader='''#version 330
        uniform mat4 view; uniform mat4 proj; in vec3 pos; in vec2 uv; out vec2 v_uv; out vec3 v_cam;
        void main(){ vec4 c = view * vec4(pos, 1.0); v_cam = c.xyz; v_uv = uv; gl_Position = proj * c; }''',
        fragment_shader='''#version 330
        uniform sampler2D tex; in vec2 v_uv; in vec3 v_cam;
        layout(location=0) out vec4 color; layout(location=1) out vec4 geo;
        void main(){ vec3 n = normalize(cross(dFdx(v_cam), dFdy(v_cam)));
                     if (dot(n, v_cam) > 0.0) n = -n;          // face the camera
                     color = vec4(texture(tex, v_uv).rgb, 1.0); geo = vec4(n, -v_cam.z); }''')
    # GL camera: x right, y up, looks down -z.  Physical camera (pr.project_object): x right, y up, z forward.
    B = -pr.world_basis(*CAM)          # columns = camera right/up/forward in world
    view = np.eye(4); view[:3, :3] = np.diag([1, 1, -1]) @ B.T; view[:3, 3] = -view[:3, :3] @ np.asarray(cam_pos)
    n, f = 0.05, 60.0
    proj = np.array([[1 / TX, 0, 0, 0], [0, 1 / TY, 0, 0], [0, 0, (f + n) / (n - f), 2 * f * n / (n - f)], [0, 0, -1, 0]])
    prog['view'].write(view.T.astype('f4').tobytes()); prog['proj'].write(proj.T.astype('f4').tobytes())
    draws = []
    for g in (scene.geometry.values() if isinstance(scene, trimesh.Scene) else [scene]):
        if getattr(g.visual, 'uv', None) is None or getattr(g.visual.material, 'image', None) is None:
            continue
        idx = g.faces.reshape(-1)
        buf = ctx.buffer(np.hstack([np.asarray(g.vertices, 'f4')[idx], np.asarray(g.visual.uv, 'f4')[idx]]).tobytes())
        img = g.visual.material.image.convert('RGB').transpose(Image.FLIP_TOP_BOTTOM)
        tex = ctx.texture(img.size, 3, img.tobytes()); tex.build_mipmaps(); tex.anisotropy = 16.0
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        draws.append((ctx.vertex_array(prog, [(buf, '3f 2f', 'pos', 'uv')]), tex))
    col, geo = ctx.texture((W, H), 4), ctx.texture((W, H), 4, dtype='f4')
    fbo = ctx.framebuffer([col, geo], ctx.depth_renderbuffer((W, H)))
    fbo.use(); ctx.enable(moderngl.DEPTH_TEST); fbo.clear(0, 0, 0, 0)
    for vao, tex in draws:
        tex.use(0); prog['tex'].value = 0; vao.render(moderngl.TRIANGLES)
    rgb = np.frombuffer(col.read(), 'u1').reshape(H, W, 4)[::-1, :, :3]
    g = np.frombuffer(geo.read(), 'f4').reshape(H, W, 4)[::-1]
    return rgb.copy(), g[..., 3].copy(), g[..., :3].copy()


def depth_png(z):
    valid = z > 0
    raw = np.where(valid, np.clip(np.round(z * 4000), 0, 65535), 0).astype(np.uint16)
    near, far = (np.percentile(z[valid], [2, 98]) if valid.any() else (0, 1))
    t = np.clip((z - near) / max(far - near, 1e-6), 0, 1)
    color = cv2.applyColorMap(np.uint8(255 - t * 255), cv2.COLORMAP_TURBO); color[~valid] = 0
    return raw, color, float(near), float(far), float(valid.mean())


def main():
    global CAM
    rid = sys.argv[1]
    man = json.loads((LOW / rid / 'render_manifest.json').read_text())
    scan, vp = man['job']['scan_id'], man['job']['viewpoint_id']
    pose, _ = pr.load_pose(scan, vp)
    heads = {v['image_number']: v['world_heading_degrees'] for v in man['views']}
    oh, op = man['oracle_heading_degrees'], man['oracle_pitch_degrees']
    out = HERE / rid; out.mkdir(parents=True, exist_ok=True)
    for name, src in [('image_1.png', LOW / rid / 'image_1.png'), ('image_2.png', LOW / rid / 'image_2.png'),
                      ('oracle_rgb.png', LOW / rid / 'oracle.png'), ('topdown_geometry.png', LOW / rid / 'topdown_geometry.png')]:
        shutil.copy2(src, out / name)
    erp, *_ = pr.panorama(scan, vp)
    # oracle reproduction from native ERP must equal the labeler oracle (camera-model check)
    rep = pr.extract(erp, pose, oh, op, W, H)[0]
    official = cv2.cvtColor(cv2.imread(str(LOW / rid / 'oracle.png')), cv2.COLOR_BGR2RGB)
    rep_mae = float(np.abs(rep.astype(float) - official.astype(float)).mean())
    # ---- depth / normals at the oracle camera from the MP3D textured mesh
    CAM = (oh, op)
    mrgb, z, nrm = mesh_render(scan, man['camera_position'], None)
    e1 = cv2.Canny(cv2.cvtColor(official, cv2.COLOR_RGB2GRAY), 60, 150).astype(float)
    e2 = cv2.Canny(cv2.cvtColor(mrgb, cv2.COLOR_RGB2GRAY), 60, 150).astype(float)
    k = np.ones((7, 7)); b1, b2 = cv2.filter2D(e1, -1, k), cv2.filter2D(e2, -1, k)
    edge_corr = float(np.corrcoef(b1.ravel(), b2.ravel())[0, 1])
    raw, dcol, near, far, valid = depth_png(z)
    cv2.imwrite(str(out / 'oracle_depth_raw_uint16.png'), raw); cv2.imwrite(str(out / 'oracle_depth.png'), dcol)
    nimg = np.uint8(np.clip((nrm + 1) / 2 * 255 + .5, 0, 255)); nimg[z <= 0] = 0
    cv2.imwrite(str(out / 'oracle_surface_normals.png'), cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / 'mesh_alignment_check.png'), cv2.cvtColor(np.hstack([official, mrgb]), cv2.COLOR_RGB2BGR))
    # ---- oracle-centred panorama: roll ERP so the oracle forward ray sits at the centre column
    fwd = pr.local_basis(pose, oh, op) @ np.array([0, 0, 1.0])     # oracle optical axis in ERP (pano) coordinates
    lon = math.atan2(fwd[0], fwd[2])
    col = ((lon / (2 * math.pi) + .5) * pr.ERP_W - .5) % pr.ERP_W   # same mapping as pr.extract()
    pano = np.roll(erp, int(round(pr.ERP_W / 2 - .5 - col)), axis=1)
    pano_small = cv2.resize(pano, (2048, 1024), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / 'oracle_panorama.png'), cv2.cvtColor(pano_small, cv2.COLOR_RGB2BGR))
    # verify centring: the centre columns of the panorama must match a level view at the oracle heading
    lvl = pr.extract(erp, pose, oh, 0.0, 400, 400)[0]
    ctr = pano[pr.ERP_H // 2 - 200:pr.ERP_H // 2 + 200, pr.ERP_W // 2 - 200:pr.ERP_W // 2 + 200]
    g = lambda im: cv2.cvtColor(cv2.resize(im, (100, 100)), cv2.COLOR_RGB2GRAY).ravel().astype(float)
    pano_corr = float(np.corrcoef(g(lvl), g(ctr))[0, 1])
    # ---- transition text + video: Image 1 -> Oracle -> Image 2
    def turn(a, b):
        d = (b - a + 180) % 360 - 180
        return ('right' if d > 0 else 'left'), abs(d)
    t1, a1 = turn(heads[1], oh); t2, a2 = turn(oh, heads[2])
    text = ("All images were taken from the same position; there is no movement, only rotation in place.\n"
            f"Your view is tilted {abs(op):.1f}° below level throughout.\n\n"
            f"Start facing as in Image 1. Turn {t1} {a1:.1f}° in place to the oracle view (the view between the two images).\n"
            f"Then turn {t2} {a2:.1f}° further in place to face as in Image 2.\n\n"
            f"In total, Image 2 is {a1 + a2:.1f}° to the {t2 if t1 == t2 else 'side'} of Image 1.\n")
    (out / 'view_transition.txt').write_text(text, encoding='utf-8')
    fps, hold = 12, 12
    seq = [heads[1]] * hold
    for a, b in [(heads[1], oh), (oh, heads[2])]:
        d = (b - a + 180) % 360 - 180; n = max(12, int(math.ceil(abs(d) / 3)))
        seq += [a + d * (3 * s * s - 2 * s ** 3) for s in np.linspace(0, 1, n + 1)[1:]] + ([b] * hold if b == oh else [])
    seq += [heads[2]] * hold
    fr = out / '_frames'; fr.mkdir(exist_ok=True)
    for i, h in enumerate(seq):
        cv2.imwrite(str(fr / f'{i:04d}.jpg'), cv2.cvtColor(pr.extract(erp, pose, h, op, W, H)[0], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', str(fr / '%04d.jpg'), '-c:v', 'libx264', '-crf', '20',
                    '-pix_fmt', 'yuv420p', str(out / 'view_transition.mp4')], check=True)
    shutil.rmtree(fr)
    first = cv2.cvtColor(pr.extract(erp, pose, heads[1], op, W, H)[0], cv2.COLOR_RGB2GRAY).astype(float)
    img1 = cv2.cvtColor(cv2.imread(str(out / 'image_1.png')), cv2.COLOR_BGR2GRAY).astype(float)
    meta = dict(resource_id=rid, scan_id=scan, viewpoint_id=vp, camera_position=man['camera_position'],
                image_headings_degrees=heads, oracle_heading_degrees=oh, pitch_degrees=op, resolution=[W, H], hfov_degrees=pr.HFOV, vfov_degrees=pr.VFOV,
                oracle_rgb=dict(source=str((LOW / rid / 'oracle.png').relative_to(ROOT)), sha256=sha(out / 'oracle_rgb.png'), native_erp_reproduction_mae=rep_mae),
                depth=dict(source='official MP3D textured mesh (matterport_mesh) rendered at the oracle camera', encoding='uint16 = metres*4000, 0 = no surface',
                           color='reversed TURBO, red/near - blue/far, black = no surface', near_m=round(near, 3), far_m=round(far, 3), valid_fraction=valid,
                           mesh_vs_oracle_edge_correlation=edge_corr),
                surface_normals=dict(source='same mesh render, geometric face normals', axes='R=camera right, G=camera up, B=toward camera; -1..1 -> 0..255'),
                panorama=dict(source='native MP3D skybox (sami) -> 4096x2048 ERP -> 2048x1024', centre='oracle heading at image centre; left quarter = 90 deg left',
                              centre_vs_oracle_level_view_correlation=pano_corr),
                view_transition=dict(turns=[[t1, a1], [t2, a2]], fps=fps, frames=len(seq), holds_seconds=hold / fps,
                                     first_frame_vs_image1_mae=float(np.abs(first - img1).mean())),
                topdown=dict(source='current labeler top-down (View 1 / View 2 arrows), unchanged'), model_calls=0)
    (out / 'modalities_manifest.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: meta[k] for k in ('oracle_rgb', 'depth', 'panorama', 'view_transition')}, indent=1))


if __name__ == '__main__':
    main()
