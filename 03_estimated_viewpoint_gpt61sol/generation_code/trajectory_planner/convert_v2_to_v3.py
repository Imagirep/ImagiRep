"""Convert estimated packages v2 -> v3: identical camera poses/renders, metric quantities rescaled with the
per-scene MoGe scale (initial image only). Regenerates Text.txt with the renderer's own template function and
updates the depth colour scale in metres. No model calls, no rendering. Skips scenes already present in v3."""
import ast, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V2 = REPO / 'outputs/hy_estimated_packages_20261002/v2'
V3 = REPO / 'outputs/hy_estimated_packages_20261002/v3'
PLANS = REPO / 'outputs/hy_nav_pilot_20261002/v2'
SCALE = REPO / 'outputs/hy_nav_pilot_20261002/scale'

# Load only the pure text-template functions from render_package.py (it imports server-only modules).
src = (HERE / 'render_package.py').read_text(encoding='utf-8')
ns = {'math': __import__('math')}
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ('wrap', 'pitch_state', 'rotation', 'facing', 'text'):
        exec(compile(ast.Module([node], []), 'render_package', 'exec'), ns)


def export_scales(sids):
    sys.path.insert(0, str(HERE)); import run_scene
    sh = run_scene.Shell(); out = {}
    try:
        for s in sids:
            out[s] = json.loads(sh.run(f'cat {run_scene.LATEST}{s}/result/ply/position_meta_info.json'))['scale']
    finally:
        sh.close()
    return out


def main():
    scenes = sorted(d.name for d in V2.iterdir() if d.is_dir() and (d / 'pipeline_status.json').exists() and not (V3 / d.name).exists())
    need_export = []
    for sid in scenes:
        for pk in (V2 / sid / 'Estimated_Modalities').iterdir():
            if 'depth_color_scale_m' not in json.loads((pk / 'manifest.json').read_text()):
                need_export.append(sid); break
    exp = export_scales(need_export) if need_export else {}
    done = 0
    for sid in scenes:
        sc = json.loads((SCALE / f'{sid}.json').read_text())
        st = json.loads((V2 / sid / 'pipeline_status.json').read_text())
        tmp = V3 / (sid + '.tmp'); shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(V2 / sid, tmp)
        for name, lead in st['packages'].items():
            plan = json.loads((PLANS / sid / lead / 'plan.json').read_text())
            f = sc['m_per_unit_moge'] / plan['m_per_unit']
            kfs = [dict(k, x=k['x'] * f, y=k['y'] * f) for k in plan['keyframes']]
            txt, total = ns['text'](kfs, name.startswith('Lookback'))
            d = tmp / 'Estimated_Modalities' / name
            (d / 'Text.txt').write_text(txt, encoding='utf-8')
            man = json.loads((d / 'manifest.json').read_text())
            if 'depth_color_scale_m' in man:
                cs = {k: v * f for k, v in man['depth_color_scale_m'].items()}
            else:
                lo, hi = man['depth_display_range_scene_units']; k = sc['m_per_unit_moge'] / exp[sid]
                cs = {'near_m': lo * k, 'far_m': hi * k}
            man.update(depth_color_scale_m=cs, total_distance_m=total, m_per_unit=sc['m_per_unit_moge'], scale_source='MoGe-2 metric depth of the initial image vs generated start-view depth',
                       planning_m_per_unit=plan['m_per_unit'], scale_factor_applied=f, converted_from='v2')
            (d / 'manifest.json').write_text(json.dumps(man, indent=2))
        tmp.rename(V3 / sid); done += 1
    print('converted', done, 'scenes to v3')


if __name__ == '__main__':
    main()
