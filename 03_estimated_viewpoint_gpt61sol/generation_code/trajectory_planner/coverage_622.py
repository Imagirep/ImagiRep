"""Coverage of the 622-question campaign freeze by estimated packages (read-only)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
tag = sys.argv[1] if len(sys.argv) > 1 else 'v5'
PK = ROOT / f'outputs/hy_estimated_packages_20261002/{tag}'
scenes = json.loads((ROOT / 'outputs/hy_autonomy_200_20261002/inputs/scenes.json').read_text())['scenes']
have = set()
for d in PK.glob('*/Estimated_Modalities/*'):
    for s in d.name.split('_')[1:]:
        have.add(f'{d.parent.parent.name}-{s}')
qids = [q['id'] for s in scenes for q in s['questions']]
missing = [q for q in qids if q not in have]
no_scene = sorted({s['id'] for s in scenes if not (PK / s['id']).is_dir()})
report = dict(tag=tag, scenes=len(scenes), scenes_with_package_dir=len(scenes) - len(no_scene), questions=len(qids),
              questions_with_package=len(qids) - len(missing), missing_questions=missing, scenes_without_package=no_scene)
(PK.parent / f'coverage_622_{tag}.json').write_text(json.dumps(report, indent=1))
print(f"coverage {tag}: scenes {report['scenes_with_package_dir']}/200, questions {report['questions_with_package']}/622; "
      f"missing scenes {no_scene[:10]}{'...' if len(no_scene) > 10 else ''}; missing questions in packaged scenes "
      f"{[q for q in missing if q.rsplit('-', 1)[0] not in no_scene][:15]}")
