"""MVC-131 supplement (2026-10-06): the 3 questions of the MVC-300 set that were never evaluated.

MVC-300 = 253 ready non-RFR live-label questions (128 scenes) + 47 restored RFR questions
(outputs/mvc_batch1_restore_rfr_to300_20261001/proposal.json). 297 of them are in the frozen 325-row modality run.
MVC-131 (labelled 2026-10-01, revision 1) was saved after both cohort exports, so it has no results in any setting.

This script evaluates it with the SAME protocol and code as the cohorts:
  export   last50 exporter_snapshot.export_rows (seed 20260930) on the live labeler resources/labels, MVC-131 only
  stage A  text_only / original_only / oracle_rgb via the last50 runner functions and frozen last50 prompts
  stage B  oracle modality resources (build_all_mvc.build) + displayed-order eval assets (build_eval_assets)
  stage C  6 oracle_rgb_<modality> conditions via run_mvc_modalities (outputs written here, not into the 325 run)
Gemini 3.7 Flash, temperature 0, HIGH thinking, A-D enum, 8192 tokens (unchanged).
Usage: python run_supplement.py export|base|assets|modalities
"""
import importlib.util, json, shutil, sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
L50 = ROOT / 'outputs/mvc_batch1_last50_scenes_three_settings_20261001'
LAB = ROOT / 'multi_view_question_labeler/data/merged'
RID = 'MVC-131'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod); return mod


def export():
    sys.path.insert(0, str(L50)); from exporter_snapshot import export_rows
    res = json.loads((LAB / 'resources.json').read_text(encoding='utf-8'))
    lab = json.loads((LAB / 'labels.json').read_text(encoding='utf-8'))
    res_one = dict(res, items=[x for x in res['items'] if x['id'] == RID])
    lab_one = dict(lab, items={RID: lab['items'][RID]})
    assert len(res_one['items']) == 1
    rows = export_rows(res_one, lab_one, 20260930)
    assert len(rows) == 3, len(rows)
    (HERE / 'resources.json').write_text(json.dumps(res_one, indent=1, ensure_ascii=False), encoding='utf-8')
    (HERE / 'labels.json').write_text(json.dumps(lab_one, indent=1, ensure_ascii=False), encoding='utf-8')
    with open(HERE / 'model_inputs.jsonl', 'w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    if not (HERE / 'prompts').exists(): shutil.copytree(L50 / 'prompts', HERE / 'prompts')
    print('exported', [(r['question_id'], r['question_type'], r['gold_answer']) for r in rows])


def base_conditions():
    run = load('l50_run', L50 / 'run.py'); base = run.base
    rows = base.read_jsonl(HERE / 'model_inputs.jsonl')
    pd = HERE / 'prompts'; C = run.CONDITIONS
    prompts = {c: (pd / n).read_text(encoding='utf-8').strip() for c, n in zip(C, ('text_only.txt', 'original_only.txt', 'additional_rgb.txt'))}
    systems = {c: (pd / ('text_system.txt' if c == 'text_only' else 'system.txt')).read_text(encoding='utf-8').strip() for c in C}
    base.prompt_parts = run.parts; base.prompt_hash = run.prompt_hash
    cfg = SimpleNamespace(output_dir=HERE / 'results', experiment_root=ROOT, resume=True, model='gemini-3.7-flash', retries=3)
    base.load_env(ROOT / '.env'); cfg.output_dir.mkdir(exist_ok=True)
    for r in rows:
        for c in C:
            out = base.run_one(r, c, cfg, systems[c], prompts)
            print(r['question_id'], c, out.get('status'), out.get('prediction'), 'gold', r['gold_answer'], flush=True)


def assets():
    sys.path.insert(0, str(ROOT / 'outputs/mvc_modalities_20261005'))
    import build_all_mvc as B
    res = json.loads((HERE / 'resources.json').read_text(encoding='utf-8'))['items'][0]
    old = B.resources; B.resources = lambda: dict(old(), **{RID: res})
    out = B.OUT / RID
    if not (out / 'modalities_manifest.json').exists():
        mesh = B.MeshRenderer(res['scan_id'])
        print('build', B.build(RID, res, mesh)); mesh.release()
    E = load('mvc_eval_assets', ROOT / 'docs/experiments/mvc_oracle_modalities_325_20261006/build_eval_assets.py')
    E.B.resources = B.resources
    rows = [dict(json.loads(l), cohort='mvc131') for l in open(HERE / 'model_inputs.jsonl', encoding='utf-8')]
    pr = E.per_resource(RID)
    for r in rows:
        E.per_order(RID, E.display_order(r), B.resources(), pr); print('eval assets', r['question_id'], E.display_order(r))


def modalities():
    m = load('mvc_run', ROOT / 'docs/experiments/mvc_oracle_modalities_325_20261006/run_mvc_modalities.py')
    m.COHORTS = {'mvc131': HERE}; m.HERE = HERE / 'modalities'; m.HERE.mkdir(exist_ok=True)
    sys.argv = [sys.argv[0], '--execute', '--workers', '3']
    m.execute()


if __name__ == '__main__':
    {'export': export, 'base': base_conditions, 'assets': assets, 'modalities': modalities}[sys.argv[1]]()
