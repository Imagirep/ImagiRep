"""End-to-end per scene: generated 3D -> question-only route planning -> estimated modalities.

Output mirrors Desktop/FO-0003_Data (estimated half):
  <sid>/Start_View/Initial_Frame.png
  <sid>/Questions/{Endpoint_Questions,Lookback_Questions,Answer_Key}.txt
  <sid>/Estimated_Modalities/{Endpoint|Lookback}_Q<slots>/{Topdown,Text,Video,Depth,Normals,Panorama,RGB}
Question options/answers are read ONLY here, for packaging; the navigator never sees them.
"""
import base64, io, json, os, re, shutil, sys, tarfile, concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from navigator import Navigator, Remote, motion_only, TASK  # noqa: E402
import paramiko  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INP = REPO / 'outputs/hy_autonomy_200_20261002/inputs'
LABELS = REPO / 'future_question_labeler/future_question_labels.json'
LATEST = TASK + '/outputs/hy_latest_scenes_20261002/'
SRV_OUT = TASK + '/outputs/hy_nav_pilot_20261002'
ORIENT = re.compile(r'(?:[,\s]+(?:and|then)?\s*(?:then\s+)?(?:turn to face (?:my|the) starting point|look back to where i come from|turn back to face (?:the way|where) i come from|'
                    r'face (?:the )?(?:same )?(?:direction|way) i go|facing the direction i go|keep facing the same direction|turn around|'
                    r'slightly turn (?:left|right)|turn (?:left|right)|face the [^,.]+|look at the [^,.]+))+\.?$', re.I)


def walk_prefix(motion):
    return ORIENT.sub('', motion.rstrip('.')).strip(' ,').lower()


def is_lookback(motion):
    return bool(re.search(r'(face (?:my|the) starting point|look back to where i come from|turn back to face (?:the way|where) i come from)\W*$', motion.rstrip('.'), re.I))


class Shell:
    def __init__(self):
        pw = os.environ['UNC_PW']
        self.j = paramiko.SSHClient(); self.j.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.j.connect(os.environ['JUMP_HOST'], username=os.environ['SSH_USER'], password=pw, timeout=30, look_for_keys=False, allow_agent=False)
        ch = self.j.get_transport().open_channel('direct-tcpip', (os.environ['GPU_HOST'], 22), ('127.0.0.1', 0), timeout=30)
        self.c = paramiko.SSHClient(); self.c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.c.connect(os.environ['GPU_HOST'], username=os.environ['SSH_USER'], password=pw, sock=ch, timeout=30, banner_timeout=60, look_for_keys=False, allow_agent=False)

    def run(self, cmd, data=None, timeout=3600):
        i, o, e = self.c.exec_command(cmd, timeout=timeout)
        if data is not None:
            i.write(data); i.channel.shutdown_write()
        out = o.read(); err = e.read(); rc = o.channel.recv_exit_status()
        if rc:
            raise RuntimeError(f'remote rc={rc}: {err.decode(errors="replace")[-800:]}')
        return out

    def close(self):
        self.c.close(); self.j.close()


def main(sid, tag):
    scenes = {s['id']: s for s in json.loads((INP / 'scenes.json').read_text())['scenes']}
    s = scenes[sid]; image = INP / 'images' / f'{sid}.png'
    work = REPO / 'outputs/hy_nav_pilot_20261002' / tag / sid; work.mkdir(parents=True, exist_ok=True)
    # Current (user-edited) question wording from the labeler; ids stay those of the active campaign set.
    lab_items = json.loads(LABELS.read_text(encoding='utf-8'))['items'].get(sid, {})
    current = {f"{sid}-Q{q['slot']}": q['question'] for q in lab_items.get('questions', []) if q.get('question')}
    qs = [dict(q, movement_condition=current.get(q['id'], q['movement_condition'])) for q in s['questions']]
    groups = {}
    for q in qs:
        groups.setdefault(walk_prefix(motion_only(q['movement_condition'])), []).append(q)
    plans = {}

    def lead_then_rest(group):
        group = sorted(group, key=lambda q: is_lookback(motion_only(q['movement_condition'])))
        lead = Navigator(group[0]['id'], image, group[0]['movement_condition'], LATEST + sid, work / group[0]['id']).run()
        out = [lead]; ws = lead.get('walk_state')
        for q in group[1:]:
            out.append(Navigator(q['id'], image, q['movement_condition'], LATEST + sid, work / q['id'], prefix=ws).run())
        return out
    with cf.ThreadPoolExecutor(3) as ex:
        for res in ex.map(lead_then_rest, groups.values()):
            for r in res:
                plans[r['question_id']] = r
    # identical motion text -> one package
    packages = {}
    for q in qs:
        if plans[q['id']].get('status') != 'complete':
            continue
        m = motion_only(q['movement_condition'])
        packages.setdefault(m, []).append(q['id'])
    sh = Shell()
    try:
        # Metric scale from the INITIAL IMAGE (MoGe-2) vs generated start-view depth; planning stays in r9 behaviour,
        # positions are then expressed in MoGe metres and rendering uses the same scale (identical camera poses).
        sc_local = REPO / 'outputs/hy_nav_pilot_20261002/scale' / f'{sid}.json'
        if not sc_local.exists():
            srv = f'{SRV_OUT}/scale/{sid}.json'
            sh.run(f'mkdir -p {SRV_OUT}/scale; [ -f {srv} ] || (source {TASK}/scripts/env.sh; export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16; '
                   f'cd {TASK}/code/hyworld2/worldgen && {TASK}/env/bin/python {TASK}/scripts/hy_nav_pilot_20261002/scale_moge.py {LATEST}{sid} '
                   f'{TASK}/inputs/hy_autonomy_200_20261002/images/{sid}.png {srv} >/dev/null 2>&1)', timeout=1800)
            sc_local.parent.mkdir(parents=True, exist_ok=True); sc_local.write_bytes(sh.run(f'cat {srv}'))
        scale = json.loads(sc_local.read_text())
        eye_moge = scale['implied_eye_height_m']
        for pl in plans.values():
            if pl.get('status') != 'complete' or pl.get('scale_source') == 'moge':
                continue
            f = scale['m_per_unit_moge'] / pl['m_per_unit']
            for k in pl['keyframes']:
                k['x'] *= f; k['y'] *= f
            pl['route_m'] = [[a * f, b * f] for a, b in pl['route_m']]
            pl['final'].update(x=pl['final']['x'] * f, y=pl['final']['y'] * f)
            pl.update(planning_m_per_unit=pl['m_per_unit'], m_per_unit=scale['m_per_unit_moge'], scale_source='moge', scale_factor_applied=f)
            (work / pl['question_id'] / 'plan_metric.json').write_text(json.dumps(pl, indent=2))
        names = {}
        for m, ids in packages.items():
            slots = '_'.join(i.split('-')[-1] for i in ids)
            name = ('Lookback_' if is_lookback(m) else 'Endpoint_') + slots
            names[name] = (ids[0], is_lookback(m))
            plan = json.dumps({k: v for k, v in plans[ids[0]].items()})
            sh.run(f'mkdir -p {SRV_OUT}/plans/{tag}/{sid} && cat > {SRV_OUT}/plans/{tag}/{sid}/{ids[0]}.json', plan.encode())
        cmds = ' ; '.join(
            f'NAV_EYE_M={eye_moge} OMP_NUM_THREADS=2 RENDER_PROCS=12 {TASK}/env/bin/python -u {TASK}/scripts/hy_nav_pilot_20261002/render_package.py '
            f'{SRV_OUT}/plans/{tag}/{sid}/{lead}.json {LATEST}{sid} {SRV_OUT}/packages/{tag}/{sid}/{name}' + (' --lookback' if lb else '') + ' >/dev/null 2>&1'
            for name, (lead, lb) in names.items())
        sh.run(f'if [ -d {SRV_OUT}/packages/{tag}/{sid} ]; then mv {SRV_OUT}/packages/{tag}/{sid} {SRV_OUT}/packages/{tag}/{sid}.bak_$(date +%Y%m%d%H%M%S); fi; cd {TASK}/code/hyworld2/worldgen && {cmds}; true', timeout=7200)
        blob = sh.run(f'cd {SRV_OUT}/packages/{tag} && tar czf - {sid}')
    finally:
        sh.close()
    final = REPO / 'outputs/hy_estimated_packages_20261002' / tag / sid
    if final.exists():
        shutil.rmtree(final)
    (final / 'Start_View').mkdir(parents=True); shutil.copy2(image, final / 'Start_View/Initial_Frame.png')
    tmp = work / '_dl'; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz').extractall(tmp, filter='data')
    shutil.copytree(tmp / sid, final / 'Estimated_Modalities'); shutil.rmtree(tmp)
    # questions (packaging only)
    lab = json.loads(LABELS.read_text(encoding='utf-8'))['items'].get(sid, {})
    slot = {f"{sid}-Q{q['slot']}": q for q in lab.get('questions', [])}
    qd = final / 'Questions'; qd.mkdir()
    end, look, key = [], [], []
    for name, (lead, lb) in sorted(names.items()):
        for qid in packages[motion_only(next(q for q in qs if q['id'] == lead)['movement_condition'])]:
            q = slot.get(qid); text = q['question'] if q else next(x for x in qs if x['id'] == qid)['movement_condition']
            block = qid + '\n' + text + ('\n' + '\n'.join(f'{k}. {v}' for k, v in q['options'].items()) if q else '') + '\n'
            (look if lb else end).append(block)
            if q:
                key.append(f'{qid}: {q["answer"]}')
    (qd / 'Endpoint_Questions.txt').write_text('\n'.join(end), encoding='utf-8')
    (qd / 'Lookback_Questions.txt').write_text('\n'.join(look), encoding='utf-8')
    (qd / 'Answer_Key.txt').write_text('\n'.join(sorted(key)) + '\n', encoding='utf-8')
    status = {qid: p.get('status') for qid, p in plans.items()}
    (final / 'pipeline_status.json').write_text(json.dumps({'scene': sid, 'questions': status, 'packages': {k: v[0] for k, v in names.items()},
                                                            'failed': {k: plans[k].get('error') for k, v in status.items() if v != 'complete'}}, indent=2))
    print(sid, json.dumps(status), list(names))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
