"""Batch3 frozen160: the six missing modality conditions of the 2026-09-27 eight-condition spec.

start_view / start_final are NOT re-run: they are read (read-only) from the existing frozen Batch3 run
batch3_160_start_endpoint_validation_gemini37_20260921_113511 (identical runner snapshot, model and settings). Offline prepare and STRICT serial execute."""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, os, shutil, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / 'docs/experiments/future201_four_condition_20260913/runs'
SOURCE = RUNS / 'batch3_160_start_endpoint_validation_gemini37_20260921_113511'
BATCH = 3
N = 160
T = RUNS / 'selected171_endpoint_modality_20260916_172748/run_modalities.py'
E = RUNS / 'selected171_endpoint_geometry_prepared_20260916_175227/run_endpoint_geometry.py'
REUSED = ('start_view', 'start_final')
CONDITIONS = ('start_final_topdown', 'start_final_text', 'start_final_video', 'start_final_depth', 'start_final_surface_normals', 'start_final_panorama')
FIXED = {'open_space_direction':'start_view', 'nearest_obstacle_direction':'start_final_text', 'visibility':'start_final_depth', 'nearest_object':'start_final', 'farthest_object':'start_view', 'object_relative_direction':'start_final_surface_normals', 'object_selection_by_direction':'start_final'}
MODEL = 'gemini-3.7-flash'
ALL = REUSED + CONDITIONS

def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def sha(p):
    with Path(p).open('rb') as f:
        h = hashlib.sha256()
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
        return h.hexdigest()

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); os.replace(q,p)

def literal(p, name):
    for n in ast.parse(Path(p).read_text(encoding='utf-8-sig')).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets): return ast.literal_eval(n.value)
    raise ValueError(name)

def assets_manifest(p, name):
    rows = read_jsonl(p)
    result = {}
    for r in rows:
        if r['question_id'] in result: raise ValueError('Duplicate modality question')
        result[r['question_id']] = r[name]
    return result

def captions(row,c,config):
    if c=='start_view': return ['Start:']
    text = ['Start:',config['final_caption']]
    if c not in ('start_view','start_final'):
        k=c.removeprefix('start_final_')
        if k in ('text','video'): return text
        prompt = config['modality_captions'][k]
        if k=='depth': prompt=prompt.format(**row['endpoint_modalities']['depth']['color_scale'])
        text.append(prompt)
    return text

def evidence(row,c):
    start = [a for a in row['evidence_images'] if a['kind']=='start']
    final = [a for a in row['evidence_images'] if a['kind']==('lookback' if row['show_lookback'] else 'endpoint')]
    if len(start)!=1 or len(final)!=1: raise ValueError('Ambiguous frozen RGB')
    out=start if c=='start_view' else start+final
    if c not in ('start_view','start_final'):
        k=c.removeprefix('start_final_')
        if k not in ('text','video'):
            item = row['endpoint_modalities' if k in ('depth','surface_normals','panorama') else 'additional_modalities'][k]
            out=out+[{'kind':k,**item}]
    return out

def signature(row,c,config):
    k=c.removeprefix('start_final_') if c not in ('start_view','start_final') else None
    asset = row['additional_modalities'][k] if k in ('text','video') else None
    return digest({'question':row['question'],'choices':row['answer_choices'],'gold':row['gold_answer'], 'show_lookback':row['show_lookback'], 'condition':c, 'assets':[(a['kind'],a['sha256']) for a in evidence(row,c)],'extra_asset':asset,'captions':captions(row,c,config),'modality_caption':config['modality_captions'].get(k),'config':config})

def make_parts(r,c,config,base):
    from google.genai import types
    out=[]
    for a,caption in zip(evidence(r,c),captions(r,c,config),strict=True):
        out.extend([types.Part.from_text(text=caption+'\n'),types.Part.from_bytes(data=Path(a['path']).read_bytes(),mime_type='image/png')])
    if c in ('start_final_text','start_final_video'):
        k=c.removeprefix('start_final_');a=r['additional_modalities'][k];caption=config['modality_captions'][k]+'\n'
        if k=='text':out.append(types.Part.from_text(text=caption+Path(a['path']).read_text(encoding='utf-8')))
        else:out.extend([types.Part.from_text(text=caption),types.Part(inline_data=types.Blob(mime_type='video/mp4',data=Path(a['path']).read_bytes()),video_metadata=types.VideoMetadata(fps=12))])
    out.append(types.Part.from_text(text=base.question_text(r)));return out

def prepare(args):
    if (HERE/'preflight.json').exists(): raise ValueError('Already prepared; preserving frozen preparation')
    rows=read_jsonl(SOURCE/'manifest.jsonl')
    if len(rows)!=N or len({r['question_id'] for r in rows})!=N: raise ValueError('Not frozen batch')
    geom=assets_manifest(args.geometry_manifest,'endpoint_modalities')
    traj=assets_manifest(args.trajectory_manifest,'additional_modalities')
    ids={x['question_id'] for x in rows}
    if set(geom)!=ids or set(traj)!=ids: raise ValueError('Geometry and trajectory manifests must match exact frozen IDs without extras')
    cls={r['question_id']:r for r in read_jsonl(ROOT/'outputs/fo_batch1_4_seven_category_audit_20260927/classification.jsonl') if r['batch']==BATCH}
    config={'model':MODEL,'temperature':0,'thinking':'HIGH','max_output_tokens':8192,'response_mime_type':'text/x.enum','response_schema':{'type':'STRING','enum':['A','B','C','D']},'system_prompt':literal(SOURCE/'runner_snapshot.py','SYSTEM_PROMPT'),'final_caption':literal(SOURCE/'runner_snapshot.py','FINAL_VIEW_PROMPT'),'modality_captions':{**literal(T,'MODALITY_PROMPTS'),**literal(E,'MODALITY_PROMPTS')},'video_sampling_fps':12,'strict_serial':True,'prompt_version':f'historical_final_state_balanced_v3_eight_conditions_batch{BATCH}_v1'}
    paths={}; sourcepaths=[SOURCE/'manifest.jsonl',SOURCE/'results/results_by_sample.jsonl',SOURCE/'runner_snapshot.py',T,E,args.geometry_manifest,args.trajectory_manifest,ROOT/'outputs/fo_batch1_4_seven_category_audit_20260927/classification.jsonl']
    for p in sourcepaths: paths[str(Path(p).resolve())]=sha(p)
    for r in rows:
        qid=r['question_id']; cl=cls[qid]
        if cl['question']!=r['question'] or cl['answer_choices']!=r['answer_choices'] or cl['show_lookback']!=r['show_lookback']: raise ValueError('Classification content drift')
        r['category']=cl['category'];r['category_label']=cl['category_label']
        r['endpoint_modalities']=geom[qid];r['additional_modalities']=traj[qid]
        for a in r['evidence_images']+list(r['endpoint_modalities'].values())+list(r['additional_modalities'].values()):
            candidate=Path(a['path']);p=str((candidate if candidate.is_absolute() else ROOT/candidate).resolve()); actual=sha(p)
            if a.get('sha256') and a['sha256']!=actual: raise ValueError('Input hash drift: '+p)
            a['path']=p;a['sha256']=actual;paths[p]=actual
    if not set(r['category'] for r in rows)<=set(FIXED): raise ValueError('Category identifiers differ: '+repr(set(r['category'] for r in rows)))
    shutil.copyfile(SOURCE/'runner_snapshot.py',HERE/'runner_snapshot.py')
    (HERE/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    save(HERE/'config.json',config)
    for p in (HERE/'manifest.jsonl',HERE/'config.json',HERE/'runner_snapshot.py',Path(__file__)): paths[str(p.resolve())]=sha(p)
    previews=[{'question_id':rows[0]['question_id'],'condition':c,'image_prompts':captions(rows[0],c,config),'evidence':evidence(rows[0],c),'input_signature':signature(rows[0],c,config)} for c in CONDITIONS]
    save(HERE/'prompt_previews.json',previews)
    save(HERE/'preflight.json',{'status':'passed','model_calls':0,'questions':N,'conditions':CONDITIONS,'reused_conditions':REUSED,'reused_from':str(SOURCE/'results/results_by_sample.jsonl'),'planned_successful_calls':N*len(CONDITIONS),'workers':1,'verified_sha256':paths})
    print(f'Prepared frozen{N} / {len(CONDITIONS)} new conditions / {N*len(CONDITIONS)} planned successful calls; no model calls.')

def verify():
    report=json.loads((HERE/'preflight.json').read_text(encoding='utf-8'))
    if report['status']!='passed': raise ValueError('Preparation not passed')
    for p,h in report['verified_sha256'].items():
        if sha(p)!=h: raise ValueError('Frozen input/config changed: '+p)
    return read_jsonl(HERE/'manifest.jsonl'),json.loads((HERE/'config.json').read_text(encoding='utf-8'))

def valid_record(result,row,c,config):
    if result.get('input_signature')!=signature(row,c,config) or result.get('prediction') not in row['answer_choices'] or result.get('correct')!=(result.get('prediction')==row['gold_answer']) or result.get('model')!=MODEL or result.get('condition')!=c or result.get('question_id')!=row['question_id']: raise ValueError('Invalid/stale raw result '+row['question_id']+' '+c)
    expected={'question':row['question'],'answer_choices':row['answer_choices'],'gold_answer':row['gold_answer'],'show_lookback':row['show_lookback'],'category':row['category'],'system_prompt':config['system_prompt'],'image_prompts':captions(row,c,config),'provider':'gemini','temperature':0,'reasoning_effort':'high','max_output_tokens':8192,'prompt_version':config['prompt_version'],'image_count':len(evidence(row,c)),'evidence_kinds':[a['kind'] for a in evidence(row,c)],'video_sampling_fps':12 if c=='start_final_video' else None}
    if any(result.get(k)!=v for k,v in expected.items()): raise ValueError('Stored protocol/input mismatch '+row['question_id']+' '+c)
    expected_parts=2*len(evidence(row,c))+1+(1 if c=='start_final_text' else 2 if c=='start_final_video' else 0)
    if result.get('request_part_count')!=expected_parts or result.get('request_mime_types')!=['image/png']*len(evidence(row,c))+(['video/mp4'] if c=='start_final_video' else []):raise ValueError('Actual request assembly audit mismatch '+row['question_id']+' '+c)

def paired(lookup, rows, candidate, baseline='start_final'):
    counts={'wrong_to_correct':0,'correct_to_wrong':0,'both_correct':0,'both_wrong':0}
    for row in rows:
        c=candidate[row['category']] if isinstance(candidate,dict) else candidate
        before=lookup[(row['question_id'],baseline)]['correct'];after=lookup[(row['question_id'],c)]['correct']
        key=('both_correct' if before else 'wrong_to_correct') if after else ('correct_to_wrong' if before else 'both_wrong')
        counts[key]+=1
    counts.update({'n':len(rows),'net_correct_gain':counts['wrong_to_correct']-counts['correct_to_wrong'],'gain_percentage_points':100*(counts['wrong_to_correct']-counts['correct_to_wrong'])/len(rows) if rows else None})
    return counts

def summarize(rows,config):
    records=[]; lookup={}; summary={}; cats=sorted({r['category'] for r in rows})
    byq={r['question_id']:r for r in rows}
    for x in read_jsonl(SOURCE/'results/results_by_sample.jsonl'):
        r=byq[x['question_id']]
        if x['model']!=MODEL or x['question']!=r['question'] or x['gold_answer']!=r['gold_answer'] or x['answer_choices']!=r['answer_choices'] or x['show_lookback']!=r['show_lookback'] or x['correct']!=(x['prediction']==r['gold_answer']): raise ValueError('Reused baseline mismatch '+x['question_id'])
        if x['condition'] in REUSED: lookup[(x['question_id'],x['condition'])]={**x,'category':r['category'],'reused_from_existing_run':True}
    for c in REUSED:
        subset=[lookup[(r['question_id'],c)] for r in rows]
        summary[c]={'n':len(subset),'correct':sum(x['correct'] for x in subset),'accuracy':sum(x['correct'] for x in subset)/len(subset),'reused':True,'by_category':{}}
        for cat in cats:
            s=[x for x in subset if x['category']==cat]
            summary[c]['by_category'][cat]={'n':len(s),'correct':sum(x['correct'] for x in s),'accuracy':sum(x['correct'] for x in s)/len(s) if s else None}
    for c in CONDITIONS:
        folder=HERE/'results/per_sample'/c
        if folder.exists() and {p.stem for p in folder.glob('*.json')}-{x['question_id'] for x in rows}:raise ValueError('Unexpected extra per-sample results in '+c)
        for r in rows:
            p=HERE/'results/per_sample'/c/(r['question_id']+'.json')
            if p.exists():
                x=json.loads(p.read_text(encoding='utf-8'));valid_record(x,r,c,config);records.append(x);lookup[(r['question_id'],c)]=x
        subset=[r for r in records if r['condition']==c]
        summary[c]={'n':len(subset),'correct':sum(x['correct'] for x in subset),'accuracy':sum(x['correct'] for x in subset)/len(subset) if subset else None,'by_category':{}}
        for cat in cats:
            s=[x for x in subset if x['category']==cat]
            summary[c]['by_category'][cat]={'n':len(s),'correct':sum(x['correct'] for x in s),'accuracy':sum(x['correct'] for x in s)/len(s) if s else None}
    complete=all(summary[c]['n']==N for c in ALL)
    result={'status':'complete' if complete else 'incomplete','question_count':N,'condition_count':len(ALL),'reused_conditions':REUSED,'successful_results':len(records),'by_condition':summary,'strict_serial':True}
    if complete:
        best={cat:max(ALL,key=lambda c:summary[c]['by_category'][cat]['correct']) for cat in cats}
        for name,mapping in [('batch1_fixed_mapping',FIXED),(f'batch{BATCH}_retrospective_best_mapping',best)]:
            correct=sum(lookup[(r['question_id'],mapping[r['category']])]['correct'] for r in rows)
            result[name]={'mapping':{k:v for k,v in mapping.items() if k in cats},'correct':correct,'n':N,'accuracy':correct/N,'selection_on_evaluation_data':name!='batch1_fixed_mapping'}
            result[name]['paired_vs_final_rgb']=paired(lookup,rows,mapping)
        result['paired_vs_final_rgb']={c:paired(lookup,rows,c) for c in ALL}
        result['any_modality_union']={'correct':sum(any(lookup[(r['question_id'],c)]['correct'] for c in ALL) for r in rows),'n':N}
        result['paired_by_category_vs_final_rgb']={cat:{c:paired(lookup,[r for r in rows if r['category']==cat],c) for c in ALL} for cat in cats}
    (HERE/'results/results_by_sample.jsonl').parent.mkdir(parents=True,exist_ok=True)
    (HERE/'results/results_by_sample.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records),encoding='utf-8')
    save(HERE/'results/summary.json',result);save(HERE/'results/EXPERIMENT_COMPLETE.json',result)
    table=['| Condition | Correct | n | Accuracy |','| --- | ---: | ---: | ---: |']+[f"| {c} | {summary[c]['correct']} | {summary[c]['n']} | {100*summary[c]['accuracy']:.2f}% |" for c in ALL if summary[c]['n']]
    (HERE/'accuracy_table.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    if complete:
        header=['| Category | n | '+' | '.join(ALL)+' |','| --- | ---: | '+' | '.join(['---:']*len(ALL))+' |']
        for cat in cats:
            s=summary['start_view']['by_category'][cat]
            label=next(x['category_label'] for x in rows if x['category']==cat)
            header.append('| '+label+f" | {s['n']} | "+' | '.join(f"{summary[c]['by_category'][cat]['correct']}/{s['n']} ({100*summary[c]['by_category'][cat]['accuracy']:.2f}%)" for c in ALL)+' |')
        header.extend(['',f'| Category | Batch1 fixed modality | Correct/n | Batch{BATCH} retrospective best | Correct/n |','| --- | --- | ---: | --- | ---: |'])
        for cat in cats:
            a=FIXED[cat];b=best[cat];n=summary[a]['by_category'][cat]['n'];label=next(x['category_label'] for x in rows if x['category']==cat)
            header.append(f"| {label} | {a} | {summary[a]['by_category'][cat]['correct']}/{n} | {b} | {summary[b]['by_category'][cat]['correct']}/{n} |")
        for name in ('batch1_fixed_mapping',f'batch{BATCH}_retrospective_best_mapping'):
            v=result[name];header.extend(['',f"{name}: **{v['correct']}/{N} = {100*v['accuracy']:.2f}%**."])
        header.extend(['',f'Batch{BATCH} best mapping is selected on these same evaluation results and is retrospective. Ties prefer Start/no auxiliary, then the declared condition order.'])
        (HERE/'seven_category_results.md').write_text('\n'.join(header)+'\n',encoding='utf-8')
        gains=[f"start_view/start_final reused from {SOURCE.name}; any-modality union {result['any_modality_union']['correct']}/{N}.",'','| Candidate | wrong → correct | correct → wrong | Net correct | Gain pp vs Final RGB |','| --- | ---: | ---: | ---: | ---: |']
        for c in ALL:
            v=result['paired_vs_final_rgb'][c];gains.append(f"| {c} | {v['wrong_to_correct']} | {v['correct_to_wrong']} | {v['net_correct_gain']} | {v['gain_percentage_points']:+.2f} |")
        for name in ('batch1_fixed_mapping',f'batch{BATCH}_retrospective_best_mapping'):
            v=result[name]['paired_vs_final_rgb'];gains.append(f"| {name} | {v['wrong_to_correct']} | {v['correct_to_wrong']} | {v['net_correct_gain']} | {v['gain_percentage_points']:+.2f} |")
        (HERE/'paired_gains.md').write_text('\n'.join(gains)+'\n',encoding='utf-8')
    return result

def execute(args):
    rows,config=verify()
    lock=HERE/'execution.lock'
    # OS exclusive-create guards against starting a second evaluator process.
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        spec=importlib.util.spec_from_file_location('frozen_rgb',HERE/'runner_snapshot.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
        if base.SYSTEM_PROMPT!=config['system_prompt'] or base.FINAL_VIEW_PROMPT!=config['final_caption']: raise ValueError('Frozen snapshot prompt mismatch')
        assembly={}
        def checked_parts(row,c):
            parts=make_parts(row,c,config,base)
            assembly.clear();assembly.update({'request_part_count':len(parts),'request_mime_types':[p.inline_data.mime_type for p in parts if p.inline_data]})
            return parts
        base.gemini_parts=checked_parts
        base.load_env(args.env_file)
        counter=0;attempts=0
        for c in CONDITIONS:
            for row in rows:
                counter+=1;p=HERE/'results/per_sample'/c/(row['question_id']+'.json')
                if p.exists():valid_record(json.loads(p.read_text(encoding='utf-8')),row,c,config);continue
                for attempt in range(1,args.retries+1):
                    attempts+=1
                    request_started=datetime.now().astimezone().isoformat(timespec='milliseconds');request_clock=time.monotonic()
                    save(HERE/'execution_status.json',{'state':'request_in_flight','current_question':row['question_id'],'condition':c,'index':counter,'total':N*len(CONDITIONS),'attempt':attempt,'new_attempts':attempts,'workers':1,'updated_at':datetime.now().astimezone().isoformat()})
                    try:
                        prediction,usage,raw=base.generate_gemini(MODEL,row,c)
                        if prediction not in row['answer_choices']:raise ValueError('Invalid model enum answer')
                        break
                    except Exception as exc:
                        # Never persist exception bodies: provider errors can contain credentials.
                        save(HERE/'last_error.json',{'question_id':row['question_id'],'condition':c,'attempt':attempt,'error_type':type(exc).__name__})
                        print(f'Retry {row["question_id"]} {c}: {type(exc).__name__}',flush=True)
                        if attempt==args.retries:raise RuntimeError('Serial request exhausted retries; inspect error type') from None
                        time.sleep(min(45,2**attempt))
                result={**row,'condition':c,'provider':'gemini','model':MODEL,'system_prompt':config['system_prompt'],'image_prompts':captions(row,c,config),'prompt_version':config['prompt_version'],'temperature':0,'reasoning_effort':'high','max_output_tokens':8192,'prediction':prediction,'correct':prediction==row['gold_answer'],'input_signature':signature(row,c,config),'image_count':len(evidence(row,c)),'evidence_kinds':[a['kind'] for a in evidence(row,c)],'video_sampling_fps':12 if c=='start_final_video' else None,'usage_metadata':usage,'raw_response':raw,'attempts':attempt,'request_started_at':request_started,'request_duration_seconds':time.monotonic()-request_clock,'completed_at':datetime.now().astimezone().isoformat(timespec='milliseconds')}
                result.update(assembly);valid_record(result,row,c,config)
                save(p,result);print(f'[{counter}/{N*len(CONDITIONS)}] OK {row["question_id"]} {c}',flush=True)
            summarize(rows,config)
        verify();report=summarize(rows,config)
        save(HERE/'execution_status.json',{'state':report['status'],'new_attempts':attempts,'successful_results':report['successful_results'],'workers':1})
        print(json.dumps({c:{k:v for k,v in report['by_condition'][c].items() if k!='by_category'} for c in ALL},indent=2),flush=True)
    finally:
        lock.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--execute',action='store_true');group.add_argument('--summarize',action='store_true')
    parser.add_argument('--geometry-manifest',type=Path,default=ROOT/f'outputs/fo_batch{BATCH}_modalities_20261004/geometry/geometry_manifest.jsonl')
    parser.add_argument('--trajectory-manifest',type=Path,default=ROOT/f'outputs/fo_batch{BATCH}_modalities_20261004/trajectory/trajectory_manifest.jsonl')
    parser.add_argument('--env-file',type=Path,default=ROOT/'.env');parser.add_argument('--retries',type=int,default=6);parser.add_argument('--workers',type=int,choices=[1],default=1)
    args=parser.parse_args()
    if args.prepare:prepare(args)
    elif args.execute:execute(args)
    else:
        rows,config=verify();print(json.dumps(summarize(rows,config),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
