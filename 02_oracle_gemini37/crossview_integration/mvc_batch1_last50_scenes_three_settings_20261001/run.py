"""User-requested matched Gemini evaluation of frozen current ready MVC labels."""
import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LEGACY=HERE
spec=importlib.util.spec_from_file_location('mvc_existing_base',HERE/'base_runner_snapshot.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
CONDITIONS=('text_only','original_only','oracle_rgb')
TYPES=('cross_view_closest','cross_view_farthest','ordering')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def parts(row,condition,root,prompts):
    content=[];paths=[]
    for asset in row['settings'][condition]:
        p=root/asset['path']
        content.extend([base.text_part(asset['label']+':\n'),base.image_part(p)])
        paths.append(str(p))
    prompt=prompts[condition]
    content.append(base.text_part(f"{prompt}\n\nQuestion: {row['question']}\n{base.option_text(row['answer_choices'])}\nAnswer:"))
    return content,paths,prompt

def prompt_hash(system,prompt,row,condition):
    value={'system':system,'prompt':prompt,'question':row['question'],'answer_choices':row['answer_choices'],
           'assets':[{'label':a['label'],'path':a['path'],'sha256':sha(ROOT/a['path'])} for a in row['settings'][condition]]}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def preflight(rows):
    sys.path.insert(0,str(HERE))
    from exporter_snapshot import export_rows
    resources=json.loads((HERE/'resources.json').read_text(encoding='utf-8'))
    labels=json.loads((HERE/'labels.json').read_text(encoding='utf-8'))
    manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['source_sha256'].items():assert sha(HERE/name)==digest
    normalized={item_id:item['questions'] for item_id,item in labels['items'].items()}
    assert len(rows)==manifest['questions'] and len({r['question_id'] for r in rows})==len(rows)
    assert len({r['resource_id'] for r in rows})==manifest['resources']
    assert Counter(r['question_type'] for r in rows)==manifest['question_types']
    reused=json.loads((HERE/'reused_model_inputs.json').read_text(encoding='utf-8'))
    expected=[reused.get(r['question_id'],r) for r in export_rows(resources,labels,manifest['seed'])]
    assert json.dumps(expected,sort_keys=True)==json.dumps(rows,sort_keys=True)
    assets={}
    for r in rows:
        q=next(q for q in normalized[r['resource_id']] if q['slot']==r['slot'])
        assert q['ready'] and q['type']==r['question_type']
        assert r['randomization']['new_to_original_option'][r['gold_answer']]==q['answer']
        assert r['settings']['text_only']==[]
        assert len(r['settings']['original_only']) in (2,3)
        assert r['settings']['oracle_rgb'][:-1]==r['settings']['original_only']
        assert r['settings']['oracle_rgb'][-1]==r['oracle_image']
        for a in r['settings']['oracle_rgb']:assets[a['path']]=sha(ROOT/a['path'])
    return assets

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    rows=base.read_jsonl(HERE/'model_inputs.jsonl')
    assets=preflight(rows)
    prompt_dir=HERE/'prompts'
    for name in ('system.txt','text_system.txt','text_only.txt','original_only.txt','additional_rgb.txt'):
        assert (prompt_dir/name).is_file(), f'Missing frozen prompt: {name}'
    prompts={c:(prompt_dir/n).read_text(encoding='utf-8').strip() for c,n in zip(CONDITIONS,('text_only.txt','original_only.txt','additional_rgb.txt'))}
    systems={c:(prompt_dir/('text_system.txt' if c=='text_only' else 'system.txt')).read_text(encoding='utf-8').strip() for c in CONDITIONS}
    base.prompt_parts=parts;base.prompt_hash=prompt_hash
    config=SimpleNamespace(output_dir=HERE/'results',experiment_root=ROOT,resume=args.resume,model='gemini-3.7-flash',retries=3)
    run_manifest={'model':config.model,'temperature':0,'thinking_level':'HIGH','max_output_tokens':8192,'response_format':'A-D enum',
      'prompt_version':'mvc_spatial_reference_v2','seed':20260930,
      'workers':4,'retries':3,'n_questions':len(rows),'jobs':len(rows)*3,'counts':dict(Counter(r['question_type'] for r in rows)),
      'asset_sha256':assets,'prompt_sha256':{p.name:sha(p) for p in prompt_dir.glob('*.txt')},
      'model_inputs_sha256':sha(HERE/'model_inputs.jsonl'),'runner_sha256':sha(Path(__file__)),
      'dependency_sha256':{n:sha(HERE/n) for n in ('base_runner_snapshot.py','exporter_snapshot.py')},
      'settings':{'text_only':'No images','original_only':'All observed Start views','oracle_rgb':'All Start views + existing oracle RGB. No topdown.'}}
    if (HERE/'run_manifest.json').exists():
        previous=json.loads((HERE/'run_manifest.json').read_text())
        assert previous==run_manifest, 'Frozen run configuration changed'
    else:base.atomic_json(HERE/'run_manifest.json',run_manifest)
    for row in rows:
        for c in CONDITIONS:
            p=config.output_dir/'per_sample'/c/f"{row['question_id']}.json"
            if p.exists():
                prior=json.loads(p.read_text())
                assert args.resume and prior['prompt_hash']==prompt_hash(systems[c],prompts[c],row,c)
    if args.preflight_only:
        base.atomic_json(HERE/'preflight.json',{'status':'ready','questions':len(rows),'resources':len({r['resource_id'] for r in rows}),'planned_jobs':len(rows)*3,'model_calls':0,'run_manifest_sha256':sha(HERE/'run_manifest.json')})
        print(f'READY: {len(rows)} questions, {len(rows)*3} planned jobs; no model calls')
        return
    base.load_env(ROOT/'.env')
    config.output_dir.mkdir(exist_ok=True)
    counts=Counter();failures=[]
    jobs=[(r,c) for r in rows for c in CONDITIONS]
    print(f'Preflight passed: {len(rows)} questions, {len(jobs)} settings, workers=4',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(base.run_one,r,c,config,systems[c],prompts):(r['question_id'],c) for r,c in jobs}
        for f in as_completed(futures):
            qid,c=futures[f]
            try:
                result=f.result();counts[result['status']]+=1
                print(f"[{sum(counts.values())}/{len(jobs)}] {result['status']} {qid} {c}",flush=True)
            except Exception as error:
                counts['failed']+=1
                failures.append({'question_id':qid,'condition':c,'error_type':type(error).__name__})
                print(f"[{sum(counts.values())}/{len(jobs)}] FAILED {qid} {c}: {type(error).__name__}",flush=True)
    results=base.collect_results(config.output_dir,rows,list(CONDITIONS))
    (config.output_dir/'results_by_sample.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    if failures or len(results)!=len(jobs):
        base.atomic_json(HERE/'INCOMPLETE.json',{'completed':len(results),'expected':len(jobs),'failures':failures})
        raise SystemExit('Incomplete run. Successful records preserved. Resume only missing results.')
    by_id={r['question_id']:r for r in rows}
    for r in results:
        row=by_id[r['question_id']];c=r['condition']
        assert r['question']==row['question'] and r['answer_choices']==row['answer_choices'] and r['gold_answer']==row['gold_answer']
        assert r['correct']==(r['prediction']==row['gold_answer'])
        assert r['prompt_hash']==prompt_hash(systems[c],prompts[c],row,c)
        assert [p.replace('\\','/') for p in r['image_paths']]==[a['path'] for a in row['settings'][c]]
    assert all(sha(ROOT/p)==digest for p,digest in assets.items())
    summary={'model':config.model,'n_questions':len(rows),'n_resources':len({r['resource_id'] for r in rows}),'n_results':len(results),'by_type':{},'by_condition':{},'audit_passed':True}
    def metric(selected):
        correct=sum(r['correct'] for r in selected)
        return {'n':len(selected),'correct':correct,'accuracy':correct/len(selected) if selected else None}
    for t in TYPES:summary['by_type'][t]={c:metric([r for r in results if r['condition']==c and by_id[r['question_id']]['question_type']==t]) for c in CONDITIONS}
    for c in CONDITIONS:summary['by_condition'][c]=metric([r for r in results if r['condition']==c])
    pairs={(r['question_id'],r['condition']):r for r in results}
    summary['oracle_transitions']={t:{'rescued':sum(not pairs[r['question_id'],'original_only']['correct'] and pairs[r['question_id'],'oracle_rgb']['correct'] for r in rows if r['question_type']==t),
        'harmed':sum(pairs[r['question_id'],'original_only']['correct'] and not pairs[r['question_id'],'oracle_rgb']['correct'] for r in rows if r['question_type']==t)} for t in TYPES}
    summary['live_label_source_unchanged']=sha(ROOT/'multi_view_question_labeler/data/merged/labels.json')==sha(HERE/'live_labels_snapshot.json')
    base.atomic_json(HERE/'analysis.json',summary)
    base.atomic_json(HERE/'EXPERIMENT_COMPLETE.json',{'status':'complete','completed_results':len(results),'audit_passed':True})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
