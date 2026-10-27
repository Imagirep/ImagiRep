"""Offline input, randomization and emitted-prompt verification; never calls a model."""
import collections
import hashlib
import importlib.util
import itertools
import json
import re
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from exporter_snapshot import export_rows, image_references
spec=importlib.util.spec_from_file_location('prepared_runner',HERE/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
def forbid(*args,**kwargs):raise RuntimeError('Model calls disabled in offline verification')
runner.base.generate=forbid;runner.base.client=forbid;runner.base.load_env=forbid
resources=json.loads((HERE/'resources.json').read_text())
labels=json.loads((HERE/'labels.json').read_text())
rows=runner.base.read_jsonl(HERE/'model_inputs.jsonl')
items={i['id']:i for i in resources['items']}
sources={(i,q['slot']):q for i,v in labels['items'].items() for q in v['questions'] if q.get('ready')}
pattern=r'\bImages?\s+(\d+(?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)\d+)*)\b'
def groups(text):return [[int(n) for n in re.findall(r'\d+',m[1])] for m in re.finditer(pattern,text,re.I)]
# Exercise all three-image permutations, including swaps and list references.
for order in itertools.permutations((1,2,3)):
 mapping={old:new for new,old in enumerate(order,1)}
 for wording in ('Image 2,3','Images 2 and 3','Images 1, 2, and 3','Image 1','Images 1 & 2'):
  assert groups(image_references(wording,mapping))==[[mapping[n] for n in g] for g in groups(wording)]
try:image_references('Image 4',{1:2,2:1})
except ValueError:pass
else:raise AssertionError('Invalid image reference accepted')
seeds=[20260926,20260928,20260930,*range(20)]
for seed in seeds:
 exported=export_rows(resources,labels,seed)
 assert exported==export_rows(resources,labels,seed)
 seen={}
 for row in exported:
  q=sources[row['resource_id'],row['slot']];mapping=row['randomization']['source_to_display_image_number']
  assert groups(row['question'])==[[mapping[n] for n in g] for g in groups(q['question'])]
  observed=[a for a in items[row['resource_id']]['images'] if a['kind']=='observed']
  for old,new in mapping.items():assert row['observed_images'][new-1]['path']==observed[old-1]['path']
  if row['resource_id'] in seen:assert seen[row['resource_id']]==row['observed_images']
  seen[row['resource_id']]=row['observed_images']
  for new,old in row['randomization']['new_to_original_option'].items():
   expected=image_references(q['options'][old],mapping)
   if row['randomization']['q3_sequences_reversed']:
    assert [s.strip() for s in row['answer_choices'][new].split(',')]==list(reversed([s.strip() for s in expected.split(',')]))
   else:assert row['answer_choices'][new]==expected
  assert row['randomization']['new_to_original_option'][row['gold_answer']]==q['answer']
  if row['slot']==3:
   direction=row['randomization']['q3_direction']
   assert ('right to left' if direction=='right_to_left' else 'left to right') in row['question'].lower()
  assert row['settings']['text_only']==[]
  assert row['settings']['oracle_rgb'][:-1]==row['settings']['original_only']==row['observed_images']
  assert row['settings']['oracle_rgb'][-1]==row['oracle_image']
 assert sum(x['randomization']['q3_direction']=='right_to_left' for x in exported)==49
runner.preflight(rows)
prompts={c:(HERE/'prompts'/n).read_text().strip() for c,n in zip(runner.CONDITIONS,('text_only.txt','original_only.txt','additional_rgb.txt'))}
previews=[]
for row in rows:
 for condition in runner.CONDITIONS:
  parts,paths,prompt=runner.parts(row,condition,ROOT,prompts)
  emitted=[]
  for part in parts:
   if part.text:emitted.append({'text':part.text})
   else:
    assert part.inline_data and part.inline_data.data
    emitted.append({'image_mime':part.inline_data.mime_type,'image_sha256':hashlib.sha256(part.inline_data.data).hexdigest()})
  assert len(paths)==len(row['settings'][condition])
  assert emitted[-1]['text']==f"{prompt}\n\nQuestion: {row['question']}\n{runner.base.option_text(row['answer_choices'])}\nAnswer:"
  assert len(emitted)==2*len(paths)+1
  for j,asset in enumerate(row['settings'][condition]):
   assert emitted[2*j]['text']==asset['label']+':\n'
   assert emitted[2*j+1]['image_sha256']==runner.sha(ROOT/asset['path'])
  if row['question_id'] in ('MVC-001-Q1','MVC3I-002-Q1','MVC3I-002-Q3'):
   system=(HERE/'prompts'/('text_system.txt' if condition=='text_only' else 'system.txt')).read_text().strip()
   previews.append({'question_id':row['question_id'],'condition':condition,'system':system,'user_parts':emitted,'image_paths':[str(p.relative_to(ROOT)) for p in map(Path,paths)]})
report={'status':'passed','model_calls':0,'seeds':seeds,'questions_per_seed':len(rows),'resources':len({r['resource_id'] for r in rows}),'emitted_requests_checked':len(rows)*3,'all_three_image_permutations_checked':6,'reference_list_questions':sum(any(len(g)>1 for g in groups(q['question'])) for q in sources.values()),'gold_counts':dict(collections.Counter(r['gold_answer'] for r in rows)),'live_source_hashes_match_snapshot':all(runner.sha(ROOT/'multi_view_question_labeler/data/merged'/n)==runner.sha(HERE/n) for n in ('resources.json','labels.json','skips.json'))}
(HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
(HERE/'prompt_previews.json').write_text(json.dumps(previews,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,indent=2))
