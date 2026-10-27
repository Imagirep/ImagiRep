import json
import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # = viewpoint_imagination/
R='future201_four_condition_20260913/runs/'
def load(p):return [json.loads(l) for l in open(p,encoding='utf-8-sig') if l.strip()]
cls={r['question_id']:r for r in load('fo_batch1_4_seven_category_audit_20260927/classification.jsonl')}
SRC={1:[R+'selected171_validation_20260916_122242',R+'selected171_endpoint_modality_20260916_172748',R+'selected171_endpoint_geometry_prepared_20260916_175227'],
 2:['fo_batch2_eight_conditions_20260927'],
 3:[R+'batch3_160_start_endpoint_validation_gemini37_20260921_113511','fo_batch3_six_modality_conditions_20261004'],
 4:[R+'batch4_200_independent_start_final_gemini37_20260924','fo_batch4_six_modality_conditions_20261004']}
C=['start_view','start_final','start_final_topdown','start_final_text','start_final_video','start_final_depth','start_final_surface_normals','start_final_panorama']
CATS=[('open_space_direction','Open Space Direction'),('nearest_obstacle_direction','Closest Obstacle Direction'),('visibility','Visibility'),('nearest_object','Closest Object'),('farthest_object','Farthest Object'),('object_relative_direction','Direction of a Given Object')]
look={}
for b,ds in SRC.items():
    for d in ds:
        for r in load(d+'/results/results_by_sample.jsonl'):
            if r['condition'] in C and cls.get(r['question_id'],{}).get('batch')==b: look[(r['question_id'],r['condition'])]=r['correct']
def stats(batches):
    qs=[q for q,c in look if c=='start_final' and cls[q]['batch'] in batches and cls[q]['category']!='object_selection_by_direction']
    out={}
    for k,_ in CATS:
        S=[q for q in qs if cls[q]['category']==k]
        assert all((q,c) in look for q in S for c in C)
        fin=sum(look[(q,'start_final')] for q in S)
        out[k]=dict(n=len(S),final=fin,net={c:sum(look[(q,c)] for q in S)-fin for c in C},
                    rescue={c:sum((not look[(q,'start_final')]) and look[(q,c)] for q in S) for c in C})
    return out
if __name__=='__main__':
    for bs in [(1,2),(1,2,3,4)]:
        o=stats(bs);print(bs,sum(v['n'] for v in o.values()))
        for k,v in o.items(): print(' ',k,v['n'],v['final'],[v['net'][c] for c in C],[v['rescue'][c] for c in C[2:]])
