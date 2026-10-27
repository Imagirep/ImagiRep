"""Per-question 8-condition oracle results for the full 622 set (2026-10-02 freeze).

Reruns (label changes other than wording, or never evaluated) come from fo_622_supplement_eight_conditions_20261005;
all other 622 questions reuse their Batch1-4 oracle results.
"""
import json, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # = viewpoint_imagination/
R = 'future201_four_condition_20260913/runs/'
SUP = 'fo_622_supplement_eight_conditions_20261005/'
def load(p): return [json.loads(l) for l in open(p, encoding='utf-8-sig') if l.strip()]
C = ['start_view', 'start_final', 'start_final_topdown', 'start_final_text', 'start_final_video', 'start_final_depth', 'start_final_surface_normals', 'start_final_panorama']
CATS = [('open_space_direction', 'Open Space Direction'), ('nearest_obstacle_direction', 'Closest Obstacle Direction'), ('visibility', 'Visibility'),
        ('nearest_object', 'Closest Object'), ('farthest_object', 'Farthest Object'), ('object_relative_direction', 'Direction of a Given Object')]
SRC = {1: [R + 'selected171_validation_20260916_122242', R + 'selected171_endpoint_modality_20260916_172748', R + 'selected171_endpoint_geometry_prepared_20260916_175227'],
       2: ['fo_batch2_eight_conditions_20260927'],
       3: [R + 'batch3_160_start_endpoint_validation_gemini37_20260921_113511', 'fo_batch3_six_modality_conditions_20261004'],
       4: [R + 'batch4_200_independent_start_final_gemini37_20260924', 'fo_batch4_six_modality_conditions_20261004']}


def table622():
    mem = json.load(open(SUP + 'membership.json', encoding='utf-8'))
    cls = {r['question_id']: r for r in load('fo_batch1_4_seven_category_audit_20260927/classification.jsonl')}
    look, cat, batch = {}, {}, {}
    for q, b in mem['reuse_from_batch'].items():
        cat[q] = cls[q]['category']; batch[q] = b
    for b, ds in SRC.items():
        for d in ds:
            for r in load(d + '/results/results_by_sample.jsonl'):
                q = r['question_id']
                if r['condition'] in C and mem['reuse_from_batch'].get(q) == b: look[(q, r['condition'])] = r['correct']
    for r in load(SUP + 'manifest.jsonl'):
        cat[r['question_id']] = r['category']; batch[r['question_id']] = r['batch']
    for r in load(SUP + 'results/results_by_sample.jsonl'):
        look[(r['question_id'], r['condition'])] = r['correct']
    qs = sorted(cat)
    assert len(qs) == 622 and all((q, c) in look for q in qs for c in C), 'incomplete 622 results'
    return qs, look, cat, batch


def stats(qs, look, cat):
    out = {}
    for k, _ in CATS:
        S = [q for q in qs if cat[q] == k]
        fin = sum(look[(q, 'start_final')] for q in S)
        out[k] = dict(n=len(S), final=fin, net={c: sum(look[(q, c)] for q in S) - fin for c in C},
                      rescue={c: sum((not look[(q, 'start_final')]) and look[(q, c)] for q in S) for c in C})
    return out
