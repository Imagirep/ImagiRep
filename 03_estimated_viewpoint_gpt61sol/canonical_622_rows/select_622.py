"""Canonical 622 rows (same rows as the oracle 622 tables) and the questions still lacking an estimated run. No model calls."""
import collections, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R = ROOT / 'docs/experiments/future201_four_condition_20260913/runs'
FROZEN = {1: R / 'selected171_validation_20260916_122242/manifest.jsonl', 2: ROOT / 'docs/experiments/fo_batch2_eight_conditions_20260927/manifest.jsonl',
          3: R / 'batch3_160_start_endpoint_validation_gemini37_20260921_113511/manifest.jsonl', 4: R / 'batch4_200_independent_start_final_gemini37_20260924/manifest.jsonl'}
SUP = ROOT / 'docs/experiments/fo_622_supplement_eight_conditions_20261005'
EXPB = ROOT / 'docs/experiments/fo_estimated_seven_conditions_b12_v5_20261004'
CLASS = ROOT / 'outputs/fo_batch1_4_seven_category_audit_20260927/classification.jsonl'
KEEP = ('row_index', 'question_id', 'trajectory_id', 'scan_id', 'question_type', 'question_set', 'show_lookback', 'endpoint_is_lookback', 'requires_separate_lookback',
        'question', 'answer_choices', 'gold_answer', 'original_answer_choices', 'original_gold_answer', 'shuffle_target_to_source', 'evidence_images', 'manifest_seed', 'source')


def rows(p): return [json.loads(l) for l in Path(p).read_text(encoding='utf-8-sig').splitlines() if l.strip()]


def main():
    mem = json.loads((SUP / 'membership.json').read_text(encoding='utf-8'))
    cls = {r['question_id']: r for r in rows(CLASS)}
    fz = {r['question_id']: r for p in FROZEN.values() for r in rows(p)}
    canon = {}
    for q, b in mem['reuse_from_batch'].items():
        r = {k: fz[q][k] for k in KEEP if k in fz[q]}; r.update(batch=b, category=cls[q]['category'], category_label=cls[q]['category_label'], row_origin=f'batch{b}_frozen_oracle_row')
        canon[q] = r
    for s in rows(SUP / 'manifest.jsonl'):
        r = {k: s[k] for k in KEEP if k in s}; r.update(batch=s['batch'], category=s['category'], category_label=s['category_label'], row_origin='622_supplement_row')
        canon[s['question_id']] = r
    assert len(canon) == 622
    expb = {r['question_id']: r for r in rows(EXPB / 'manifest.jsonl')}
    same = lambda a, b: all(a[k] == b[k] for k in ('question', 'answer_choices', 'gold_answer', 'show_lookback'))
    reuse = sorted(q for q in canon if q in expb and same(canon[q], expb[q]))
    todo = sorted(set(canon) - set(reuse))
    (HERE / 'canonical_622_rows.jsonl').write_text(''.join(json.dumps(canon[q], ensure_ascii=False) + '\n' for q in sorted(canon)), encoding='utf-8')
    (HERE / 'selection.json').write_text(json.dumps(dict(
        rule='622 freeze rows (identical to oracle 622 tables); questions whose exact row already has an estimated v5 run in fo_estimated_seven_conditions_b12_v5_20261004 reuse it',
        reused_from_expB=reuse, expB_rows_superseded=sorted(q for q in expb if q in canon and q not in reuse), question_ids=todo, images=[]), indent=1), encoding='utf-8')
    print('reuse', len(reuse), 'todo', len(todo), 'superseded expB rows', sorted(q for q in expb if q in canon and q not in reuse),
          collections.Counter(canon[q]['batch'] for q in todo))


if __name__ == '__main__':
    main()
