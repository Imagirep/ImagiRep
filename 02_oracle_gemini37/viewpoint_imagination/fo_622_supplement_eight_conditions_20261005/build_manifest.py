"""Build the frozen source manifest for the 622-set supplement (no model calls).

622 set = 2026-10-02 campaign freeze. A 622 question is re-evaluated here (all 8 conditions, 2026-09-27 spec) when its
frozen 10/02 label differs from the label used in its Batch1-4 oracle run in gold / options / show_lookback / evidence,
or when it never had an oracle run. Other questions reuse their
existing Batch1-4 results.

Choice order: questions that already had a frozen run keep that run's exact shuffle (target->source letter map);
questions without one get the same balanced-gold shuffle as the Batch4 preparation, seed 20261005.
Modalities: reuse existing oracle final-side assets for the same final RGB, else the 622-supplement generation.
"""
import collections, hashlib, json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SUPP = ROOT / 'outputs/fo_622_supplement_modalities_20261005'
sys.path.insert(0, str(SUPP)); sys.path.insert(0, str(ROOT / 'outputs/fo_batch1_4_seven_category_audit_20260927'))
from prepare import labels_622, existing_final_assets, final_src, norm, sha  # noqa: E402
from analyze import classify, CATEGORIES  # noqa: E402

R = ROOT / 'docs/experiments/future201_four_condition_20260913/runs'
FROZEN = {1: R / 'selected171_validation_20260916_122242/manifest.jsonl', 2: ROOT / 'docs/experiments/fo_batch2_eight_conditions_20260927/manifest.jsonl',
          3: R / 'batch3_160_start_endpoint_validation_gemini37_20260921_113511/manifest.jsonl', 4: R / 'batch4_200_independent_start_final_gemini37_20260924/manifest.jsonl'}
SEED = 20261005


def rows(p): return [json.loads(l) for l in Path(p).read_text(encoding='utf-8-sig').splitlines() if l.strip()]


def evidence_key(images, key):
    return sorted((im['kind'], norm(im[key])) for im in images if im['kind'] in ('start', 'endpoint', 'lookback'))


def main():
    lv = labels_622(); pool = existing_final_assets()
    fz = {r['question_id']: (b, r) for b, p in FROZEN.items() for r in rows(p)}
    supp = {}
    for name, key in [('geometry/geometry_manifest.jsonl', 'endpoint_modalities'), ('trajectory/trajectory_manifest.jsonl', 'additional_modalities')]:
        for r in rows(SUPP / name): supp.setdefault(r['question_id'], {})[key] = r[key]
    reasons = {}; wording_only = []
    for i, q in lv.items():
        if i not in fz: reasons[i] = ['no_oracle_run']; continue
        b = fz[i][1]; w = []
        if q['question'] != b['question']: w.append('text')
        if q['show_lookback'] != b['show_lookback']: w.append('show_lookback')
        if q['options'] != b['original_answer_choices']: w.append('options')
        if q['answer'] != b['original_gold_answer']: w.append('gold')
        if evidence_key(q['images'], 'src') != evidence_key(b['evidence_images'], 'path'): w.append('evidence_images')
        if w and w != ['text']: reasons[i] = w
        elif w: wording_only.append(i)
    todo = sorted(reasons)
    fresh = [i for i in todo if i not in fz]
    rng = random.Random(SEED); order = list(fresh); rng.shuffle(order)
    gold = (list('ABCD') * ((len(order) + 3) // 4))[:len(order)]; rng.shuffle(gold)
    new_map = {}
    for i, g in zip(order, gold):
        ans = lv[i]['answer']; others_t = [x for x in 'ABCD' if x != g]; others_s = [x for x in 'ABCD' if x != ans]; rng.shuffle(others_s)
        new_map[i] = {g: ans, **dict(zip(others_t, others_s))}
    out = []
    for n, i in enumerate(todo, 1):
        q = lv[i]; original = {k: str(q['options'][k]) for k in 'ABCD'}
        t2s = fz[i][1]['shuffle_target_to_source'] if i in fz else new_map[i]
        choices = {t: original[s] for t, s in t2s.items()}; gold_t = next(t for t, s in t2s.items() if s == q['answer'])
        ev = [{'label': im['label'], 'kind': im['kind'], 'path': str((ROOT / norm(im['src'])).resolve()), 'sha256': sha(ROOT / norm(im['src']))}
              for im in q['images'] if im['kind'] != 'topdown']
        kinds = collections.Counter(im['kind'] for im in ev)
        assert kinds['start'] == 1 and kinds['endpoint'] == 1 and kinds['lookback'] == int(q['show_lookback']), i
        f = final_src(q)
        mods = json.loads(json.dumps(pool[f] if f in pool else dict(supp[i], source='fo_622_supplement_modalities_20261005')))
        for group in ('endpoint_modalities', 'additional_modalities'):
            for item in mods[group].values():
                # Batch1 panoramas were repaired in place on 2026-09-19; use the current file and keep the old hash on record.
                now = sha(item['path'])
                if item.get('sha256') and item['sha256'] != now:
                    item['sha256_at_source_run'] = item['sha256']; item['sha256'] = now; item['changed_since_source_run'] = True
        row = dict(row_index=n, question_id=i, trajectory_id=q['question_id'], scan_id=q['scan_id'], batch=int(i[3]) if i.startswith('FOB') else 1,
                   question_type=int(q['slot']), question_set=q['question_set'], show_lookback=bool(q['show_lookback']),
                   endpoint_is_lookback=bool(q['endpoint_is_lookback']), requires_separate_lookback=bool(q['requires_separate_lookback']),
                   question=q['question'], answer_choices=choices, gold_answer=gold_t, original_answer_choices=original, original_gold_answer=q['answer'],
                   shuffle_target_to_source=t2s, shuffle_origin='frozen_oracle_run' if i in fz else f'balanced_seed_{SEED}',
                   rerun_reasons=reasons[i], evidence_images=ev, endpoint_modalities=mods['endpoint_modalities'],
                   additional_modalities=mods['additional_modalities'], modality_source=mods['source'], manifest_seed=SEED,
                   source='622_freeze_labels_20261002')
        row['category'], row['subtype'] = classify(row); row['category_label'] = CATEGORIES[row['category']]
        out.append(row)
    (HERE / 'source_manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out), encoding='utf-8')
    reuse = sorted(set(lv) - set(todo))
    (HERE / 'membership.json').write_text(json.dumps(dict(set_size=len(lv), rerun=len(todo), reused=len(reuse), reused_wording_only=sorted(wording_only), reuse_from_batch={i: fz[i][0] for i in reuse},
                                                          rerun_reasons={i: reasons[i] for i in todo},
                                                          dropped_from_frozen_not_in_622=sorted(set(fz) - set(lv))), indent=1, ensure_ascii=False), encoding='utf-8')
    print(len(out), 'rows;', collections.Counter(tuple(r['rerun_reasons']) for r in out), collections.Counter(r['modality_source'] for r in out),
          collections.Counter(r['category'] for r in out), 'gold', collections.Counter(r['gold_answer'] for r in out if r['question_id'] in new_map))


if __name__ == '__main__':
    main()
