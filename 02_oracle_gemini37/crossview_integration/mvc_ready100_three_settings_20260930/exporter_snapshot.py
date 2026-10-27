"""Freeze randomized model inputs from saved ready questions; makes no model calls."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LETTERS = "ABCD"


def image_references(text, mapping):
    # Replace an entire reference list in one pass; chained replacements can
    # accidentally remap a newly assigned number a second time.
    pattern = r"\bImages?\s+(\d+(?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)\d+)*)\b"
    def replace(match):
        numbers = [int(value) for value in re.findall(r"\d+", match.group(1))]
        if any(value not in mapping for value in numbers):
            raise ValueError(f"Image reference outside observed views: {match.group(0)}")
        displayed = [str(mapping[value]) for value in numbers]
        if len(displayed) == 1:
            return f"Image {displayed[0]}"
        return "Images " + (" and ".join(displayed) if len(displayed) == 2 else ", ".join(displayed[:-1]) + " and " + displayed[-1])
    return re.sub(pattern, replace, text, flags=re.I)


def export_rows(resources, labels, seed):
    rng = random.Random(seed)
    ready = [(item, q) for item in resources["items"] for q in labels.get("items", {}).get(item["id"], {}).get("questions", []) if q.get("ready")]
    ordering = [f"{item['id']}-Q{q['slot']}" for item,q in ready if q["slot"] == 3]
    rtl_ids = set(rng.sample(ordering, len(ordering)//2))
    image_plans = {}
    # Shuffle each resource once, then reuse that plan across its questions/settings.
    for item in resources["items"]:
        observed = [asset for asset in item["images"] if asset["kind"] == "observed"]
        if [asset['label'] for asset in observed] != [f'Image {i+1}' for i in range(len(observed))]:
            raise ValueError(f"Observed labels/order mismatch: {item['id']}")
        order = list(range(len(observed)))
        rng.shuffle(order)
        image_plans[item["id"]] = (observed, order)
    outputs = []
    for item, original in ready:
        q = dict(original)
        question_id = f"{item['id']}-Q{q['slot']}"
        options = dict(q["options"])
        if set(options) != set(LETTERS) or q["answer"] not in LETTERS or not all(options.values()):
            raise ValueError(f"Invalid saved ready question: {question_id}")
        direction = None
        reversed_sequences = False
        source_direction = None
        if q["slot"] == 3:
            matches = [value for value,phrase in [("left_to_right", "left to right"), ("right_to_left", "right to left")] if re.search(re.escape(phrase),q["question"],re.I)]
            if len(matches) != 1:
                raise ValueError(f"Ordering wording is not unambiguous: {question_id}")
            current = source_direction = matches[0]
            direction = "right_to_left" if question_id in rtl_ids else "left_to_right"
            source_phrase = "left to right" if current == "left_to_right" else "right to left"
            target_phrase = "left to right" if direction == "left_to_right" else "right to left"
            if not re.search(re.escape(source_phrase), q["question"], re.I):
                raise ValueError(f"Ordering wording/metadata mismatch: {question_id}")
            q["question"] = re.sub(re.escape(source_phrase), target_phrase, q["question"], flags=re.I)
            reversed_sequences = current != direction
            sequences = {letter:[s.strip() for s in value.split(",")] for letter,value in options.items()}
            if any(len(seq) != 4 or len(set(s.casefold() for s in seq)) != 4 for seq in sequences.values()):
                raise ValueError(f"Invalid ordering objects: {question_id}")
            if any({s.casefold() for s in seq} != {s.casefold() for s in sequences['A']} for seq in sequences.values()):
                raise ValueError(f"Mismatched ordering objects: {question_id}")
            if reversed_sequences:
                options = {letter:", ".join(reversed(seq)) for letter,seq in sequences.items()}
        observed, order = image_plans[item["id"]]
        reference_map = {source+1:position+1 for position,source in enumerate(order)}
        question = image_references(q["question"], reference_map)
        options = {letter:image_references(value,reference_map) for letter,value in options.items()}
        pre_shuffle_gold = q["answer"]
        option_order = list(LETTERS)
        rng.shuffle(option_order)
        new_to_original = dict(zip(LETTERS, option_order))
        shuffled_options = {new:options[old] for new,old in new_to_original.items()}
        gold = next(new for new,old in new_to_original.items() if old == pre_shuffle_gold)
        assert shuffled_options[gold] == options[pre_shuffle_gold]
        shown = [{"label":f"Image {position+1}","path":observed[source]["path"],"original_label":observed[source]["label"]} for position,source in enumerate(order)]
        oracle = next(asset for asset in item["images"] if asset["kind"] == "oracle")
        topdown = next(asset for asset in item["images"] if asset["kind"] == "topdown")
        outputs.append({"row_index":len(outputs),"question_id":question_id,"resource_id":item["id"],"slot":q["slot"],
                        "question_type":q['type'], "reference_object":q.get('reference_object'),
                        "question":question,"answer_choices":shuffled_options,"gold_answer":gold,
                        "observed_images":shown,"oracle_image":{"label":"Oracle RGB","path":oracle["path"]},
                        "topdown_image":topdown["path"],
                        "settings":{"text_only":[],"original_only":shown,"oracle_rgb":shown+[{"label":"Oracle RGB","path":oracle["path"]}]},
                        "randomization":{"seed":seed,"source_to_display_image_number":reference_map,
                                         "new_to_original_option":new_to_original,"gold_before_option_shuffle":pre_shuffle_gold,
                                         "q3_direction":direction,"q3_sequences_reversed":reversed_sequences},
                        "source_order_direction_from_text":source_direction,
                        "source_order_direction_metadata":q.get("order_direction") if q["slot"]==3 else None,
                        "source_revision":labels["items"][item["id"]]["revision"]})
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Output directory already exists. Choose a new directory to preserve the prior freeze.")
    paths = [HERE/"data/merged/resources.json", HERE/"data/merged/labels.json"]
    resources, labels = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    rows = export_rows(resources, labels, args.seed)
    if not rows:
        raise SystemExit("No saved ready questions to export.")
    args.output_dir.mkdir(parents=True)
    (args.output_dir/"model_inputs.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    for source in paths:
        (args.output_dir/source.name).write_bytes(source.read_bytes())
    info = {"seed":args.seed,"questions":len(rows),"ordering_questions":sum(r['slot']==3 for r in rows),
            "right_to_left_questions":sum(r['randomization']['q3_direction']=='right_to_left' for r in rows),
            "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "model_calls":0,"rule":"One frozen randomization shared across all settings and repeated tests. Image references are remapped to preserve the original reference view."}
    (args.output_dir/"manifest.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(info,indent=2))


if __name__ == "__main__":
    main()
