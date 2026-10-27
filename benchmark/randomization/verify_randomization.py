"""Check the randomization records against the released benchmark (no model calls).

Run from this folder:  python verify_randomization.py
For every question it re-derives the displayed question/options/gold from the original
annotation and the recorded transforms, and checks that they match ../benchmark922/questions.jsonl.
"""
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = {q["uid"]: q for q in map(json.loads, open(HERE.parent / "benchmark922" / "questions.jsonl", encoding="utf-8"))}
LETTERS = "ABCD"
errors = []


def in_benchmark(rec):
    q = BENCH[rec["uid"]]
    d = rec["displayed"]
    if q["gold"] != d["gold"]:
        errors.append((rec["uid"], "gold differs from benchmark"))
    prompt = q["question_prompt"]
    if d["question"].strip() not in prompt:
        errors.append((rec["uid"], "question text not in benchmark prompt"))
    for k in LETTERS:
        if f"{k}. {d['options'][k]}" not in prompt:
            errors.append((rec["uid"], f"option {k} not in benchmark prompt"))


def remap_images(text, m):
    """Remap image numbers in 'Image k', 'Images a and b', 'Image a,b'."""
    def block(mm):
        return re.sub(r"\d+", lambda d: str(m.get(d.group(), d.group())), mm.group(0))
    return re.sub(r"\bImages?\s+\d+(?:\s*(?:,|and)\s*\d+)*", block, text, flags=re.I)


def canon(t):
    # the export writes 'Image a,b' as 'Images a and b'
    return re.sub(r"\bImages?\s+(\d+)\s*(?:,|and)\s*(\d+)", r"Images \1 and \2", t.strip())


n = {"viewpoint_imagination": 0, "crossview_integration": 0}
for line in open(HERE / "viewpoint_imagination_randomization.jsonl", encoding="utf-8"):
    r = json.loads(line); n[r["setting"]] += 1
    o, d, m = r["original"], r["displayed"], r["option_map_display_to_original"]
    if sorted(m) != list(LETTERS) or sorted(m.values()) != list(LETTERS):
        errors.append((r["uid"], "option map is not a permutation"))
    if any(d["options"][k] != o["options"][m[k]] for k in LETTERS):
        errors.append((r["uid"], "displayed options != original options under the map"))
    if m[d["gold"]] != o["gold"]:
        errors.append((r["uid"], "gold not preserved"))
    in_benchmark(r)

for line in open(HERE / "crossview_integration_randomization.jsonl", encoding="utf-8"):
    r = json.loads(line); n[r["setting"]] += 1
    o, d = r["original"], r["displayed"]
    im, om = r["image_map_original_to_display"], r["option_map_display_to_original"]
    if sorted(im.values()) != list(range(1, len(im) + 1)):
        errors.append((r["uid"], "image map is not a permutation"))
    q = o["question"]
    if r["ordering_direction"]:
        src = "right to left" if r["ordering_direction"] == "left_to_right" else "left to right"
        dst = "left to right" if r["ordering_direction"] == "left_to_right" else "right to left"
        q = re.sub(re.escape(src), dst, q, flags=re.I)
        if dst not in d["question"].lower():
            errors.append((r["uid"], "ordering direction phrase missing"))
    if canon(remap_images(q, im)) != canon(d["question"]):
        errors.append((r["uid"], "question text"))
    opts = dict(o["options"])
    if r["ordering_sequences_reversed"]:
        opts = {k: ", ".join(reversed([s.strip() for s in v.split(",")])) for k, v in opts.items()}
    opts = {k: remap_images(v, im) for k, v in opts.items()}
    if any(d["options"][k] != opts[om[k]] for k in LETTERS):
        errors.append((r["uid"], "options"))
    if om[d["gold"]] != o["gold"]:
        errors.append((r["uid"], "gold not preserved"))
    in_benchmark(r)

print("checked", n, "| errors:", len(errors))
for e in errors[:20]:
    print("  ", e)
raise SystemExit(1 if errors else 0)
