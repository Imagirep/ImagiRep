"""No-network tests of the exact three input contracts."""
import unittest
from pathlib import Path

import run_modalities as run


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = run.base.read_jsonl(Path(__file__).resolve().parent / "manifest.jsonl")

    def test_frozen_set(self):
        self.assertEqual(len(self.rows), 171)
        self.assertEqual(len({r["trajectory_id"] for r in self.rows}), 50)

    def test_all_513_input_contracts(self):
        for row in self.rows:
            final_kind = "lookback" if row["show_lookback"] else "endpoint"
            for condition in run.CONDITIONS:
                with self.subTest(question=row["question_id"], condition=condition):
                    kind = run.modality(condition)
                    parts = run.gemini_parts(row, condition)
                    self.assertEqual(parts[0].text, "Start:\n")
                    self.assertEqual(parts[2].text, run.base.FINAL_VIEW_PROMPT + "\n")
                    self.assertEqual(parts[-1].text, run.base.question_text(row))
                    self.assertEqual([i["kind"] for i in run.selected_images(row, condition)][:2], ["start", final_kind])
                    inline = [p for p in parts if p.inline_data is not None]
                    self.assertEqual([p.inline_data.mime_type for p in inline],
                                     ["image/png"] * 3 if kind == "topdown" else
                                     ["image/png", "image/png", "video/mp4"] if kind == "video" else
                                     ["image/png", "image/png"])
                    if kind == "video":
                        self.assertEqual(inline[-1].video_metadata.fps, 12)
                        self.assertEqual(inline[-1].inline_data.data, Path(row["additional_modalities"][kind]["path"]).read_bytes())
                    elif kind == "text":
                        self.assertEqual(parts[4].text, run.MODALITY_PROMPTS[kind] + "\n" +
                                         Path(row["additional_modalities"][kind]["path"]).read_text(encoding="utf-8"))
                    self.assertEqual(len(run.input_signature(row, condition, "gemini-3.7-flash")), 64)

    def test_expansion_overrides(self):
        for row in self.rows:
            if row["trajectory_id"] in {"FO-0197", "FO-0199"}:
                for asset in row["additional_modalities"].values():
                    self.assertIn("current_expansion", asset["path"])

    def test_no_other_conditions(self):
        with self.assertRaises(ValueError):
            run.modality("start_final_depth")


if __name__ == "__main__":
    unittest.main()
