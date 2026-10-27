"""Offline tests; model clients and API entry points are explicitly forbidden."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import run_endpoint_geometry as run


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = run.base.read_jsonl(run.HERE / "manifest.jsonl")

    def test_frozen171_and_final_mapping(self):
        self.assertEqual(len(self.rows), 171)
        self.assertEqual(len({r["trajectory_id"] for r in self.rows}), 50)
        self.assertEqual(sum(r["show_lookback"] for r in self.rows), 55)

    def test_all_513_input_contracts_without_client(self):
        with patch.object(run.base, "gemini_client", side_effect=AssertionError("No API allowed")):
            for row in self.rows:
                expected_final = "lookback" if row["show_lookback"] else "endpoint"
                for condition in run.CONDITIONS:
                    with self.subTest(qid=row["question_id"], condition=condition):
                        kind = run.modality(condition)
                        images = run.selected_images(row, condition)
                        parts = run.gemini_parts(row, condition)
                        self.assertEqual([i["kind"] for i in images], ["start", expected_final, kind])
                        self.assertEqual(len(parts), 7)
                        self.assertEqual(parts[0].text, "Start:\n")
                        self.assertEqual(parts[2].text, run.base.FINAL_VIEW_PROMPT + "\n")
                        self.assertEqual(parts[4].text, run.modality_prompt(row, condition) + "\n")
                        self.assertEqual(parts[-1].text, run.base.question_text(row))
                        self.assertEqual([parts[i].inline_data.mime_type for i in (1, 3, 5)], ["image/png"] * 3)
                        for i, image in zip((1, 3, 5), images):
                            self.assertEqual(parts[i].inline_data.data, Path(image["path"]).read_bytes())
                        self.assertEqual(row["endpoint_modalities"][kind]["final_image_kind"], expected_final)
                        self.assertEqual(len(run.input_signature(row, condition, "gemini-3.7-flash")), 64)

    def test_depth_dynamic_scale_and_invalid_explanation(self):
        for row in self.rows:
            scale = row["endpoint_modalities"]["depth"]["color_scale"]
            caption = run.modality_prompt(row, "start_final_depth")
            self.assertIn(f"{scale['near_m']:.3f} m", caption)
            self.assertIn(f"{scale['far_m']:.3f} m", caption)
            self.assertIn("black means missing depth", caption)
            self.assertIn("z-depth", caption)

    def test_depth_palette_near_and_far(self):
        # The generator maps nearer depth to TURBO index255 (red), farther to
        # index0 (purple). Test display RGB, not OpenCV's internal BGR order.
        rgb = cv2.applyColorMap(np.array([[255, 0]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, :, ::-1]
        self.assertGreater(int(rgb[0, 0]), int(rgb[0, 2]))
        self.assertGreater(int(rgb[1, 2]), int(rgb[1, 0]))

    def test_actual_normals_projector_channel_axes(self):
        scripts = run.ROOT / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location("native_normal_projector", scripts / "add_mvc001_oracle_states.py")
            projector = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(projector)
            with tempfile.TemporaryDirectory(prefix="endpoint_normal_contract_") as directory:
                for world_normal, expected_rgb in [([1, 0, 0], [255, 127, 127]),
                                                   ([0, 0, 1], [127, 255, 127]),
                                                   ([0, 1, 0], [127, 127, 255]),
                                                   ([0, -1, 0], [127, 127, 0])]:
                    filename, _ = projector.make_perspective_normal_map(
                        np.array([[0, 2, 0]], dtype=np.float32), np.array([world_normal], dtype=np.float32),
                        np.zeros(3, dtype=np.float32), 0.0, 0.0, Path(directory), 60, 16, 12)
                    actual_rgb = cv2.imread(str(filename))[6, 8, ::-1]
                    self.assertEqual(actual_rgb.tolist(), expected_rgb)
        finally:
            sys.path.remove(str(scripts))

    def test_panorama_relative_directions(self):
        caption = run.MODALITY_PROMPTS["panorama"]
        self.assertIn("one-quarter of the width is left", caption)
        self.assertIn("three-quarters is right", caption)
        self.assertIn("edges are behind", caption)
        self.assertIn("one position", caption)

    def test_default_cli_cannot_run_inference_or_read_key(self):
        stream = io.StringIO()
        with patch.object(sys, "argv", ["run_endpoint_geometry.py"]), contextlib.redirect_stdout(stream), \
                patch.object(run.base, "main", side_effect=AssertionError("No inference allowed")) as inference, \
                patch.object(run.base, "load_env", side_effect=AssertionError("Do not load key offline")) as env:
            run.main()
        result = json.loads(stream.getvalue())
        self.assertEqual(result["status"], "prepared_not_run")
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["planned_cases"], 513)
        inference.assert_not_called()
        env.assert_not_called()

    def test_other_modalities_rejected(self):
        with self.assertRaises(ValueError):
            run.modality("start_final_topdown")

    def test_expansion_mapping_not_legacy(self):
        for row in self.rows:
            if row["trajectory_id"] in {"FO-0197", "FO-0199"}:
                for asset in row["endpoint_modalities"].values():
                    self.assertIn("current_expansion", asset["path"])


if __name__ == "__main__":
    unittest.main()
