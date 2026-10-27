import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plan_path import plan
from render_seva import render


def fixture(angles: list[float]) -> dict:
    poses = []
    for i, angle in enumerate(angles, start=1):
        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_euler("y", angle, degrees=True).as_matrix()
        c2w[:3, 3] = [0.01 * i, 0.0, 0.0]
        poses.append({"image": f"image_{i}.png", "c2w_opencv": c2w.tolist(),
                      "K_original": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                      "median_scene_depth": 4.0})
    return {"case_dir": "/tmp/case", "image_hw": [480, 640], "poses": poses}


class CameraPathTests(unittest.TestCase):
    def test_two_images_rotate_at_one_center(self) -> None:
        transforms, split, info = plan(fixture([0, 90]))
        frames = transforms["frames"]
        self.assertEqual(len(frames), 21)
        self.assertEqual(split["train_ids"], [0, 1])
        centers = np.stack([np.asarray(f["transform_matrix"])[:3, 3] for f in frames])
        self.assertLess(np.max(np.linalg.norm(centers - centers[0], axis=1)), 1e-10)
        final_cv = np.asarray(frames[-1]["transform_matrix"]) @ np.diag([1, -1, -1, 1])
        angle = Rotation.from_matrix(final_cv[:3, :3]).magnitude()
        self.assertAlmostEqual(np.degrees(angle), 45.0, places=6)
        self.assertEqual(info["mode"], "rotation_midpoint")

    def test_three_images_move_horizontally_and_scale(self) -> None:
        transforms, split, info = plan(fixture([-30, 0, 30]), three_baseline=0.55)
        frames = transforms["frames"]
        self.assertEqual(len(frames), 21)
        self.assertEqual(split["train_ids"], [0, 1, 2])
        self.assertEqual(info["start_view"], 2)
        centers = np.stack([np.asarray(f["transform_matrix"])[:3, 3] for f in frames])
        self.assertLess(abs(centers[-1, 1] - centers[0, 1]), 1e-10)
        self.assertLess(centers[-1, 2], centers[0, 2])
        first_radius = np.linalg.norm(centers[0] - centers.mean(axis=0))
        end_distance = np.linalg.norm(centers[-1] - centers[0])
        self.assertAlmostEqual(end_distance * info["camera_scale"] / first_radius,
                               0.55, places=8)

    def test_renderer_uses_planned_scale_and_final_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / "scenes" / "case"
            scene.mkdir(parents=True)
            transforms, split, path = plan(fixture([-30, 0, 30]))
            (scene / "transforms.json").write_text(json.dumps(transforms))
            (scene / "train_test_split_3.json").write_text(json.dumps(split))
            (scene / "path.json").write_text(json.dumps(path))
            repo = root / "seva"
            repo.mkdir()
            (repo / "demo.py").write_text(
                "import argparse\nfrom pathlib import Path\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--data_items')\n"
                "p.add_argument('--save_subdir')\n"
                "p.add_argument('--camera_scale')\n"
                "a,_=p.parse_known_args()\n"
                "o=Path.cwd()/'work_dirs/demo/img2trajvid'/a.save_subdir/a.data_items\n"
                "(o/'samples-rgb').mkdir(parents=True)\n"
                "(o/'samples-rgb.mp4').write_bytes(b'video')\n"
                "(o/'samples-rgb/017.png').write_bytes(b'frame')\n"
            )
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "modelv1.1.safetensors").write_bytes(b"weights")
            result = render(root / "scenes", "case", root, Path(sys.executable),
                            repo, model_dir)
            self.assertTrue(Path(result["video"]).is_file())
            self.assertEqual(Path(result["final_rgb"]).name, "017.png")
            self.assertEqual(result["camera_scale"], path["camera_scale"])


if __name__ == "__main__":
    unittest.main()
