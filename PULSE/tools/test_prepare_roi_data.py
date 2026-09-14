"""CPU-only data-preparation unit tests; no detector weights required.

Run: python -m unittest discover -s tools -p test_prepare_roi_data.py -v
"""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_roi_data.py"
spec = importlib.util.spec_from_file_location("prepare_roi_data", SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class PreparationTests(unittest.TestCase):
    def test_filter_and_prepare_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["high_res/CLIC/train/a.png", "high_res/Flickr2K/b.png"]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            # Missing held-out images are excluded before file decoding/existence checks.
            extra = ["high_res/CLIC/valid/v.png", "high_res/CLIC/test/t.png",
                     "high_res/CLIC/unknown/u.png", "kodak_ori/k.png", "tecnick/t.png"]
            manifest = root / "input.txt"
            manifest.write_text("\n".join(names + extra + [names[0]]))
            output = root / "prepared"
            prep.main(["--metadata", str(manifest), "--data-root", str(root),
                       "--output-dir", str(output), "--prepare-only"])
            self.assertEqual((output / "hr_train.txt").read_text().splitlines(), names)
            report = json.loads((output / "manifest.json").read_text())
            self.assertEqual(report["summary"]["excluded_entries"], 5)
            self.assertEqual(report["summary"]["training_images"], 2)
            with self.assertRaises(SystemExit):
                prep.main(["--metadata", str(manifest), "--data-root", str(root),
                           "--output-dir", str(output), "--prepare-only"])
            exclusion = root / "exclude.txt"
            exclusion.write_text(names[0])
            self.assertEqual(prep.training_images(manifest, root, exclusion)[0], [names[1]])
            (root / names[1]).unlink()
            with self.assertRaises(FileNotFoundError):
                prep.training_images(manifest, root)

    def test_symlink_cannot_hide_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hidden = root / "CLIC/test/image.png"
            hidden.parent.mkdir(parents=True)
            hidden.touch()
            (root / "innocent.png").symlink_to(hidden)
            manifest = root / "input.txt"
            manifest.write_text("innocent.png\n")
            with self.assertRaises(ValueError):
                prep.training_images(manifest, root)

    def test_geometry_and_thresholds(self):
        args = SimpleNamespace(face_score=.75, text_score=.85, max_area_fraction=.10,
                               min_canonical_side=8, max_canonical_side=96)
        box = [10, 10, 60, 30]
        self.assertTrue(prep.keep_box("face", box, .8, 1024, 1024, args))
        self.assertFalse(prep.keep_box("text", box, .8, 1024, 1024, args))
        self.assertTrue(prep.keep_box("text", box, .9, 1024, 1024, args))
        self.assertFalse(prep.keep_box("face", [0, 0, 512, 512], .9, 1024, 1024, args))
        self.assertEqual(prep.clamp_box([-1, -3, 20, 30], 10, 15), [0, 0, 10, 15])

    def test_ctc_blank_repeat_collapse(self):
        import numpy as np
        a = np.full((6, 1, 3), -10.)
        for i, token in enumerate([0, 1, 1, 0, 1, 2]):
            a[i, 0, token] = np.log(.8)
        length, confidence = prep.ctc_evidence(a)
        self.assertEqual(length, 3)
        self.assertAlmostEqual(confidence, .8)
        with self.assertRaises(ValueError):
            prep.ctc_evidence(np.ones((3, 1, 4)))

    def test_mixed_manifest(self):
        records = [{"image": str(i), "faces": [[0, 0, 8, 8]] if i == 0 else [],
                    "texts": []} for i in range(4)]
        roi, mixed = prep.mixed_records(records, .75, 42)
        self.assertEqual(len(roi), 1)
        self.assertEqual(len(mixed), 4)
        self.assertGreaterEqual(sum(bool(r["faces"]) for r in mixed), 3)
        self.assertEqual(mixed, prep.mixed_records(records, .75, 42)[1])
        with self.assertRaises(ValueError):
            prep.mixed_records([records[1]], .75, 42)


if __name__ == "__main__":
    unittest.main()
