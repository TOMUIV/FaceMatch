import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safe_storage import (
    load_face_database,
    safe_child_path,
    safe_output_filename,
    sanitize_filename_component,
    save_face_database,
)


class Evil:
    def __reduce__(self):
        return (os.system, ("echo blocked",))


class SafeStorageTests(unittest.TestCase):
    def test_legacy_pickle_rce_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_path = Path(temp_dir) / "face_database.pkl"
            with open(legacy_path, "wb") as file_obj:
                pickle.dump({"face_database": Evil()}, file_obj)

            with self.assertRaises(pickle.UnpicklingError):
                load_face_database(legacy_path)

    def test_safe_output_filename_removes_path_components(self):
        safe_name = safe_output_filename("../../nested/evil.png", prefix="result_", fallback="image")
        self.assertEqual(safe_name, "result_evil.png")

    def test_safe_child_path_rejects_escape_attempts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                safe_child_path(temp_dir, "../evil.png")

    def test_sanitize_filename_component_strips_separators(self):
        sanitized = sanitize_filename_component("../../client:id", fallback="client")
        self.assertNotIn("/", sanitized)
        self.assertNotIn("\\", sanitized)
        self.assertNotIn(":", sanitized)

    def test_json_roundtrip_when_numpy_available(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed in this environment")

        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "face_database.json"
            original_database = {
                "alice": {
                    "embedding": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    "sample_count": 2,
                    "image_count": 2,
                }
            }

            save_face_database(original_database, processing_stats={"faces": 2}, save_path=database_path)
            loaded = load_face_database(database_path)

            self.assertEqual(loaded["face_database"]["alice"]["sample_count"], 2)
            self.assertEqual(loaded["face_database"]["alice"]["image_count"], 2)
            np.testing.assert_allclose(
                loaded["face_database"]["alice"]["embedding"],
                original_database["alice"]["embedding"],
                rtol=1e-6,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
