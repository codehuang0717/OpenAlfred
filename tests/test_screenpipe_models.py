"""Pinned model installation never trusts a successful HTTP status alone."""

import hashlib
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import screenpipe_models


class TestScreenpipeModels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.content = b"valid onnx fixture"
        self.enterContext(patch.object(screenpipe_models, "MODELS", {
            "test.onnx": ("https://example.test/model.onnx", len(self.content), hashlib.sha256(self.content).hexdigest())
        }))

    def test_missing_and_corrupt_models_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "test.onnx"):
            screenpipe_models.verify_models(self.directory)
        (self.directory / "test.onnx").write_bytes(b"HTML not a model")
        with self.assertRaisesRegex(RuntimeError, "test.onnx"):
            screenpipe_models.verify_models(self.directory)

    def test_valid_install_is_verified_and_idempotent(self):
        with patch.object(screenpipe_models, "urlopen", return_value=BytesIO(self.content)) as download:
            self.assertEqual(screenpipe_models.install_models(self.directory), ["test.onnx"])
            self.assertEqual(screenpipe_models.install_models(self.directory), [])
        download.assert_called_once()
        screenpipe_models.verify_models(self.directory)

    def test_invalid_download_preserves_existing_cache(self):
        target = self.directory / "test.onnx"
        target.write_bytes(b"old cache")
        with patch.object(screenpipe_models, "urlopen", return_value=BytesIO(b"HTML page")):
            with self.assertRaisesRegex(RuntimeError, "下载校验失败"):
                screenpipe_models.install_models(self.directory)
        self.assertEqual(target.read_bytes(), b"old cache")

    def test_install_failure_restores_previous_cache(self):
        target = self.directory / "test.onnx"
        target.write_bytes(b"old cache")
        real_replace = Path.replace

        def replace(path, destination):
            if path.name.endswith(".download"):
                raise OSError("simulated installation failure")
            return real_replace(path, destination)

        with patch.object(screenpipe_models, "urlopen", return_value=BytesIO(self.content)):
            with patch.object(Path, "replace", autospec=True, side_effect=replace):
                with self.assertRaisesRegex(OSError, "simulated installation failure"):
                    screenpipe_models.install_models(self.directory)
        self.assertEqual(target.read_bytes(), b"old cache")
        self.assertEqual(list(self.directory.glob("*.invalid-*")), [])


if __name__ == "__main__":
    unittest.main()
