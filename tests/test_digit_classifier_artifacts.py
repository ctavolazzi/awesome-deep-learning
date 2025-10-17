"""Regression test for the digit-classifier demo artifacts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = PROJECT_ROOT / "examples" / "digit-classifier"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from artifact_schemas import (  # type: ignore import-not-found
    validate_loss_curves_payload,
    validate_metadata_payload,
    validate_metrics_payload,
    validate_predictions_payload,
)

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required for this test")
class DigitClassifierArtifactsTest(unittest.TestCase):
    """Exercise the demo end-to-end and validate the emitted artifacts."""

    def _run_demo(self, output_dir: Path) -> None:
        cmd = [
            sys.executable,
            "examples/digit-classifier/run_demo.py",
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "256",
            "--learning-rate",
            "0.5",
            "--seed",
            "123",
            "--run-name",
            "unittest",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def _load_json(self, path: Path) -> Mapping[str, object]:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, Mapping)
        return data

    def test_demo_generates_valid_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "artifacts"
            self._run_demo(output_dir)

            metrics = self._load_json(output_dir / "metrics.json")
            predictions = self._load_json(output_dir / "predictions.json")
            loss_curves = self._load_json(output_dir / "loss_curves.json")
            metadata = self._load_json(output_dir / "run_metadata.json")

            validate_metrics_payload(metrics)
            validate_predictions_payload(predictions)
            validate_loss_curves_payload(loss_curves)
            validate_metadata_payload(metadata)

            self.assertGreater(len(predictions.get("samples", [])), 0)
            self.assertTrue((output_dir / "gallery.png").exists())
            self.assertEqual(metadata["run_name"], "unittest")
            self.assertEqual(metadata["training_config"]["epochs"], 1)
            self.assertEqual(metadata["artifacts"]["metrics"], "metrics.json")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
