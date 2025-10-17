"""Integration tests for the digit classifier demo outputs."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")


def _load_run_demo_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "examples" / "digit-classifier" / "run_demo.py"
    spec = importlib.util.spec_from_file_location("digit_classifier_run_demo", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_digit_classifier_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"

    command = [
        sys.executable,
        "examples/digit-classifier/run_demo.py",
        "--output-dir",
        str(output_dir),
        "--epochs",
        "5",
        "--roc-per-class",
        "--learning-rate-trace",
        "--timing-stats",
    ]

    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert completed.stderr == ""

    summary = json.loads(completed.stdout)
    assert set(summary.keys()) == {"train_accuracy", "test_accuracy", "artifacts"}
    assert isinstance(summary["artifacts"], list)

    expected_artifacts = {
        "confusion_matrix.png",
        "metrics.json",
        "roc_curves.json",
        "roc_curves.png",
        "training_dynamics.json",
        "learning_rate_trace.png",
        "timing_stats.json",
        "timing_stats.png",
    }
    assert expected_artifacts.issubset(set(summary["artifacts"]))

    run_demo = _load_run_demo_module()

    metrics_payload = _read_json(output_dir / "metrics.json")
    run_demo.validate_metrics_payload(metrics_payload)

    roc_payload = _read_json(output_dir / "roc_curves.json")
    _, _, _, _, class_names = run_demo.load_dataset(test_size=0.2, random_state=13)
    expected_classes = [str(name) for name in class_names]
    run_demo.validate_roc_payload(roc_payload, expected_classes)

    training_payload = _read_json(output_dir / "training_dynamics.json")
    run_demo.validate_training_dynamics_payload(training_payload)

    timing_payload = _read_json(output_dir / "timing_stats.json")
    run_demo.validate_timing_stats_payload(timing_payload, expected_epochs=5)

    for filename in expected_artifacts:
        path = output_dir / filename
        assert path.exists(), f"expected artifact missing: {path}"
