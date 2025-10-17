"""Smoke tests for the digit-classifier demo artifacts."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest


RUN_DEMO_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples"
    / "digit-classifier"
    / "run_demo.py"
)
SEED_HELPER_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples"
    / "digit-classifier"
    / "web_demo"
    / "seed_sample_artifacts.py"
)


def _load_module(module_path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # narrow type for mypy/static checkers
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[misc]
    return module


def test_run_demo_writes_expected_artifacts(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")

    run_demo = _load_module(RUN_DEMO_PATH, "digit_classifier_run_demo")

    args = SimpleNamespace(
        artifacts_root=str(tmp_path),
        run_label="pytest",  # ensure deterministic directory naming
        hidden_sizes=(32,),
        test_size=0.25,
        sample_count=4,
        random_state=0,
    )

    run_path = run_demo.run_demo(args)

    assert run_path.exists()

    expected_files = {
        run_demo.METRICS_FILENAME,
        run_demo.SUMMARY_FILENAME,
        run_demo.PREDICTIONS_FILENAME,
        run_demo.LOSS_CURVE_FILENAME,
        run_demo.GALLERY_FILENAME,
        run_demo.CONFUSION_MATRIX_FILENAME,
    }
    produced_files = {child.name for child in run_path.iterdir()}

    assert expected_files.issubset(produced_files)

    metrics = json.loads((run_path / run_demo.METRICS_FILENAME).read_text())
    assert metrics["schema_version"] == run_demo.ARTIFACT_SCHEMA_VERSION
    assert "confusion_matrix" in metrics

    predictions = json.loads((run_path / run_demo.PREDICTIONS_FILENAME).read_text())
    assert predictions["schema_version"] == run_demo.ARTIFACT_SCHEMA_VERSION
    assert len(predictions["samples"]) == args.sample_count

    index_payload = json.loads((tmp_path / run_demo.RUN_INDEX_FILENAME).read_text())
    assert index_payload["latest_run_id"] == run_path.name

    latest_dir = tmp_path / "latest"
    assert latest_dir.exists()
    assert {child.name for child in latest_dir.iterdir()} == produced_files


def test_seed_sample_artifacts_generates_bundle(tmp_path, monkeypatch):
    seed_helper = _load_module(SEED_HELPER_PATH, "digit_classifier_seed_helper")

    monkeypatch.setattr(seed_helper, "ARTIFACT_ROOT", tmp_path)

    seed_helper.seed_all()

    # Two runs plus the latest mirror are expected in the artifact root.
    run_dirs = [child for child in tmp_path.iterdir() if child.is_dir()]
    run_names = {child.name for child in run_dirs}

    assert "latest" in run_names
    run_names.remove("latest")
    assert len(run_names) == 2

    for run_name in run_names:
        run_path = tmp_path / run_name
        assert (run_path / "metrics.json").exists()
        assert (run_path / "predictions.json").exists()
        assert (run_path / "loss_curve.json").exists()
        assert (run_path / "run_summary.json").exists()
        assert (run_path / "sample_gallery.png").exists()
        assert (run_path / "confusion_matrix.png").exists()

    index_payload = json.loads((tmp_path / "index.json").read_text())
    indexed_runs = {entry["run_id"] for entry in index_payload["runs"]}
    assert indexed_runs == run_names
    assert index_payload["latest_run_id"] in indexed_runs

    latest_dir = tmp_path / "latest"
    latest_contents = {child.name for child in latest_dir.iterdir()}
    # The latest directory mirrors one of the seeded runs exactly.
    expected_contents = {
        "metrics.json",
        "predictions.json",
        "loss_curve.json",
        "run_summary.json",
        "sample_gallery.png",
        "confusion_matrix.png",
    }
    assert latest_contents == expected_contents
