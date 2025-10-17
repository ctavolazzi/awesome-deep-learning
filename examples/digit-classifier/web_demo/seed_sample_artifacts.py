"""Seed synthetic artifacts for the digit-classifier dashboard.

This helper produces a fully-populated artifact bundle using only Python's
standard library. It is intended for situations where third-party
dependencies such as ``scikit-learn`` are unavailable (for example, in
network-restricted sandboxes) but you still want to preview the dashboard.

Run from the repository root::

    python examples/digit-classifier/web_demo/seed_sample_artifacts.py

After execution, serve ``examples/digit-classifier/web_demo`` with your
favorite static server and open ``index.html``. The dashboard will load the
synthetic run history and behave just like a real training run.
"""
from __future__ import annotations

import base64
import json
import math
import pathlib
import shutil
from dataclasses import dataclass
from typing import Iterable, List, Sequence

ARTIFACT_ROOT = pathlib.Path(__file__).resolve().parent / "artifacts"

PNG_PLACEHOLDER = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAIwAAACMCAYAAADnXzNDAAAACXBIWXMAAAsTAAALEwEAmpwY"
    "AAAFgElEQVR4nO2czW4TQRCH/0oUAgWKgQRMxiAEzMCIEhCCyQkpsbPseGe5WWdnd2d9rvu6"
    "urmzJ9+q6p6Z3fvruquru7p6urq7u7u7q6uqnt6Z4PBFuAnAM+AbwFHgUeBR4FHgUeBR4FHgU"
    "eBR4FHgUeBR4FHgUeBR4FHgUeBR4FHgUeBR4FHgUeBR4FHgUeBR4FHgUeBR4FHgceKqGV8pKa"
    "Sm62n9VZSUlLKM0nTwDJ2hJ6pabqh7mWUVB4AcuQUdYJ4Q51kX8gDIrFjRg+AkclwBoCD5FBb"
    "QGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoC"
    "D5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQ"
    "W0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAa"
    "Ag+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPk"
    "UFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQ"
    "GgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD"
    "5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW"
    "0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaA"
    "g+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkU"
    "FtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQG"
    "gIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5"
    "FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0"
    "BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg"
    "+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUF"
    "tAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGg"
    "IPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RQW0BoCD5F"
    "BbQGgIPkUFtAaAg+RQW0BoCD5FBbQGgIPkUFtAaAg+RAbwr8ADRH2b1RjjeYAAAAASUVORK5C"
    "YII="
)


@dataclass
class RunSpec:
    """Description of a synthetic training run."""

    run_id: str
    run_name: str
    generated_at: str
    hidden_layers: Sequence[int]
    activation: str
    solver: str
    alpha: float
    learning_rate_init: float
    max_iter: int
    train_accuracy: float
    test_accuracy: float
    train_time: float
    random_state: int
    test_size: float
    train_samples: int
    test_samples: int
    loss_curve: Sequence[float]
    iterations: int
    converged: bool
    macro_f1: float
    sample_labels: Sequence[int]
    confusion_diagonal: int
    confusion_offdiag: int


DIGIT_PATTERNS = {
    "0": [
        "00111100",
        "01000010",
        "01000010",
        "01011010",
        "01000010",
        "01000010",
        "00111100",
        "00000000",
    ],
    "1": [
        "00011000",
        "00111000",
        "00011000",
        "00011000",
        "00011000",
        "00011000",
        "00111100",
        "00000000",
    ],
    "2": [
        "00111100",
        "01000010",
        "00000010",
        "00000100",
        "00001000",
        "00010000",
        "01111110",
        "00000000",
    ],
    "3": [
        "00111100",
        "01000010",
        "00000010",
        "00011100",
        "00000010",
        "01000010",
        "00111100",
        "00000000",
    ],
    "4": [
        "00000100",
        "00001100",
        "00010100",
        "00100100",
        "01000100",
        "01111110",
        "00000100",
        "00000000",
    ],
    "5": [
        "01111110",
        "01000000",
        "01111100",
        "00000010",
        "00000010",
        "01000010",
        "00111100",
        "00000000",
    ],
    "6": [
        "00111100",
        "01000000",
        "01111100",
        "01000010",
        "01000010",
        "01000010",
        "00111100",
        "00000000",
    ],
    "7": [
        "01111110",
        "00000010",
        "00000100",
        "00001000",
        "00010000",
        "00100000",
        "00100000",
        "00000000",
    ],
    "8": [
        "00111100",
        "01000010",
        "01000010",
        "00111100",
        "01000010",
        "01000010",
        "00111100",
        "00000000",
    ],
    "9": [
        "00111100",
        "01000010",
        "01000010",
        "00111110",
        "00000010",
        "00000100",
        "00111000",
        "00000000",
    ],
}


def art_to_pixels(pattern: Sequence[str]) -> List[int]:
    """Convert 8×8 ASCII art into intensity values."""

    flat: List[int] = []
    for row in pattern:
        row = row.strip()
        if len(row) != 8:
            raise ValueError("Each digit pattern must have 8 columns")
        flat.extend(15 if char == "1" else 0 for char in row)
    if int(math.sqrt(len(flat))) ** 2 != len(flat):
        raise ValueError("Pixel grid must be a perfect square")
    return flat


def normalize(values: Iterable[float]) -> List[float]:
    total = float(sum(values))
    if total == 0:
        return [0.0 for _ in values]
    return [round(value / total, 4) for value in values]


def build_samples(run: RunSpec, misclassified_every: int = 5) -> list[dict]:
    samples: list[dict] = []
    for order, label in enumerate(run.sample_labels):
        predicted = label
        if (order + 1) % misclassified_every == 0:
            predicted = (label + 1) % 10
        base_probs = [0.05] * 10
        base_probs[predicted] = 0.62
        base_probs[label] = 0.82 if label == predicted else 0.18
        probabilities = normalize(base_probs)
        confidence = max(probabilities)
        samples.append(
            {
                "id": f"{run.run_id}-sample-{order}",
                "order": order,
                "true_label": int(label),
                "predicted_label": int(predicted),
                "confidence": float(confidence),
                "probabilities": probabilities,
                "pixels": art_to_pixels(DIGIT_PATTERNS[str(label)]),
            }
        )
    return samples


def build_confusion_matrix(diagonal: int, offdiag: int) -> list[list[int]]:
    matrix = []
    for digit in range(10):
        row = [0] * 10
        row[digit] = diagonal + digit
        row[(digit + 1) % 10] = offdiag
        matrix.append(row)
    return matrix


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def write_png(path: pathlib.Path) -> None:
    path.write_bytes(PNG_PLACEHOLDER)


def seed_run(run: RunSpec) -> None:
    run_dir = ARTIFACT_ROOT / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(run)
    confusion = build_confusion_matrix(run.confusion_diagonal, run.confusion_offdiag)
    report_support = sum(45 + digit for digit in range(10))

    metrics = {
        "schema_version": "1.0.0",
        "accuracy": run.test_accuracy,
        "macro_f1": run.macro_f1,
        "classification_report": {
            str(digit): {
                "precision": round(0.95 + digit * 0.002, 3),
                "recall": round(0.94 + digit * 0.002, 3),
                "f1-score": round(0.945 + digit * 0.002, 3),
                "support": 45 + digit,
            }
            for digit in range(10)
        },
        "confusion_matrix": confusion,
    }
    metrics["classification_report"]["accuracy"] = run.test_accuracy
    metrics["classification_report"]["macro avg"] = {
        "precision": run.macro_f1,
        "recall": run.macro_f1,
        "f1-score": run.macro_f1,
        "support": report_support,
    }
    metrics["classification_report"]["weighted avg"] = {
        "precision": run.macro_f1,
        "recall": run.macro_f1,
        "f1-score": run.macro_f1,
        "support": report_support,
    }

    summary = {
        "schema_version": "1.0.0",
        "run_id": run.run_id,
        "run_name": run.run_name,
        "artifact_dir": run.run_id,
        "generated_at": run.generated_at,
        "model": "MLPClassifier",
        "hidden_layer_sizes": list(run.hidden_layers),
        "activation": run.activation,
        "solver": run.solver,
        "alpha": run.alpha,
        "learning_rate_init": run.learning_rate_init,
        "max_iter": run.max_iter,
        "train_accuracy": run.train_accuracy,
        "test_accuracy": run.test_accuracy,
        "train_time_seconds": run.train_time,
        "random_state": run.random_state,
        "test_size": run.test_size,
        "dataset": {
            "name": "scikit-learn digits",
            "feature_count": 64,
            "train_samples": run.train_samples,
            "test_samples": run.test_samples,
            "total_samples": run.train_samples + run.test_samples,
        },
        "artifacts": {
            "metrics": "metrics.json",
            "predictions": "predictions.json",
            "loss_curve": "loss_curve.json",
            "gallery_image": "sample_gallery.png",
            "confusion_matrix_image": "confusion_matrix.png",
        },
        "timestamp": run.generated_at,
    }

    loss_curve = {
        "schema_version": "1.0.0",
        "loss": list(run.loss_curve),
        "n_iterations": run.iterations,
        "converged": run.converged,
    }

    predictions = {
        "schema_version": "1.0.0",
        "samples": samples,
        "gallery_image": "sample_gallery.png",
    }

    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "run_summary.json", summary)
    write_json(run_dir / "loss_curve.json", loss_curve)
    write_json(run_dir / "predictions.json", predictions)
    write_png(run_dir / "sample_gallery.png")
    write_png(run_dir / "confusion_matrix.png")


def refresh_latest(run: RunSpec) -> None:
    latest_dir = ARTIFACT_ROOT / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(ARTIFACT_ROOT / run.run_id, latest_dir)


def update_index(runs: Sequence[RunSpec]) -> None:
    payload = {
        "schema_version": "1.0.0",
        "latest_run_id": runs[0].run_id,
        "runs": [
            {
                "run_id": run.run_id,
                "run_name": run.run_name,
                "artifact_dir": run.run_id,
                "generated_at": run.generated_at,
                "test_accuracy": run.test_accuracy,
                "macro_f1": run.macro_f1,
                "hidden_layer_sizes": list(run.hidden_layers),
                "train_accuracy": run.train_accuracy,
                "train_time_seconds": run.train_time,
            }
            for run in runs
        ],
    }
    write_json(ARTIFACT_ROOT / "index.json", payload)


def seed_all() -> None:
    runs = [
        RunSpec(
            run_id="20240210-120000-mvp",
            run_name="MVP reference",
            generated_at="2024-02-10T12:00:00+00:00",
            hidden_layers=[64, 32],
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=300,
            train_accuracy=0.995,
            test_accuracy=0.968,
            train_time=11.4,
            random_state=0,
            test_size=0.25,
            train_samples=1347,
            test_samples=450,
            loss_curve=[1.8, 1.2, 0.85, 0.6, 0.45, 0.36, 0.31, 0.28, 0.26],
            iterations=120,
            converged=True,
            macro_f1=0.964,
            sample_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
            confusion_diagonal=45,
            confusion_offdiag=1,
        ),
        RunSpec(
            run_id="20240115-090000-baseline",
            run_name="Baseline sweep",
            generated_at="2024-01-15T09:00:00+00:00",
            hidden_layers=[32],
            activation="tanh",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=200,
            train_accuracy=0.982,
            test_accuracy=0.942,
            train_time=7.9,
            random_state=13,
            test_size=0.3,
            train_samples=1250,
            test_samples=537,
            loss_curve=[2.1, 1.5, 1.1, 0.9, 0.75, 0.63, 0.55, 0.5],
            iterations=150,
            converged=False,
            macro_f1=0.935,
            sample_labels=[3, 5, 7, 9, 1, 4, 6, 8, 0, 2, 3, 5],
            confusion_diagonal=40,
            confusion_offdiag=2,
        ),
    ]

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    for run in runs:
        seed_run(run)

    refresh_latest(runs[0])
    update_index(runs)


if __name__ == "__main__":
    seed_all()
    print(f"Seeded synthetic artifacts under {ARTIFACT_ROOT}")
