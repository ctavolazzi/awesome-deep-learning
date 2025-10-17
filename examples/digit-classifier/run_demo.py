"""Interactive-ready demo to train and visualize a simple digit classifier.

This script fits a small neural network to scikit-learn's digits dataset and
produces a bundle of JSON + PNG artifacts. The outputs are meant to be consumed
by the static dashboard under ``web_demo/`` so that a full experiment review can
be performed without writing additional glue code.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

plt.switch_backend("Agg")

ARTIFACT_SCHEMA_VERSION = "1.0.0"
INDEX_SCHEMA_VERSION = "1.0.0"
GALLERY_FILENAME = "sample_gallery.png"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"
METRICS_FILENAME = "metrics.json"
SUMMARY_FILENAME = "run_summary.json"
PREDICTIONS_FILENAME = "predictions.json"
LOSS_CURVE_FILENAME = "loss_curve.json"
RUN_INDEX_FILENAME = "index.json"


def build_model(hidden_layer_sizes: Tuple[int, ...], random_state: int) -> Pipeline:
    """Create a preprocessing + neural network pipeline."""

    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    max_iter=400,
                    random_state=random_state,
                    verbose=False,
                ),
            ),
        ]
    )


def format_hidden_sizes(value: str) -> Tuple[int, ...]:
    """Parse a comma-separated list of layer sizes."""

    if not value.strip():
        raise argparse.ArgumentTypeError("hidden layer sizes may not be empty")

    try:
        sizes = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "hidden layer sizes must be integers separated by commas"
        ) from exc

    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("hidden layer sizes must be positive")

    return sizes


def sanitize_label(label: str) -> str:
    """Turn an arbitrary label into a filesystem-friendly slug."""

    slug = re.sub(r"[^a-zA-Z0-9-_]+", "-", label.strip()).strip("-")
    return slug or "run"


def prepare_artifact_directory(
    base_dir: pathlib.Path, *, label: str | None
) -> tuple[pathlib.Path, str, str]:
    """Create a timestamped directory to store run artifacts."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = f"-{sanitize_label(label)}" if label else ""
    run_id = f"{timestamp}{suffix}"
    candidate = base_dir / run_id
    counter = 1

    while candidate.exists():
        counter += 1
        candidate = base_dir / f"{run_id}-{counter:02d}"

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate, timestamp, candidate.name


def select_samples(
    images: np.ndarray,
    truths: np.ndarray,
    preds: np.ndarray,
    *,
    limit: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select a deterministic sample of predictions to visualize."""

    capped = min(limit, images.shape[0])
    if capped <= 0:
        raise ValueError("sample count must be at least 1")

    rng = np.random.default_rng(seed)
    indices = np.arange(images.shape[0])
    rng.shuffle(indices)
    chosen = np.sort(indices[:capped])
    return chosen, images[chosen], truths[chosen], preds[chosen]


def create_prediction_gallery(
    images: np.ndarray,
    truths: np.ndarray,
    preds: np.ndarray,
    *,
    output_path: pathlib.Path,
) -> None:
    """Save a tiled plot of sample predictions."""

    grid_size = int(np.ceil(np.sqrt(images.shape[0])))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    for ax, image, truth, pred in zip(axes.flat, images, truths, preds):
        ax.imshow(image, cmap=plt.cm.gray_r)
        ax.axis("off")
        ax.set_title(f"pred: {pred}\ntrue: {truth}", fontsize=10)

    for ax in axes.flat[images.shape[0] :]:
        ax.axis("off")

    fig.suptitle("Digit classifier predictions", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(
    matrix: np.ndarray, labels: Sequence[int], *, output_path: pathlib.Path
) -> None:
    """Persist a confusion-matrix heatmap for quick sharing."""

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accuracy: float,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, dict]:
    """Persist evaluation metrics as JSON for reuse by other tools."""

    report = classification_report(y_true, y_pred, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)

    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    metrics_path = output_dir / METRICS_FILENAME
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return metrics_path, payload


def save_run_summary(
    *,
    accuracy: float,
    train_accuracy: float,
    train_time: float,
    hidden_sizes: Tuple[int, ...],
    params: dict,
    dataset_name: str,
    train_samples: int,
    test_samples: int,
    feature_count: int,
    random_state: int,
    test_size: float,
    timestamp: str,
    artifact_dir_name: str,
    run_label: str | None,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, dict]:
    """Persist metadata about the run for the dashboard."""

    run_id = artifact_dir_name
    summary = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_name": run_label or run_id,
        "artifact_dir": artifact_dir_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "MLPClassifier",
        "hidden_layer_sizes": list(hidden_sizes),
        "activation": params.get("activation"),
        "solver": params.get("solver"),
        "alpha": params.get("alpha"),
        "learning_rate_init": params.get("learning_rate_init"),
        "max_iter": params.get("max_iter"),
        "train_accuracy": train_accuracy,
        "test_accuracy": accuracy,
        "train_time_seconds": train_time,
        "random_state": random_state,
        "test_size": test_size,
        "dataset": {
            "name": dataset_name,
            "feature_count": feature_count,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "total_samples": train_samples + test_samples,
        },
        "artifacts": {
            "metrics": METRICS_FILENAME,
            "predictions": PREDICTIONS_FILENAME,
            "loss_curve": LOSS_CURVE_FILENAME,
            "gallery_image": GALLERY_FILENAME,
            "confusion_matrix_image": CONFUSION_MATRIX_FILENAME,
        },
        "timestamp": timestamp,
    }

    summary_path = output_dir / SUMMARY_FILENAME
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary_path, summary


def save_loss_curve(
    *,
    losses: Iterable[float],
    n_iterations: int,
    converged: bool,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, dict]:
    """Save the loss curve recorded by scikit-learn's MLP classifier."""

    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "loss": [float(value) for value in losses],
        "n_iterations": int(n_iterations),
        "converged": bool(converged),
    }

    curve_path = output_dir / LOSS_CURVE_FILENAME
    with curve_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return curve_path, payload


def save_sample_predictions(
    *,
    indices: np.ndarray,
    images: np.ndarray,
    truths: np.ndarray,
    preds: np.ndarray,
    probabilities: np.ndarray,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, dict]:
    """Persist per-sample prediction details for the dashboard."""

    samples = []
    for offset, (index, image, truth, pred, probs) in enumerate(
        zip(indices, images, truths, preds, probabilities)
    ):
        confidence = float(np.max(probs))
        samples.append(
            {
                "id": f"sample-{index}",
                "order": offset,
                "true_label": int(truth),
                "predicted_label": int(pred),
                "confidence": confidence,
                "probabilities": [float(value) for value in probs],
                "pixels": [int(value) for value in image.reshape(-1)],
            }
        )

    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "samples": samples,
        "gallery_image": GALLERY_FILENAME,
    }

    samples_path = output_dir / PREDICTIONS_FILENAME
    with samples_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return samples_path, payload


def refresh_latest_copy(base_dir: pathlib.Path, run_dir: pathlib.Path) -> pathlib.Path:
    """Copy the newest run into a ``latest`` directory for quick serving."""

    latest_dir = base_dir / "latest"
    if latest_dir.exists():
        if latest_dir.is_symlink():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)
    return latest_dir


def update_run_index(
    base_dir: pathlib.Path,
    *,
    run_summary: dict,
    metrics: dict,
) -> pathlib.Path:
    """Append metadata about the current run to ``index.json``."""

    index_path = base_dir / RUN_INDEX_FILENAME
    runs: list[dict]
    latest_run_id: str | None = None

    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        runs = existing.get("runs", []) if isinstance(existing, dict) else []
    else:
        runs = []

    run_id = run_summary["run_id"]
    latest_run_id = run_id
    runs = [entry for entry in runs if entry.get("run_id") != run_id]

    runs.append(
        {
            "run_id": run_id,
            "run_name": run_summary.get("run_name", run_id),
            "artifact_dir": run_summary["artifact_dir"],
            "generated_at": run_summary["generated_at"],
            "test_accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "hidden_layer_sizes": run_summary.get("hidden_layer_sizes", []),
            "train_accuracy": run_summary.get("train_accuracy"),
            "train_time_seconds": run_summary.get("train_time_seconds"),
        }
    )

    runs.sort(key=lambda entry: entry.get("generated_at", ""), reverse=True)

    index_payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "latest_run_id": latest_run_id,
        "runs": runs,
    }

    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index_payload, handle, indent=2)

    return index_path


def run_demo(args: argparse.Namespace) -> pathlib.Path:
    base_dir = pathlib.Path(args.artifacts_root)
    base_dir.mkdir(parents=True, exist_ok=True)

    run_dir, timestamp, artifact_dir_name = prepare_artifact_directory(
        base_dir, label=args.run_label
    )

    print(f"Artifacts for this run will be stored in {run_dir}")
    print("Loading digits dataset …")
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=digits.target,
    )

    model = build_model(args.hidden_sizes, args.random_state)
    print("Training classifier …")
    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - train_start

    score = model.score(X_test, y_test)
    print(f"Test accuracy: {score:.4f}")

    train_score = model.score(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics_path, metrics_payload = save_metrics(
        y_true=y_test,
        y_pred=y_pred,
        accuracy=score,
        output_dir=run_dir,
    )
    print(f"Detailed metrics saved to {metrics_path}")

    selected_indices, selected_images, selected_truths, selected_preds = select_samples(
        images=X_test.reshape(-1, 8, 8),
        truths=y_test,
        preds=y_pred,
        limit=args.sample_count,
        seed=args.random_state,
    )

    y_proba = model.predict_proba(X_test)
    selected_probabilities = y_proba[selected_indices]

    gallery_path = run_dir / GALLERY_FILENAME
    create_prediction_gallery(
        selected_images,
        selected_truths,
        selected_preds,
        output_path=gallery_path,
    )
    print(f"Saved sample predictions to {gallery_path}")

    confusion_png_path = run_dir / CONFUSION_MATRIX_FILENAME
    save_confusion_matrix_plot(
        np.array(metrics_payload["confusion_matrix"]),
        labels=list(range(len(digits.target_names))),
        output_path=confusion_png_path,
    )

    mlp_estimator = model.named_steps["mlp"]
    predictions_path, predictions_payload = save_sample_predictions(
        indices=selected_indices,
        images=selected_images,
        truths=selected_truths,
        preds=selected_preds,
        probabilities=selected_probabilities,
        output_dir=run_dir,
    )
    print(f"Sample predictions JSON saved to {predictions_path}")

    n_iterations = getattr(mlp_estimator, "n_iter_", 0)
    curve_path, curve_payload = save_loss_curve(
        losses=getattr(mlp_estimator, "loss_curve_", []),
        n_iterations=n_iterations,
        converged=bool(n_iterations and n_iterations < mlp_estimator.max_iter),
        output_dir=run_dir,
    )
    print(f"Loss curve saved to {curve_path}")

    summary_path, summary_payload = save_run_summary(
        accuracy=score,
        train_accuracy=train_score,
        train_time=train_time,
        hidden_sizes=mlp_estimator.hidden_layer_sizes,
        params=mlp_estimator.get_params(deep=False),
        dataset_name="scikit-learn digits",
        train_samples=X_train.shape[0],
        test_samples=X_test.shape[0],
        feature_count=X_train.shape[1],
        random_state=args.random_state,
        test_size=args.test_size,
        timestamp=timestamp,
        artifact_dir_name=artifact_dir_name,
        run_label=args.run_label,
        output_dir=run_dir,
    )
    print(f"Run summary saved to {summary_path}")

    latest_dir = refresh_latest_copy(base_dir, run_dir)
    index_path = update_run_index(
        base_dir,
        run_summary=summary_payload,
        metrics=metrics_payload,
    )
    print(f"Latest artifacts mirrored to {latest_dir}")
    print(f"Run index updated at {index_path}")

    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small neural network on the scikit-learn digits dataset and "
            "visualize example predictions."
        )
    )
    parser.add_argument(
        "--artifacts-root",
        default="examples/digit-classifier/web_demo/artifacts",
        help="Directory where timestamped artifact bundles will be stored.",
    )
    parser.add_argument(
        "--output-dir",
        dest="deprecated_output_dir",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional label appended to the artifact directory name.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=format_hidden_sizes,
        default=(64,),
        help="Comma-separated hidden layer sizes, e.g. '128,64'.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset used for evaluation (0 < x < 1).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=16,
        help="Number of predictions to visualize in the gallery grid.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser


def parse_args(namespace: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(namespace)
    if getattr(args, "deprecated_output_dir", None):
        print(
            "[warning] --output-dir is deprecated; use --artifacts-root instead",
            file=sys.stderr,
        )
        args.artifacts_root = args.deprecated_output_dir
    return args


if __name__ == "__main__":
    namespace = parse_args()
    run_path = run_demo(namespace)
    print(f"Artifacts stored under: {run_path}")
