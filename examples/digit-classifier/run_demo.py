#!/usr/bin/env python3
"""Digit classification demo with optional analytics exports.

This script trains a small multinomial logistic regression model using a
hand-written gradient descent loop.  By default it reports core accuracy
metrics and saves a confusion matrix.  Additional analytics such as
per-class ROC curves, learning-rate traces, and timing information can be
computed on demand via CLI flags.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline import (
    DatasetLoader,
    PipelineConfig,
    SoftmaxGDBuilder,
    TrainingConfig,
    run_training_pipeline,
    DatasetSplit,
)


class DigitsDatasetLoader(DatasetLoader):
    """Loads and normalizes the scikit-learn digits dataset."""

    def load(self, *, test_size: float, random_state: int) -> DatasetSplit:
        digits = load_digits()
        features = digits.data.astype(np.float32)
        labels = digits.target.astype(np.int64)

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        return DatasetSplit(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            class_names=digits.target_names,
        )


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metrics(
    output_dir: Path,
    class_names: Iterable[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_accuracy: float,
    test_accuracy: float,
) -> None:
    report = classification_report(y_true, y_pred, target_names=list(class_names), output_dict=True)
    payload: Dict[str, object] = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "classification_report": report,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def save_confusion_matrix(output_dir: Path, class_names: Iterable[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Digit Classifier Confusion Matrix")
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def maybe_save_roc_curves(
    enabled: bool,
    output_dir: Path,
    class_names: Iterable[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    if not enabled:
        return

    class_names = list(class_names)
    roc_data: Dict[str, Dict[str, List[float]]] = {}

    fig, ax = plt.subplots(figsize=(7, 6))
    for class_index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y_true == class_index).astype(int), probabilities[:, class_index])
        auc_score = roc_auc_score((y_true == class_index).astype(int), probabilities[:, class_index])
        roc_data[class_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(auc_score),
        }
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={auc_score:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    ax.set_title("Per-Class ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=200)
    plt.close(fig)

    with (output_dir / "roc_curves.json").open("w", encoding="utf-8") as fh:
        json.dump(roc_data, fh, indent=2)


def maybe_save_learning_rate_trace(
    enabled: bool,
    output_dir: Path,
    learning_rates: List[float],
    losses: List[float],
) -> None:
    if not enabled:
        return

    epochs = list(range(1, len(learning_rates) + 1))
    payload = {
        "epochs": epochs,
        "learning_rates": learning_rates,
        "losses": losses,
    }
    with (output_dir / "training_dynamics.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(epochs, learning_rates, color="tab:blue", label="Learning rate")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Learning rate", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, losses, color="tab:orange", label="Loss")
    ax2.set_ylabel("Cross-entropy loss", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    ax1.set_title("Learning Rate and Loss Evolution")
    fig.tight_layout()
    fig.savefig(output_dir / "learning_rate_trace.png", dpi=200)
    plt.close(fig)


def maybe_save_timing_stats(enabled: bool, output_dir: Path, timings: List[float], total_time: float) -> None:
    if not enabled:
        return

    avg_time = float(sum(timings) / len(timings)) if timings else math.nan
    payload = {
        "total_training_time_sec": total_time,
        "average_epoch_time_sec": avg_time,
        "epoch_durations_sec": timings,
    }
    with (output_dir / "timing_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    if timings:
        epochs = list(range(1, len(timings) + 1))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(epochs, timings, color="tab:green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Duration (s)")
        ax.set_title("Epoch Timing Breakdown")
        plt.tight_layout()
        fig.savefig(output_dir / "timing_stats.png", dpi=200)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a digit classifier and optionally export diagnostics.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Where to store generated assets.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs (default: 150).")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Initial learning rate for gradient descent.")
    parser.add_argument(
        "--lr-decay", type=float, default=0.01, help="Linear decay factor applied per epoch to the learning rate."
    )
    parser.add_argument("--test-split", type=float, default=0.2, help="Fraction of the dataset reserved for evaluation.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed controlling the train/test split.")
    parser.add_argument(
        "--roc-per-class",
        action="store_true",
        help="Compute ROC curves and AUC scores for each class (requires probability estimates).",
    )
    parser.add_argument(
        "--learning-rate-trace",
        action="store_true",
        help="Persist the learning-rate schedule and loss values as JSON/PNG artifacts.",
    )
    parser.add_argument(
        "--timing-stats",
        action="store_true",
        help="Record timing statistics for each epoch and export summary visualizations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    dataset_loader = DigitsDatasetLoader()
    model_builder = SoftmaxGDBuilder()
    config = PipelineConfig(
        test_size=args.test_split,
        random_state=args.seed,
        training=TrainingConfig(
            epochs=args.epochs,
            base_learning_rate=args.learning_rate,
            learning_rate_decay=args.lr_decay,
        ),
    )

    result = run_training_pipeline(dataset_loader, model_builder, config)

    save_metrics(
        output_dir,
        result.dataset.class_names,
        result.dataset.y_test,
        result.test_predictions,
        result.train_accuracy,
        result.test_accuracy,
    )
    save_confusion_matrix(output_dir, result.dataset.class_names, result.dataset.y_test, result.test_predictions)

    maybe_save_roc_curves(
        args.roc_per_class,
        output_dir,
        result.dataset.class_names,
        result.dataset.y_test,
        result.test_probabilities,
    )
    maybe_save_learning_rate_trace(
        args.learning_rate_trace,
        output_dir,
        list(result.training.learning_rates),
        list(result.training.losses),
    )
    maybe_save_timing_stats(
        args.timing_stats,
        output_dir,
        list(result.training.epoch_timings),
        float(result.training.total_time),
    )

    summary = {
        "train_accuracy": result.train_accuracy,
        "test_accuracy": result.test_accuracy,
        "artifacts": sorted(str(path.name) for path in output_dir.iterdir() if path.is_file()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
