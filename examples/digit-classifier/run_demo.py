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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Number = (int, float)


def _ensure_number(name: str, value: object) -> float:
    if not isinstance(value, Number) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite numeric value, received {value!r}")
    return float(value)


def _ensure_probability(name: str, value: object) -> float:
    number = _ensure_number(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], received {value!r}")
    return number


def _ensure_number_list(name: str, values: object, *, allow_empty: bool = False) -> List[float]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of numeric values")
    result = []
    for index, value in enumerate(values):
        result.append(_ensure_number(f"{name}[{index}]", value))
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    return result


def validate_metrics_payload(payload: Dict[str, object]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("metrics payload must be a dictionary")

    for key in ("train_accuracy", "test_accuracy", "classification_report"):
        if key not in payload:
            raise ValueError(f"metrics payload missing required key '{key}'")

    for key in ("train_accuracy", "test_accuracy"):
        _ensure_probability(key, payload[key])

    report = payload["classification_report"]
    if not isinstance(report, dict):
        raise ValueError("classification_report must be a dictionary")

    required_fields = {"precision", "recall", "f1-score", "support"}
    for label, stats in report.items():
        if isinstance(stats, dict):
            missing = required_fields - stats.keys()
            if missing:
                raise ValueError(f"classification_report entry '{label}' missing fields {sorted(missing)}")
            for field in ("precision", "recall", "f1-score"):
                _ensure_probability(f"classification_report['{label}']['{field}']", stats[field])
            support_value = stats["support"]
            support = _ensure_number(f"classification_report['{label}']['support']", support_value)
            if support < 0:
                raise ValueError("classification_report support values must be non-negative")
        else:
            if label != "accuracy":
                raise ValueError(f"classification_report entry '{label}' must be a mapping")
            _ensure_probability("classification_report['accuracy']", stats)


def validate_roc_payload(payload: Dict[str, object], expected_classes: Sequence[str]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("roc payload must be a dictionary")
    expected = set(expected_classes)
    actual = set(payload.keys())
    if expected != actual:
        raise ValueError(f"roc payload keys {sorted(actual)} do not match expected classes {sorted(expected)}")

    for class_name, stats in payload.items():
        if not isinstance(stats, dict):
            raise ValueError(f"roc entry '{class_name}' must be a dictionary")
        for key in ("fpr", "tpr", "auc"):
            if key not in stats:
                raise ValueError(f"roc entry '{class_name}' missing key '{key}'")

        fpr = _ensure_number_list(f"roc['{class_name}']['fpr']", stats["fpr"])
        tpr = _ensure_number_list(f"roc['{class_name}']['tpr']", stats["tpr"])
        if len(fpr) != len(tpr):
            raise ValueError(f"roc entry '{class_name}' must have equally sized fpr and tpr arrays")
        for index, value in enumerate(fpr):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"roc['{class_name}']['fpr'][{index}] must be within [0, 1]")
        for index, value in enumerate(tpr):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"roc['{class_name}']['tpr'][{index}] must be within [0, 1]")
        _ensure_probability(f"roc['{class_name}']['auc']", stats["auc"])


def validate_training_dynamics_payload(payload: Dict[str, object]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("training dynamics payload must be a dictionary")

    for key in ("epochs", "learning_rates", "losses"):
        if key not in payload:
            raise ValueError(f"training dynamics payload missing key '{key}'")

    epochs = payload["epochs"]
    learning_rates = payload["learning_rates"]
    losses = payload["losses"]

    if not isinstance(epochs, Iterable) or isinstance(epochs, (str, bytes)):
        raise ValueError("epochs must be an iterable of integers")
    epochs_list = []
    for index, value in enumerate(epochs):
        if not isinstance(value, int):
            raise ValueError(f"epochs[{index}] must be an integer")
        epochs_list.append(value)
    if not epochs_list:
        raise ValueError("epochs must not be empty")

    learning_rates_list = _ensure_number_list("learning_rates", learning_rates)
    losses_list = _ensure_number_list("losses", losses)

    if not (len(epochs_list) == len(learning_rates_list) == len(losses_list)):
        raise ValueError("epochs, learning_rates, and losses must be the same length")

    for index, epoch in enumerate(epochs_list):
        if epoch != index + 1:
            raise ValueError("epochs must start at 1 and increase by 1 for each entry")


def validate_timing_stats_payload(payload: Dict[str, object], expected_epochs: int) -> None:
    if not isinstance(payload, dict):
        raise ValueError("timing stats payload must be a dictionary")
    for key in ("total_training_time_sec", "average_epoch_time_sec", "epoch_durations_sec"):
        if key not in payload:
            raise ValueError(f"timing stats payload missing key '{key}'")

    total = _ensure_number("total_training_time_sec", payload["total_training_time_sec"])
    average = _ensure_number("average_epoch_time_sec", payload["average_epoch_time_sec"])
    if total < 0 or average < 0:
        raise ValueError("timing stats durations must be non-negative")

    durations = _ensure_number_list("epoch_durations_sec", payload["epoch_durations_sec"], allow_empty=True)
    if durations and len(durations) != expected_epochs:
        raise ValueError("epoch_durations_sec length must match number of epochs")
    if durations and any(value < 0 for value in durations):
        raise ValueError("epoch durations must be non-negative")


@dataclass
class TrainingResult:
    """Holds parameters and diagnostics from the optimization loop."""

    weights: np.ndarray
    bias: np.ndarray
    learning_rates: List[float]
    losses: List[float]
    epoch_timings: List[float]
    total_time: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.weights + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)


def load_dataset(test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return x_train, x_test, y_train, y_test, digits.target_names


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = logits - logits.max(axis=1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    n = logits.shape[0]
    picked = log_probs[np.arange(n), labels]
    return float(-picked.mean())


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    base_lr: float,
    lr_decay: float,
) -> TrainingResult:
    n_samples, n_features = features.shape
    num_classes = int(labels.max() + 1)

    weights = np.zeros((n_features, num_classes), dtype=np.float64)
    bias = np.zeros(num_classes, dtype=np.float64)
    eye = np.eye(num_classes, dtype=np.float64)

    learning_rates: List[float] = []
    losses: List[float] = []
    epoch_timings: List[float] = []

    start = time.perf_counter()
    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        logits = features @ weights + bias
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        targets = eye[labels]
        grad = (probs - targets) / float(n_samples)
        grad_w = features.T @ grad
        grad_b = grad.sum(axis=0)

        current_lr = base_lr / (1.0 + lr_decay * epoch)
        weights -= current_lr * grad_w
        bias -= current_lr * grad_b

        loss = softmax_cross_entropy(logits, labels)
        learning_rates.append(float(current_lr))
        losses.append(float(loss))
        epoch_timings.append(time.perf_counter() - epoch_start)

    total_time = time.perf_counter() - start
    return TrainingResult(weights, bias, learning_rates, losses, epoch_timings, total_time)


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
    validate_metrics_payload(payload)
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

    validate_roc_payload(roc_data, class_names)
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
    validate_training_dynamics_payload(payload)
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
    validate_timing_stats_payload(payload, expected_epochs=len(timings))
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

    x_train, x_test, y_train, y_test, class_names = load_dataset(args.test_split, args.seed)

    training = train_model(x_train, y_train, args.epochs, args.learning_rate, args.lr_decay)

    train_predictions = training.predict(x_train)
    test_probabilities = training.predict_proba(x_test)
    test_predictions = test_probabilities.argmax(axis=1)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    save_metrics(output_dir, class_names, y_test, test_predictions, train_accuracy, test_accuracy)
    save_confusion_matrix(output_dir, class_names, y_test, test_predictions)

    maybe_save_roc_curves(args.roc_per_class, output_dir, class_names, y_test, test_probabilities)
    maybe_save_learning_rate_trace(args.learning_rate_trace, output_dir, training.learning_rates, training.losses)
    maybe_save_timing_stats(args.timing_stats, output_dir, training.epoch_timings, training.total_time)

    summary = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "artifacts": sorted(str(path.name) for path in output_dir.iterdir() if path.is_file()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
