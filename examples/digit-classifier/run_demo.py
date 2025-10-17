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
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from matplotlib import pyplot as plt

import numpy as np
import yaml
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


_PYPLOT: Optional["plt"] = None


def get_pyplot():
    global _PYPLOT
    if _PYPLOT is None:
        import matplotlib

        matplotlib.use("Agg")

        from matplotlib import pyplot as _plt

        _PYPLOT = _plt
    return _PYPLOT


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
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def save_confusion_matrix(output_dir: Path, class_names: Iterable[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt = get_pyplot()
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

    plt = get_pyplot()
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

    plt = get_pyplot()
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
        plt = get_pyplot()
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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML configuration file (default: config.yaml next to this script).",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to store generated assets.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Initial learning rate for gradient descent.")
    parser.add_argument("--lr-decay", type=float, default=None, help="Linear decay factor applied per epoch.")
    parser.add_argument("--test-split", type=float, default=None, help="Fraction of the dataset reserved for evaluation.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed controlling the train/test split.")
    parser.add_argument(
        "--roc-per-class",
        dest="roc_per_class",
        action="store_true",
        help="Compute ROC curves and AUC scores for each class (requires probability estimates).",
    )
    parser.add_argument(
        "--no-roc-per-class",
        dest="roc_per_class",
        action="store_false",
        help="Disable ROC curve generation even if enabled in the config file.",
    )
    parser.add_argument(
        "--learning-rate-trace",
        dest="learning_rate_trace",
        action="store_true",
        help="Persist the learning-rate schedule and loss values as JSON/PNG artifacts.",
    )
    parser.add_argument(
        "--no-learning-rate-trace",
        dest="learning_rate_trace",
        action="store_false",
        help="Disable learning-rate tracing even if enabled in the config file.",
    )
    parser.add_argument(
        "--timing-stats",
        dest="timing_stats",
        action="store_true",
        help="Record timing statistics for each epoch and export summary visualizations.",
    )
    parser.add_argument(
        "--no-timing-stats",
        dest="timing_stats",
        action="store_false",
        help="Disable timing statistics even if enabled in the config file.",
    )
    parser.set_defaults(roc_per_class=None, learning_rate_trace=None, timing_stats=None)
    return parser.parse_args()


def load_config(path: Path | None) -> tuple[Path, Dict[str, Any]]:
    if path is None:
        path = Path(__file__).with_name("config.yaml")
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
    except FileNotFoundError as exc:  # pragma: no cover - surfaces as a user-facing error.
        raise SystemExit(f"Configuration file not found: {path}") from exc

    if not isinstance(payload, dict):
        raise SystemExit(f"Configuration file must define a mapping at the top level: {path}")

    return path, payload


def resolve_settings(args: argparse.Namespace, config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    hyperparameters = config.get("hyperparameters", {}) if isinstance(config.get("hyperparameters", {}), dict) else {}
    analytics = config.get("analytics", {}) if isinstance(config.get("analytics", {}), dict) else {}

    def pick(cli_value: Any, config_value: Any, fallback: Any) -> Any:
        return cli_value if cli_value is not None else (config_value if config_value is not None else fallback)

    resolved = {
        "config_path": config_path,
        "output_dir": Path(pick(args.output_dir, config.get("output_dir"), Path("artifacts"))),
        "epochs": int(pick(args.epochs, hyperparameters.get("epochs"), 150)),
        "learning_rate": float(pick(args.learning_rate, hyperparameters.get("learning_rate"), 0.1)),
        "lr_decay": float(pick(args.lr_decay, hyperparameters.get("lr_decay"), 0.01)),
        "test_split": float(pick(args.test_split, hyperparameters.get("test_split"), 0.2)),
        "seed": int(pick(args.seed, hyperparameters.get("seed"), 13)),
        "roc_per_class": bool(pick(args.roc_per_class, analytics.get("roc_per_class"), False)),
        "learning_rate_trace": bool(pick(args.learning_rate_trace, analytics.get("learning_rate_trace"), False)),
        "timing_stats": bool(pick(args.timing_stats, analytics.get("timing_stats"), False)),
    }
    return resolved


def main() -> None:
    args = parse_args()
    config_path, config = load_config(args.config)
    settings = resolve_settings(args, config, config_path)

    output_dir = ensure_output_dir(settings["output_dir"])

    x_train, x_test, y_train, y_test, class_names = load_dataset(settings["test_split"], settings["seed"])

    training = train_model(
        x_train,
        y_train,
        settings["epochs"],
        settings["learning_rate"],
        settings["lr_decay"],
    )

    train_predictions = training.predict(x_train)
    test_probabilities = training.predict_proba(x_test)
    test_predictions = test_probabilities.argmax(axis=1)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    save_metrics(output_dir, class_names, y_test, test_predictions, train_accuracy, test_accuracy)
    save_confusion_matrix(output_dir, class_names, y_test, test_predictions)

    maybe_save_roc_curves(settings["roc_per_class"], output_dir, class_names, y_test, test_probabilities)
    maybe_save_learning_rate_trace(
        settings["learning_rate_trace"], output_dir, training.learning_rates, training.losses
    )
    maybe_save_timing_stats(settings["timing_stats"], output_dir, training.epoch_timings, training.total_time)

    summary = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "config_path": str(settings["config_path"]),
        "resolved_settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in settings.items()
            if key != "config_path"
        },
        "artifacts": sorted(str(path.name) for path in output_dir.iterdir() if path.is_file()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
