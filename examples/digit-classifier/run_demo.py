"""Train a lightweight digit classifier and export demo artifacts.

This script trains a very small softmax regression model on the
``sklearn`` digits dataset (8x8 grayscale images).  It writes several
artifacts that are consumed by the accompanying static dashboard:

* ``metrics.json`` – aggregate metrics from the final model.
* ``predictions.json`` – per-sample predictions on the test split.
* ``loss_curves.json`` – training/validation losses and accuracies per epoch.
* ``run_metadata.json`` – information about the run configuration.
* ``gallery.png`` – a grid of example predictions for quick inspection.

The goal is to keep the dependencies minimal (numpy, pillow, sklearn) so
that the demo can be reproduced without a heavy deep-learning stack.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

if __package__:
    from .artifact_schemas import (
        validate_loss_curves_payload,
        validate_metadata_payload,
        validate_metrics_payload,
        validate_predictions_payload,
    )
else:  # pragma: no cover - compatibility when executing as a script
    from artifact_schemas import (
        validate_loss_curves_payload,
        validate_metadata_payload,
        validate_metrics_payload,
        validate_predictions_payload,
    )


DEFAULT_ARGUMENTS = {
    "epochs": 40,
    "batch_size": 128,
    "learning_rate": 0.5,
    "seed": 13,
    "output_dir": "artifacts/latest",
}

ARTIFACT_FILES = {
    "metrics": "metrics.json",
    "predictions": "predictions.json",
    "loss_curves": "loss_curves.json",
    "metadata": "run_metadata.json",
}


@dataclass
class TrainingConfig:
    """Configuration parameters for the training run."""

    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    output_dir: Path
    run_name: str


@dataclass
class DatasetInfo:
    name: str
    num_classes: int
    num_features: int
    image_shape: Tuple[int, int]
    train_size: int
    val_size: int
    test_size: int


@dataclass
class TrainingMetrics:
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


def load_config_overrides(path: Path) -> Dict[str, object]:
    """Load configuration overrides from a JSON file."""

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, Mapping):
        raise ValueError("Configuration file must contain a JSON object.")

    allowed_keys = set(DEFAULT_ARGUMENTS) | {"run_name"}
    overrides: Dict[str, object] = {}
    for key in allowed_keys:
        if key in data:
            overrides[key] = data[key]
    return overrides


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Combine CLI arguments with config-file overrides."""

    resolved: Dict[str, object] = dict(DEFAULT_ARGUMENTS)
    if args.config:
        resolved.update(load_config_overrides(Path(args.config)))

    for key in ("epochs", "batch_size", "learning_rate", "seed", "run_name"):
        value = getattr(args, key, None)
        if value is not None:
            resolved[key] = value

    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        resolved["output_dir"] = output_dir

    output_dir_path = Path(str(resolved["output_dir"]))
    run_name = resolved.get("run_name")
    seed = int(resolved.get("seed", DEFAULT_ARGUMENTS["seed"]))

    return TrainingConfig(
        seed=seed,
        epochs=int(resolved.get("epochs", DEFAULT_ARGUMENTS["epochs"])),
        batch_size=int(resolved.get("batch_size", DEFAULT_ARGUMENTS["batch_size"])),
        learning_rate=float(resolved.get("learning_rate", DEFAULT_ARGUMENTS["learning_rate"])),
        output_dir=output_dir_path,
        run_name=str(run_name) if run_name else build_run_name(seed),
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    """Apply the softmax function in a numerically stable way."""

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    sums = exp_values.sum(axis=1, keepdims=True)
    return exp_values / sums


def cross_entropy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Average cross entropy between predicted probabilities and labels."""

    eps = 1e-12
    picked = probabilities[np.arange(labels.size), labels]
    return float(-np.log(picked + eps).mean())


def accuracy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    preds = probabilities.argmax(axis=1)
    return float((preds == labels).mean())


def iterate_minibatches(
    rng: np.random.Generator,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    indices = rng.permutation(features.shape[0])
    for start in range(0, indices.size, batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]
        yield features[batch_indices], labels[batch_indices]


def train_model(
    config: TrainingConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[TrainingMetrics]]:
    """Train a softmax regression model using mini-batch gradient descent."""

    rng = np.random.default_rng(config.seed)
    num_features = x_train.shape[1]
    num_classes = y_train.max() + 1

    weights = rng.normal(scale=0.01, size=(num_features, num_classes))
    biases = np.zeros(num_classes, dtype=np.float64)

    history: List[TrainingMetrics] = []

    for epoch in range(config.epochs):
        for batch_x, batch_y in iterate_minibatches(rng, x_train, y_train, config.batch_size):
            logits = batch_x @ weights + biases
            probabilities = softmax(logits)

            grad_logits = probabilities
            grad_logits[np.arange(batch_y.size), batch_y] -= 1.0
            grad_logits /= batch_y.size

            grad_weights = batch_x.T @ grad_logits
            grad_biases = grad_logits.sum(axis=0)

            weights -= config.learning_rate * grad_weights
            biases -= config.learning_rate * grad_biases

        train_probs = softmax(x_train @ weights + biases)
        val_probs = softmax(x_val @ weights + biases)

        metrics = TrainingMetrics(
            train_loss=cross_entropy(train_probs, y_train),
            train_accuracy=accuracy(train_probs, y_train),
            val_loss=cross_entropy(val_probs, y_val),
            val_accuracy=accuracy(val_probs, y_val),
        )
        history.append(metrics)

    return weights, biases, history


def render_gallery(
    output_path: Path,
    digits_images: np.ndarray,
    test_indices: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Create a gallery grid that shows sample predictions."""

    grid_rows = 5
    grid_cols = 5
    cell_size = 48
    spacing = 8
    palette = {
        "correct": (50, 168, 82),
        "incorrect": (207, 76, 65),
    }

    canvas_width = grid_cols * cell_size + (grid_cols + 1) * spacing
    canvas_height = grid_rows * cell_size + (grid_rows + 1) * spacing
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(30, 30, 30))

    for slot, sample_idx in enumerate(test_indices[: grid_rows * grid_cols]):
        row = slot // grid_cols
        col = slot % grid_cols
        x_offset = spacing + col * (cell_size + spacing)
        y_offset = spacing + row * (cell_size + spacing)

        raw_image = digits_images[sample_idx]
        max_value = float(raw_image.max()) or 1.0
        image = Image.fromarray((raw_image / max_value * 255).astype(np.uint8), mode="L")
        tile = image.resize((cell_size, cell_size), resample=Image.NEAREST).convert("RGB")

        correct = int(predictions[slot]) == int(labels[slot])
        border_color = palette["correct" if correct else "incorrect"]

        bordered = Image.new("RGB", (cell_size + 8, cell_size + 8), color=border_color)
        bordered.paste(tile, (4, 4))
        canvas.paste(bordered, (x_offset - 4, y_offset - 4))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_artifacts(
    config: TrainingConfig,
    dataset: DatasetInfo,
    history: List[TrainingMetrics],
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_indices: np.ndarray,
) -> Dict[str, Mapping[str, object]]:
    """Create JSON-serialisable payloads for the dashboard."""

    if not history:
        raise ValueError("Training history is empty; increase --epochs above zero.")

    metrics = {
        "train_accuracy": float(history[-1].train_accuracy),
        "val_accuracy": float(history[-1].val_accuracy),
        "test_accuracy": float(accuracy(test_probs, test_labels)),
        "train_loss": float(history[-1].train_loss),
        "val_loss": float(history[-1].val_loss),
        "test_loss": float(cross_entropy(test_probs, test_labels)),
        "num_epochs": int(config.epochs),
    }

    predictions_payload = {
        "dataset": dataset.name,
        "split": "test",
        "num_classes": dataset.num_classes,
        "samples": [],
    }

    for index, (probs, label, dataset_idx) in enumerate(
        zip(test_probs, test_labels, test_indices)
    ):
        sample = {
            "index": int(index),
            "dataset_index": int(dataset_idx),
            "true_label": int(label),
            "predicted_label": int(np.argmax(probs)),
            "probabilities": [float(p) for p in probs.tolist()],
        }
        predictions_payload["samples"].append(sample)

    loss_curves_payload = {
        "epochs": [int(idx + 1) for idx in range(len(history))],
        "train_loss": [float(entry.train_loss) for entry in history],
        "train_accuracy": [float(entry.train_accuracy) for entry in history],
        "val_loss": [float(entry.val_loss) for entry in history],
        "val_accuracy": [float(entry.val_accuracy) for entry in history],
    }

    metadata_payload = {
        "run_name": config.run_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": asdict(dataset),
        "training_config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
        },
        "artifacts": {
            "metrics": ARTIFACT_FILES["metrics"],
            "predictions": ARTIFACT_FILES["predictions"],
            "loss_curves": ARTIFACT_FILES["loss_curves"],
            "gallery": "gallery.png",
        },
    }

    return {
        "metrics": metrics,
        "predictions": predictions_payload,
        "loss_curves": loss_curves_payload,
        "metadata": metadata_payload,
    }


def write_artifacts(output_dir: Path, artifacts: Mapping[str, Mapping[str, object]]) -> None:
    """Validate and serialize the JSON artifacts."""

    validators = {
        "metrics": validate_metrics_payload,
        "predictions": validate_predictions_payload,
        "loss_curves": validate_loss_curves_payload,
        "metadata": validate_metadata_payload,
    }

    for key, payload in artifacts.items():
        validator = validators.get(key)
        if validator is None:
            raise KeyError(f"Unknown artifact '{key}'")
        validator(payload)
        filename = ARTIFACT_FILES[key]
        with (output_dir / filename).open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)


def prepare_features(features: np.ndarray) -> np.ndarray:
    """Normalise features and add a small bias term for stability."""

    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    return (features - mean) / std


def build_run_name(seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"digits-softmax-{timestamp}-seed{seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file containing argument overrides.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where artifacts will be written (overrides config/default).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Mini-batch size for gradient descent."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Gradient descent learning rate."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional identifier for the run; autogenerated when omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_training_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config.seed)
    np.random.seed(config.seed)

    digits = load_digits()
    features = prepare_features(digits.data.astype(np.float64))
    labels = digits.target.astype(np.int64)
    indices = np.arange(labels.size)

    x_train, x_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        features,
        labels,
        indices,
        test_size=0.2,
        random_state=config.seed,
        stratify=labels,
    )

    x_val, x_test, y_val, y_test, idx_val, idx_test = train_test_split(
        x_temp,
        y_temp,
        idx_temp,
        test_size=0.5,
        random_state=config.seed,
        stratify=y_temp,
    )

    weights, biases, history = train_model(config, x_train, y_train, x_val, y_val)

    test_logits = x_test @ weights + biases
    test_probs = softmax(test_logits)

    dataset_info = DatasetInfo(
        name="sklearn_digits",
        num_classes=int(labels.max() + 1),
        num_features=features.shape[1],
        image_shape=digits.images.shape[1:3],
        train_size=int(y_train.size),
        val_size=int(y_val.size),
        test_size=int(y_test.size),
    )

    artifacts = build_artifacts(config, dataset_info, history, test_probs, y_test, idx_test)

    write_artifacts(config.output_dir, artifacts)

    # Save gallery using the first N test samples.
    render_gallery(
        config.output_dir / "gallery.png",
        digits.images,
        idx_test,
        test_probs.argmax(axis=1),
        y_test,
    )

    print(f"Artifacts written to {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
