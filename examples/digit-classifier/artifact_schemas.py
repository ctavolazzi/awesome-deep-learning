"""Validation helpers for digit-classifier demo artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass
class SchemaError(ValueError):
    """Exception raised when an artifact payload does not match expectations."""

    path: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.path}: {self.message}"


def _require_keys(mapping: Mapping[str, Any], keys: Iterable[str], path: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise SchemaError(path, f"missing keys: {', '.join(sorted(missing))}")


def _require_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(path, f"expected number, received {type(value).__name__}")
    return float(value)


def _require_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(path, f"expected integer, received {type(value).__name__}")
    return int(value)


def _require_sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(path, f"expected sequence, received {type(value).__name__}")
    return value


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise SchemaError(path, f"expected string, received {type(value).__name__}")
    return value


def _normalise_probabilities(probabilities: Sequence[Any], path: str) -> Sequence[float]:
    result = []
    for idx, entry in enumerate(probabilities):
        result.append(_require_number(entry, f"{path}[{idx}]"))
    return result


def _require_number_sequence(value: Any, path: str) -> Sequence[float]:
    sequence = _require_sequence(value, path)
    return [_require_number(entry, f"{path}[{idx}]") for idx, entry in enumerate(sequence)]


def validate_metrics_payload(payload: Mapping[str, Any]) -> None:
    """Validate the structure of ``metrics.json``."""

    _require_keys(payload, {
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
        "train_loss",
        "val_loss",
        "test_loss",
        "num_epochs",
    }, "metrics")

    for key in [
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
        "train_loss",
        "val_loss",
        "test_loss",
    ]:
        _require_number(payload[key], f"metrics.{key}")

    _require_int(payload["num_epochs"], "metrics.num_epochs")


def validate_predictions_payload(payload: Mapping[str, Any]) -> None:
    """Validate the structure of ``predictions.json``."""

    _require_keys(payload, {"dataset", "split", "num_classes", "samples"}, "predictions")
    _require_int(payload["num_classes"], "predictions.num_classes")
    samples = _require_sequence(payload["samples"], "predictions.samples")

    for idx, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise SchemaError(f"predictions.samples[{idx}]", "expected object")
        _require_keys(
            sample,
            {"index", "dataset_index", "true_label", "predicted_label", "probabilities"},
            f"predictions.samples[{idx}]",
        )
        _require_int(sample["index"], f"predictions.samples[{idx}].index")
        _require_int(sample["dataset_index"], f"predictions.samples[{idx}].dataset_index")
        _require_int(sample["true_label"], f"predictions.samples[{idx}].true_label")
        _require_int(sample["predicted_label"], f"predictions.samples[{idx}].predicted_label")
        probs = _normalise_probabilities(
            _require_sequence(sample["probabilities"], f"predictions.samples[{idx}].probabilities"),
            f"predictions.samples[{idx}].probabilities",
        )
        if len(probs) != payload["num_classes"]:
            raise SchemaError(
                f"predictions.samples[{idx}].probabilities",
                f"expected {payload['num_classes']} entries, received {len(probs)}",
            )


def validate_loss_curves_payload(payload: Mapping[str, Any]) -> None:
    """Validate the structure of ``loss_curves.json``."""

    _require_keys(
        payload,
        {"epochs", "train_loss", "train_accuracy", "val_loss", "val_accuracy"},
        "loss_curves",
    )

    epochs = [
        _require_int(epoch, f"loss_curves.epochs[{idx}]")
        for idx, epoch in enumerate(_require_sequence(payload["epochs"], "loss_curves.epochs"))
    ]

    for key in ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]:
        series = [
            _require_number(value, f"loss_curves.{key}[{idx}]")
            for idx, value in enumerate(_require_sequence(payload[key], f"loss_curves.{key}"))
        ]
        if len(series) != len(epochs):
            raise SchemaError(
                f"loss_curves.{key}",
                "expected the same number of entries as epochs",
            )


def validate_metadata_payload(payload: Mapping[str, Any]) -> None:
    """Validate the structure of ``run_metadata.json``."""

    _require_keys(
        payload,
        {"run_name", "generated_at", "dataset", "training_config", "feature_normalization", "artifacts"},
        "run_metadata",
    )

    dataset = payload["dataset"]
    if not isinstance(dataset, Mapping):
        raise SchemaError("run_metadata.dataset", "expected object")
    _require_keys(dataset, {"name", "num_classes", "num_features", "image_shape", "train_size", "val_size", "test_size"}, "run_metadata.dataset")
    _require_int(dataset["num_classes"], "run_metadata.dataset.num_classes")
    num_features = _require_int(dataset["num_features"], "run_metadata.dataset.num_features")
    _require_sequence(dataset["image_shape"], "run_metadata.dataset.image_shape")
    _require_int(dataset["train_size"], "run_metadata.dataset.train_size")
    _require_int(dataset["val_size"], "run_metadata.dataset.val_size")
    _require_int(dataset["test_size"], "run_metadata.dataset.test_size")

    training = payload["training_config"]
    if not isinstance(training, Mapping):
        raise SchemaError("run_metadata.training_config", "expected object")
    _require_keys(training, {"epochs", "batch_size", "learning_rate", "seed"}, "run_metadata.training_config")
    _require_int(training["epochs"], "run_metadata.training_config.epochs")
    _require_int(training["batch_size"], "run_metadata.training_config.batch_size")
    _require_number(training["learning_rate"], "run_metadata.training_config.learning_rate")
    _require_int(training["seed"], "run_metadata.training_config.seed")

    normalization = payload["feature_normalization"]
    if not isinstance(normalization, Mapping):
        raise SchemaError("run_metadata.feature_normalization", "expected object")
    _require_keys(
        normalization,
        {"method", "stats_source", "mean", "std"},
        "run_metadata.feature_normalization",
    )
    _require_string(normalization["method"], "run_metadata.feature_normalization.method")
    _require_string(normalization["stats_source"], "run_metadata.feature_normalization.stats_source")
    means = _require_number_sequence(normalization["mean"], "run_metadata.feature_normalization.mean")
    stds = _require_number_sequence(normalization["std"], "run_metadata.feature_normalization.std")
    if len(means) != len(stds):
        raise SchemaError(
            "run_metadata.feature_normalization.std",
            "expected std sequence to match mean sequence length",
        )
    if not means:
        raise SchemaError(
            "run_metadata.feature_normalization.mean",
            "expected at least one feature statistic",
        )
    if len(means) != num_features:
        raise SchemaError(
            "run_metadata.feature_normalization.mean",
            f"expected {num_features} entries, received {len(means)}",
        )

    artifacts = payload["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise SchemaError("run_metadata.artifacts", "expected object")
    _require_keys(
        artifacts,
        {"metrics", "predictions", "loss_curves", "gallery"},
        "run_metadata.artifacts",
    )


__all__ = [
    "SchemaError",
    "validate_metrics_payload",
    "validate_predictions_payload",
    "validate_loss_curves_payload",
    "validate_metadata_payload",
]
