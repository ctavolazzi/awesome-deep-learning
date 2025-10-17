"""Reusable training pipeline for the digit classifier example."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from sklearn.metrics import accuracy_score


@dataclass
class DatasetSplit:
    """Container holding train/test arrays for the classification task."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    class_names: Sequence[str]


@dataclass
class TrainingConfig:
    """Hyper-parameters steering the optimization procedure."""

    epochs: int
    base_learning_rate: float
    learning_rate_decay: float


@dataclass
class PipelineConfig:
    """Top-level configuration for the training/evaluation pipeline."""

    test_size: float
    random_state: int
    training: TrainingConfig


class DatasetLoader(Protocol):
    """Loads the dataset split used for training and evaluation."""

    def load(self, *, test_size: float, random_state: int) -> DatasetSplit:
        """Return the train/test split used for a training run."""


class ProbabilisticClassifier(Protocol):
    """Classifier that can produce both hard and probabilistic predictions."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return the most likely class index for each sample."""

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return class probability estimates with shape ``(n_samples, n_classes)``."""


class TrainableModel(Protocol):
    """Factory for producing a trained classifier together with diagnostics."""

    def fit(self, features: np.ndarray, labels: np.ndarray, config: TrainingConfig) -> "TrainingReport":
        """Run optimization and return a report containing the fitted classifier."""


class ModelBuilder(Protocol):
    """Builds model instances that can be trained by the pipeline."""

    def build(self, *, num_features: int, num_classes: int) -> TrainableModel:
        """Create a new trainable model tailored to the dataset dimensions."""


@dataclass
class TrainingReport:
    """Diagnostics captured while fitting a model."""

    model: ProbabilisticClassifier
    learning_rates: Sequence[float]
    losses: Sequence[float]
    epoch_timings: Sequence[float]
    total_time: float


@dataclass
class PipelineResult:
    """Aggregated outcome of a training/evaluation run."""

    dataset: DatasetSplit
    training: TrainingReport
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    test_probabilities: np.ndarray
    train_accuracy: float
    test_accuracy: float


class SoftmaxGDModel:
    """Multinomial logistic regression trained with batch gradient descent."""

    def __init__(self, *, num_features: int, num_classes: int) -> None:
        self.num_features = num_features
        self.num_classes = num_classes
        self.weights = np.zeros((num_features, num_classes), dtype=np.float64)
        self.bias = np.zeros(num_classes, dtype=np.float64)

    def fit(self, features: np.ndarray, labels: np.ndarray, config: TrainingConfig) -> TrainingReport:
        n_samples = features.shape[0]
        eye = np.eye(self.num_classes, dtype=np.float64)

        learning_rates = []
        losses = []
        epoch_timings = []

        start = time.perf_counter()
        for epoch in range(config.epochs):
            epoch_start = time.perf_counter()

            logits = features @ self.weights + self.bias
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)

            targets = eye[labels]
            grad = (probs - targets) / float(n_samples)
            grad_w = features.T @ grad
            grad_b = grad.sum(axis=0)

            current_lr = config.base_learning_rate / (1.0 + config.learning_rate_decay * epoch)
            self.weights -= current_lr * grad_w
            self.bias -= current_lr * grad_b

            loss = _softmax_cross_entropy(logits, labels)
            learning_rates.append(float(current_lr))
            losses.append(float(loss))
            epoch_timings.append(time.perf_counter() - epoch_start)

        total_time = time.perf_counter() - start
        return TrainingReport(self, learning_rates, losses, epoch_timings, total_time)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.weights + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)


class SoftmaxGDBuilder:
    """Factory for gradient-descent-based softmax classifiers."""

    def build(self, *, num_features: int, num_classes: int) -> TrainableModel:
        return SoftmaxGDModel(num_features=num_features, num_classes=num_classes)


def run_training_pipeline(
    dataset_loader: DatasetLoader,
    model_builder: ModelBuilder,
    config: PipelineConfig,
) -> PipelineResult:
    """Train and evaluate a classifier using the provided components."""

    dataset = dataset_loader.load(test_size=config.test_size, random_state=config.random_state)
    model = model_builder.build(num_features=dataset.x_train.shape[1], num_classes=len(dataset.class_names))
    training = model.fit(dataset.x_train, dataset.y_train, config.training)

    train_predictions = training.model.predict(dataset.x_train)
    test_probabilities = training.model.predict_proba(dataset.x_test)
    test_predictions = test_probabilities.argmax(axis=1)

    train_accuracy = accuracy_score(dataset.y_train, train_predictions)
    test_accuracy = accuracy_score(dataset.y_test, test_predictions)

    return PipelineResult(
        dataset=dataset,
        training=training,
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        test_probabilities=test_probabilities,
        train_accuracy=float(train_accuracy),
        test_accuracy=float(test_accuracy),
    )


def _softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = logits - logits.max(axis=1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    picked = log_probs[np.arange(logits.shape[0]), labels]
    return float(-picked.mean())
