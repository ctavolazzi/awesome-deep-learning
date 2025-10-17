# Digit Classifier Demo

This example trains a lightweight multinomial logistic regression model on the
scikit-learn digits dataset using a pure NumPy training loop.  The script is
intended for quick experiments and doubles as a showcase for the analytics
artifacts that can be generated alongside the trained model metrics.

## Usage

```bash
python examples/digit-classifier/run_demo.py \
  --output-dir artifacts/digits \
  --epochs 120 \
  --learning-rate 0.12
```

Pass `-h/--help` to see all available flags together with their default values.

Running without any optional flags keeps the workflow fast and produces:

* `metrics.json` – aggregate accuracy numbers plus a classification report.
* `confusion_matrix.png` – confusion matrix visualization for the hold-out split.

The script prints a JSON-formatted summary to stdout containing the final train
and test accuracy together with the list of generated artifact filenames.

## Extensibility

The training and evaluation workflow is implemented in
[`pipeline.py`](./pipeline.py).  The module defines small interfaces that make
it easy to swap individual pieces of the experiment:

* **Dataset loaders** implement the `DatasetLoader` protocol and only need to
  expose a `load(test_size=..., random_state=...)` method that returns a
  `DatasetSplit` object.  The included `DigitsDatasetLoader` demonstrates how to
  normalize inputs and preserve class names.
* **Model builders** implement the `ModelBuilder` protocol and are responsible
  for constructing a `TrainableModel` tailored to the feature and class
  dimensions of the dataset.  `SoftmaxGDBuilder` instantiates the provided
  gradient-descent-based softmax classifier, but custom architectures can be
  injected without modifying the CLI.
* **Training reports** produced by `TrainableModel.fit` must include a
  `ProbabilisticClassifier` capable of `predict` and `predict_proba`.  The
  pipeline relies on these methods to drive all analytics (metrics, ROC curves,
  and exported figures), so alternative models should expose the same
  prediction surface.

`run_demo.py` wires these components together via `run_training_pipeline`, which
handles fitting the model, generating predictions, and computing headline
metrics.  Additional analytics hooks consume the resulting `PipelineResult`, so
alternative implementations only need to honor the pipeline interfaces to reuse
the reporting stack.

## Expanded analytics

Heavier diagnostics can be toggled on individually via CLI flags so that the
common path remains lightweight:

* `--roc-per-class` generates per-class ROC curves, stores their AUC scores in
  `roc_curves.json`, and exports an overview plot in `roc_curves.png`.
* `--learning-rate-trace` captures the learning-rate schedule and loss values
  observed during optimization.  The data are stored in
  `training_dynamics.json` and visualized in `learning_rate_trace.png` with a
  dual-axis plot.
* `--timing-stats` records epoch-level timings.  A structured summary is saved
  in `timing_stats.json`, and `timing_stats.png` plots the duration of each
  training epoch.

When a flag is omitted the corresponding computation is skipped, avoiding the
extra work of generating large intermediate arrays or Matplotlib figures.

All artifacts are written to the directory specified by `--output-dir` (which
defaults to `./artifacts`).  This keeps the new analytics assets colocated with
existing outputs, allowing dashboards or notebooks that already watch the
artifact directory to surface the richer context automatically.
