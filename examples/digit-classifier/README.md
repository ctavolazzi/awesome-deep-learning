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

Running without any optional flags keeps the workflow fast and produces:

* `metrics.json` – aggregate accuracy numbers plus a classification report.
* `confusion_matrix.png` – confusion matrix visualization for the hold-out split.

The script prints a JSON-formatted summary to stdout containing the final train
and test accuracy together with the list of generated artifact filenames.

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

## Artifact validation

Each JSON artifact is validated before it is written to disk.  The helper
functions in `run_demo.py` check for required fields, numeric ranges, and shape
consistency so that downstream tooling can rely on a stable schema.  The
integration test in `tests/test_digit_classifier_artifacts.py` loads the module
directly and exercises the validators against the generated files.

When adding a new export, define a corresponding `validate_*` helper near the
top of `run_demo.py` and invoke it just before writing the JSON payload.  This
keeps the guarantees close to the data producer and makes the validation logic
reusable in other scripts or tests.  For more complex scenarios you can follow
the same pattern but replace the custom checks with a Pydantic model.
