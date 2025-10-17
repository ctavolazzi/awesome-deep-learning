# Digit Classifier Demo

This example trains a tiny softmax regression model on the
[`sklearn.datasets.load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
corpus and ships a static dashboard for exploring the resulting metrics.
The workflow is intentionally lightweight so that you can iterate on the
model without needing a full-featured experiment tracker.

## Requirements

The Python demo uses only NumPy, Pillow, and scikit-learn in addition to
the standard library.  Install them with:

```bash
pip install numpy pillow scikit-learn
```

The web dashboard is plain HTML/JS and can be served by any static file
server.

## Running the demo

Generate a fresh set of artifacts by running:

```bash
python examples/digit-classifier/run_demo.py --output-dir artifacts/latest
```

Key options:

- `--epochs`, `--batch-size`, `--learning-rate` – tweak the training
  schedule for the softmax regression model.
- `--seed` – ensures reproducible dataset shuffling and initial weights.
- `--run-name` – optional, otherwise a timestamped identifier is used.
- `--config` – load overrides from a JSON file (see below) so that teams can
  share a reproducible training recipe.

The repository ships with `config.example.json` that mirrors the CLI
arguments.  Provide your own configuration file if you want to pin a
specific output directory or learning schedule:

```bash
python examples/digit-classifier/run_demo.py --config examples/digit-classifier/config.example.json
```

The command writes five artifacts into the requested directory:

| File | Description |
| ---- | ----------- |
| `metrics.json` | Aggregate metrics from the final model (loss/accuracy for each split plus the number of epochs).
| `predictions.json` | Per-sample predictions on the test split.  Every record includes the index, ground truth label, predicted label, and the full probability vector.
| `loss_curves.json` | Training and validation loss/accuracy for each epoch so the dashboard can render curves or tables.
| `run_metadata.json` | Metadata describing the dataset, run configuration, feature normalisation strategy, and the list of exported artifacts.
| `gallery.png` | A 5×5 grid of test examples coloured by correctness for a quick qualitative check.

All JSON files are indented and human-readable, making it easy to diff
runs or inspect them manually.  They are also validated against a small
set of schema helpers (`artifact_schemas.py`) before being written so
that front-end code can rely on stable shapes.

### Configuration and validation

- `artifact_schemas.py` centralises the schema checks for
  `metrics.json`, `predictions.json`, `loss_curves.json`, and
  `run_metadata.json`.  If you change an artifact, update both the
  schema helper and the dashboard loader.
- `config.example.json` shows how to define a repeatable training run.
  When a config file is supplied the CLI still takes precedence for any
  flags you pass explicitly.

The training script fits normalisation statistics (mean and standard
deviation) on the training split only and reuses them for validation and
test data.  This prevents data leakage from the held-out sets and is
documented in the metadata under `feature_normalization`.

### JSON schema overview

Below is an excerpt of the shapes you can expect from the generated
artifacts (field ordering may differ in the actual files):

```jsonc
// metrics.json
{
  "train_accuracy": 0.991,
  "val_accuracy": 0.978,
  "test_accuracy": 0.975,
  "train_loss": 0.034,
  "val_loss": 0.071,
  "test_loss": 0.089,
  "num_epochs": 40
}

// predictions.json
{
  "dataset": "sklearn_digits",
  "split": "test",
  "num_classes": 10,
  "samples": [
    {
      "index": 0,
      "dataset_index": 1287,
      "true_label": 8,
      "predicted_label": 8,
      "probabilities": [0.002, 0.001, ..., 0.968]
    }
  ]
}

// loss_curves.json
{
  "epochs": [1, 2, 3, ...],
  "train_loss": [0.64, 0.29, ...],
  "train_accuracy": [0.74, 0.88, ...],
  "val_loss": [0.71, 0.31, ...],
  "val_accuracy": [0.70, 0.85, ...]
}

// run_metadata.json
{
  "run_name": "digits-softmax-20240101-120000-seed13",
  "generated_at": "2024-01-01T12:00:00+00:00",
  "dataset": {
    "name": "sklearn_digits",
    "num_classes": 10,
    "num_features": 64,
    "image_shape": [8, 8],
    "train_size": 1147,
    "val_size": 287,
    "test_size": 287
  },
  "training_config": {
    "epochs": 40,
    "batch_size": 128,
    "learning_rate": 0.5,
    "seed": 13
  },
  "feature_normalization": {
    "method": "zscore",
    "stats_source": "train_split"
  },
  "artifacts": {
    "metrics": "metrics.json",
    "predictions": "predictions.json",
    "loss_curves": "loss_curves.json",
    "gallery": "gallery.png"
  }
}
```

## Refreshing the dashboard

The `web_demo/` directory contains a vanilla JavaScript app that reads
all of the JSON outputs produced by `run_demo.py`.  To preview it:

1. Re-run the training script to produce fresh artifacts (as shown
   above).
2. Copy or symlink the generated files so they live alongside the web
   assets.  The simplest approach is to place the dashboard and
   artifacts under a common directory, e.g.:

   ```bash
   mkdir -p examples/digit-classifier/dashboard
   cp -r examples/digit-classifier/web_demo/* examples/digit-classifier/dashboard/
   cp artifacts/latest/*.json artifacts/latest/gallery.png examples/digit-classifier/dashboard/
   ```

3. Serve that directory with a static server.  Python ships with one
   out of the box:

   ```bash
   python -m http.server --directory examples/digit-classifier/dashboard 9000
   ```

4. Open `http://localhost:9000/index.html` in your browser.  The
   dashboard automatically fetches `metrics.json`, `predictions.json`,
   `loss_curves.json`, and `run_metadata.json` from the same directory.

Whenever you re-run the training script, refresh the browser page (or
restart the server if you place the artifacts elsewhere) to see the new
results.  The dashboard disables HTTP caching when requesting the JSON
files, so a simple browser refresh is enough to pick up the latest
metrics.

## Testing the pipeline

The repository contains a lightweight regression test that exercises the
demo end-to-end and validates the emitted artifacts.  The test is marked
as skipped automatically when scikit-learn is unavailable (for example
in minimal CI containers).

```bash
python -m unittest tests.test_digit_classifier_artifacts
```

Running the demo manually remains the fastest way to sanity check the
dashboard, but the automated test protects the schema contract from
accidental regressions during development.
