# Digit Classifier Demo

This miniature project provides a reproducible way to explore a classic
supervised-learning workflow with a compact neural network. It relies only on
the batteries-included [scikit-learn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html),
so it runs quickly on CPU-only machines and requires no dataset downloads.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r examples/digit-classifier/requirements.txt
python examples/digit-classifier/run_demo.py --run-label baseline
```

The script prints the test-set accuracy and emits a bundle of artifacts under
`examples/digit-classifier/web_demo/artifacts/` by default. Each run receives a
timestamped directory (optionally suffixed by `--run-label`), and the dashboard
keeps a copy in `artifacts/latest/` for quick sharing.

Artifacts captured per run:

- `metrics.json` with accuracy, macro F1, and the full classification report.
- `run_summary.json` capturing configuration, dataset sizes, and timing.
- `predictions.json` with per-sample labels, confidences, and flattened pixels.
- `loss_curve.json` storing the training loss history for visualization.
- `sample_gallery.png` – a tiled gallery suited for slide decks.
- `confusion_matrix.png` – a ready-to-share heatmap of the hold-out results.

An `index.json` file in the artifacts root tracks run history so the web UI can
offer a selector. A typical layout looks like this:

```
examples/digit-classifier/web_demo/artifacts/
├── 20240305-210455-baseline/
│   ├── confusion_matrix.png
│   ├── loss_curve.json
│   ├── metrics.json
│   ├── predictions.json
│   ├── run_summary.json
│   └── sample_gallery.png
├── 20240305-205832-experiment-2/
│   └── ...
├── index.json
└── latest/  # mirror of the most recent run
```

## Visual dashboard

Prefer to explore the project in your browser? Open the
[`web_demo`](./web_demo/) front-end to browse curated digit samples, inspect the
representative training metrics, and follow the step-by-step workflow.

### Syncing the dashboard with your latest run

The dashboard reads JSON artifacts directly from disk. When you run the demo,
`latest/` is refreshed automatically and `index.json` is updated so the
**Artifact bundle** selector lists every saved run. Serve the `web_demo/`
folder (see below), refresh your browser, and pick the run you want to inspect.

Prefer to store artifacts elsewhere? Pass a relative path via
`?artifacts=../my-artifacts/20240305-210455-baseline` in the dashboard URL or
update the `data-artifact-path` attribute on the `<body>` element. The selector
will still light up runs listed in the `index.json` that lives alongside your
artifacts.

### Previewing the dashboard without third-party dependencies

Need to demo the UI from a restricted environment where you cannot install
`scikit-learn` or `matplotlib`? Seed a synthetic artifact bundle with the
standard-library helper:

```bash
python examples/digit-classifier/web_demo/seed_sample_artifacts.py
```

The script writes two curated runs, refreshes `latest/`, and updates
`index.json` so every widget remains interactive. You can regenerate the sample
data at any time—existing synthetic directories are overwritten automatically.

### Launching the dashboard

```bash
# from the repository root
python -m http.server 8000
```

Then visit [http://localhost:8000/examples/digit-classifier/web_demo/](http://localhost:8000/examples/digit-classifier/web_demo/)
in your browser. The page is fully static, so you can also double-click
`index.html` to open it directly from disk.

### What you will see

* A **dataset studio** rendered from the same 8×8 grayscale format that the
  training script consumes, complete with filtering, shuffling, and a download
  button for the selected sample.
* A **model insights wall** summarizing architecture, accuracy, macro F1, and
  convergence behaviour with an animated loss curve and interactive confusion
  matrix.
* An **experiment playset** that highlights the run summary emitted by the
  script and a **workflow playbook** mapping each step to the relevant files so
  you can extend the example with confidence.

## Customization

Key parameters are exposed as command-line options:

* `--hidden-sizes`: choose the multi-layer perceptron architecture. For example,
  `--hidden-sizes 128,64` uses two hidden layers with 128 and 64 units.
* `--test-size`: adjust the evaluation split (default: `0.2`).
* `--sample-count`: control how many predictions are visualized (default: `16`).
* `--random-state`: seed for reproducibility (default: `42`).
* `--run-label`: append a human-readable suffix to the artifact directory.
* `--artifacts-root`: change where artifact bundles and `index.json` are stored.

Each run is deterministic for a given seed, so you can easily compare different
network shapes or train/test splits.

## Next steps

Feel free to extend this skeleton by swapping in your favorite dataset,
experimenting with different model architectures, or piping the saved metrics
into visualization dashboards such as TensorBoard or Netron—both linked from
our curated list.
This example trains a lightweight multinomial logistic regression model on the
scikit-learn digits dataset using a pure NumPy training loop.  The script is
intended for quick experiments and doubles as a showcase for the analytics
artifacts that can be generated alongside the trained model metrics.

## Usage

### Quick start

```bash
python examples/digit-classifier/run_demo.py
```

The script resolves its defaults from `config.yaml` (see [Configuration](#configuration)) and reports a
JSON-formatted summary to stdout containing the final train/test accuracy together with the list of
generated artifact filenames.

Running without additional flags keeps the workflow fast and produces:

* `metrics.json` – aggregate accuracy numbers plus a classification report.
* `confusion_matrix.png` – confusion matrix visualization for the hold-out split.

### Command-line reference

All runtime knobs can be tweaked from the CLI and override the YAML configuration on a
field-by-field basis.  The most commonly used options are summarized below:

| Flag | Purpose |
| --- | --- |
| `--config PATH` | Load settings from a YAML file (defaults to `config.yaml` next to the script). |
| `--output-dir DIR` | Store generated metrics and plots under `DIR`. |
| `--epochs INT` | Number of training epochs to perform. |
| `--learning-rate FLOAT` | Initial learning rate used by gradient descent. |
| `--lr-decay FLOAT` | Linear learning-rate decay applied at each epoch. |
| `--test-split FLOAT` | Fraction of the dataset reserved for evaluation. |
| `--seed INT` | Random seed controlling the train/test split. |
| `--[no-]roc-per-class` | Toggle per-class ROC generation regardless of the config file. |
| `--[no-]learning-rate-trace` | Toggle persistence of the learning-rate schedule. |
| `--[no-]timing-stats` | Toggle export of epoch-level timing diagnostics. |

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
