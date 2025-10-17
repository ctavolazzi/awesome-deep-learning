# Digit Classifier Demo

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

## Configuration

The default configuration lives in [`config.yaml`](./config.yaml) and mirrors the CLI flags:

```yaml
output_dir: artifacts/digits
hyperparameters:
  epochs: 150
  learning_rate: 0.1
  lr_decay: 0.01
  test_split: 0.2
  seed: 13
analytics:
  roc_per_class: false
  learning_rate_trace: false
  timing_stats: false
```

Update this file to share reproducible experiment settings with teammates or keep separate
profiles for CI versus local exploration.  Command-line overrides always take precedence, so you
can perform ad-hoc tweaks without mutating the shared defaults.

## Automation tips

To streamline day-to-day experimentation, the repository includes `Makefile` targets that wrap the
common invocations:

```bash
make digit-demo       # Standard run using config.yaml defaults
make digit-dashboard  # Full analytics sweep with ROC, LR trace, and timing stats
```

Both targets accept additional CLI overrides via the `ARGS` variable, e.g. `make digit-demo ARGS="--epochs 50"`.
