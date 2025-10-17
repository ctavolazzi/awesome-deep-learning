# Digit Classifier Experience — MVP Blueprint

## 1. Product framing
- **Audience**: data science learners and ML practitioners who want a guided, hands-on tour of a classic computer-vision workflow without heavy infrastructure.
- **Problem it solves**: bridging the gap between a training script and an exploratory UI so users can run a lightweight experiment and immediately review their own results.
- **Success criteria**:
  1. A non-expert can install dependencies, run one command, and load a dashboard that reflects their run without editing code.
  2. Key performance signals (accuracy, confusion matrix, per-sample gallery) update automatically based on the latest training artifacts.
  3. Documentation provides a 15-minute quickstart from clone to insight.

## 2. MVP capabilities
1. **Run orchestration**
   - Single CLI (`python examples/digit-classifier/run_demo.py`) that trains, evaluates, and saves artifacts in a predictable directory.
   - Deterministic artifact layout (`artifacts/<timestamp>/...`) for dashboards and future tooling.
2. **Artifact bundle**
   - `run_summary.json`: metadata (hyperparameters, timestamps, dataset splits, scoring metrics).
   - `metrics.json`: aggregate metrics (accuracy, precision/recall per class).
   - `predictions.json`: per-sample predictions with ground truth, probabilities, and optional image path.
   - `loss_curve.json`: training + validation loss over epochs.
   - Image assets: confusion-matrix heatmap, sample gallery thumbnails.
3. **Dashboard**
   - Static web app (`examples/digit-classifier/web_demo`) that reads artifacts via relative paths.
   - Auto-populated sections: overview KPIs, confusion matrix, sample inspector, experiment metadata.
   - Lightweight runtime toggle between the most recent run and archived runs.
4. **Documentation & onboarding**
   - Step-by-step setup in README plus troubleshooting for common issues.
   - Explanation of artifact schema so teams can extend or integrate with other tools.
   - Quickstart video/gif or screenshot showing the MVP outcome.
5. **Automation hooks (stretch)**
   - Optional script/Make target to serve the dashboard (e.g., `python -m http.server`).
   - CI smoke check (lint + `python -m compileall`) to keep repo healthy.

## 3. Technical requirements
- **Python pipeline**
  - Use scikit-learn for the baseline model; ensure deterministic seeds and configurable CLI flags.
  - Validate output directories, emit schema version in each JSON, and provide graceful error messages on failure.
  - Unit-style test for the artifact writer (e.g., verifying schema fields) runnable without heavy dependencies.
- **Front-end**
  - Pure static assets (HTML/CSS/JS) with fetch-based data loading and basic state management (no heavy framework in MVP).
  - Schema validation and fallback messaging when artifacts are missing or outdated.
  - Responsive layout that works on desktop/tablet widths; accessibility baseline (ARIA labels for interactive elements).
- **Project hygiene**
  - `examples/digit-classifier/requirements.txt` pinned to minor versions with hashes (or instructions for reproducible install).
  - Documented folder conventions for artifacts and how to clean them (`--output-dir`, `--keep-artifacts`).
  - Versioned changelog entry describing MVP milestone.

## 4. Deliberate simplifications
- No backend server or database—artifacts stay on disk; the dashboard is static.
- Single-model focus (MLPClassifier) until user feedback warrants alternatives.
- Manual run triggering; no need for scheduler/queue in MVP.
- Keep visualization assets minimal (JSON + PNG) before investing in richer graphics libraries.

## 5. Risks & mitigations
- **Stale artifacts**: add timestamp + schema version to catch incompatible runs.
- **Dependency friction**: provide `pipx`/virtualenv guidance and preflight checks in the CLI.
- **Large bundles**: cap gallery size or allow `--sample-count` to keep disk footprint manageable.

## 6. Roadmap toward MVP
1. Finalize JSON schemas and implement serialization inside `run_demo.py` (in progress, ensure tests cover it).
2. Refactor dashboard data loading to read the full artifact bundle and expose run-selection UI.
3. Harden documentation with architecture diagram and troubleshooting appendix.
4. Add automation: linting + compile check in CI, Make targets for `train` and `serve`.
5. Capture a guided walkthrough (gif/screenshot) for README once experience stabilizes.

## 7. Expert recommendations
- **Tighten artifact-to-UI contract**: Introduce TypeScript-like schema definitions (e.g., using JSDoc typedefs) to reduce front-end regressions.
- **Invest in modular JS**: Split current `script.js` into data, state, and view modules even before adopting a bundler—clarifies ownership and enables targeted tests.
- **Incremental testing**: Add Python unit tests (e.g., `pytest`) focusing on artifact generation; adopt Vitest for UI state logic after modularization.
- **Telemetry-friendly structure**: When ready, you can swap local artifacts for API responses with minimal churn if the schema remains stable.
- **Design for collaboration**: Document contribution guidelines (coding standards, review checklist) so future contributors can evolve the MVP without rework.

## 8. MVP delivery snapshot
- Artifact schema versioning, timestamped directories, and a `latest/` mirror now ship out-of-the-box from `run_demo.py`.
- The static dashboard consumes `index.json`, supports run selection, and reads the dynamic JSON/PNG bundle rather than hard-coded placeholders.
- Documentation, dependency pins, and smoke tests cover the end-to-end workflow so contributors can reproduce the MVP confidently.

