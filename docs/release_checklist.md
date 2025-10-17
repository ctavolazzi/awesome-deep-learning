# Release Checklist

Use this checklist to prepare the digit-classifier MVP for a local test drive or a
formal review. Each step assumes you are inside the repository root.

## 0. Verify your environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r examples/digit-classifier/requirements.txt
```

*Ensures reviewers execute the demo with the pinned dependency set before
regenerating artifacts or running tests.*

## 1. Refresh artifacts

```bash
python examples/digit-classifier/run_demo.py --artifact-root examples/digit-classifier/web_demo/artifacts
```

*Generates a new timestamped run, updates `latest/`, and rewrites
`index.json` so the dashboard reflects your latest experiment.*

## 2. Optional: seed synthetic data

```bash
python examples/digit-classifier/web_demo/seed_sample_artifacts.py
```

*Creates two representative runs without external dependencies—perfect when
reviewers cannot install scikit-learn or matplotlib.*

## 3. Launch the dashboard

```bash
cd examples/digit-classifier/web_demo
python -m http.server 8000
```

*Serve the static assets and open `http://localhost:8000/` in your browser.
The dashboard auto-loads the freshest run; override with
`?artifacts=<relative-path>` if needed.*

## 4. Run the smoke tests

```bash
pytest tests/test_run_demo.py
```

*Validates that the artifact pipeline and seeding helper produce the expected
bundle even when optional dependencies are missing.*

## 5. Capture a preview (optional)

Use your favorite screenshot tool or an automated browser (e.g., Playwright) to
capture the dashboard once artifacts are in place. Attach the image to your PR
so reviewers can see the experience without spinning it up.

---

Check off each step before opening the final pull request to keep the MVP demo
stable and reviewer-friendly.
