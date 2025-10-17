PYTHON ?= python
DIGIT_SCRIPT := examples/digit-classifier/run_demo.py
DIGIT_CONFIG := examples/digit-classifier/config.yaml

.PHONY: digit-demo digit-dashboard

## Run the digit classifier demo using the shared config defaults.
digit-demo:
	$(PYTHON) $(DIGIT_SCRIPT) --config $(DIGIT_CONFIG) $(ARGS)

## Run the digit classifier demo with all analytics enabled for dashboards.
digit-dashboard:
	$(PYTHON) $(DIGIT_SCRIPT) --config $(DIGIT_CONFIG) --roc-per-class --learning-rate-trace --timing-stats $(ARGS)
