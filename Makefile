PY := .venv/bin/python

.PHONY: all data macro arima features train lgd ecl stress validate report resume_numbers test
all: data macro arima features train lgd ecl stress validate report resume_numbers

data:
	$(PY) -m creditcast.data
macro:
	$(PY) -m creditcast.macro
	$(PY) -m creditcast.paths
arima:
	$(PY) -m creditcast.arima
features:
	$(PY) -m creditcast.features
train:
	$(PY) -m creditcast.pd_model
lgd:
	$(PY) -m creditcast.lgd_ead
ecl:
	$(PY) -m creditcast.ecl
stress:
	$(PY) -m creditcast.stress
validate:
	$(PY) -m creditcast.validate
report:
	$(PY) -m creditcast.report
resume_numbers:
	$(PY) -m creditcast.resume_numbers
test:
	$(PY) -m pytest -q
