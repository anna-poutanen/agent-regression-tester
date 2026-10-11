VENV_PYTHON := .venv/bin/python
VENV_PIP := .venv/bin/pip

.PHONY: setup check test demo build

setup:
	python3 -m venv .venv
	$(VENV_PIP) install -e '.[dev]'

check:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
	.venv/bin/mypy
	$(VENV_PYTHON) -m coverage run -m pytest
	$(VENV_PYTHON) -m coverage report

test:
	$(VENV_PYTHON) -m pytest

demo:
	.venv/bin/agent-regression demo --repeats 3

build:
	$(VENV_PYTHON) -m build
