VENV := $(PWD)/.venv
PYTHON = $(VENV)/bin/python
PIP := $(PYTHON) -m pip

.PHONY: help clean clean-build clean-pyc clean-env env env-dev lint test package install upload

help:	# The following lines will print the available commands when entering just 'make'
ifeq ($(UNAME), Linux)
	@grep -P '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
else
	@awk -F ':.*###' '$$0 ~ FS {printf "%15s%s\n", $$1 ":", $$2}' \
		$(MAKEFILE_LIST) | grep -v '@awk' | sort
endif

clean: clean-build clean-pyc ### Clean build binaries

clean-build: ### Removes builds
	find . -type d -iname "build" ! -path "./.venv/*" -exec rm -rf {} +
	find . -type d -iname "dist" ! -path "./.venv/*" -exec rm -rf {} +
	find . -type d -iname "*.egg-info" ! -path "./.venv/*" -exec rm -rf {} +

clean-pyc: ### Removes python compiled bytecode files
	find . -iname "*.pyc" ! -path "./.venv/*" -delete
	find . -type d -iname "__pycache__" ! -path "./.venv/*" -exec rm -rf {} +

clean-env: ### Removes environment directory
	rm -rf $(VENV)

env: ### Create a virtual environment
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

env-dev: env install-deps

install-deps:
	$(PIP) install -r requirements-dev.txt

lint: ### Validates project with linting rules
	$(PYTHON) -m prospector src/

test: ### Runs all the project tests
	$(PYTHON) -m pytest tests/

package: clean ### Runs the project setup
	echo __version__ = \"$(version)\" > src/ydata_quality/__version__.py
	echo "$(version)" > VERSION
	$(PYTHON) setup.py sdist bdist_wheel

install: ### Installs required dependencies
	$(PIP) install dist/ydata-quality-$(version).tar.gz

link-local: ### Installs the lib in dev mode
	echo __version__ = \"$(version)\" > src/ydata_quality/__version__.py
	$(PIP) install -e .

upload:
	$(PYTHON) -m twine upload dist/*
