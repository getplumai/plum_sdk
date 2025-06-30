##
# Makefile for the Plum SDK
# Author: prem@getplum.ai
# Date:   2025-03-24
# 
# Usage: `make help` to see available commands.
##


help:
	@echo "Makefile usage:"
	@echo "make test         run pytest unit tests"
	@echo "make build        build the sdist"
	@echo "make publish      publish to PyPI"
	@echo "make install-dev  install package in development mode and test imports"
	@echo "make all          test, build, install dev package"

.PHONY: test build publish install-dev

test:
	@pytest -v plum_sdk/tests

build:
	@python -m black plum_sdk
	@mkdir -p dist
	@rm -f dist/*.tar.gz || true
	@python -m build --sdist
	@echo "Built package. Check that it contains all required files with: tar -tvf dist/*.tar.gz"

publish:
	@echo "Publishing to PyPI..."
	@python -m twine upload dist/*

install-dev:
	@echo "Installing package in development mode..."
	@pip install -e .
	@echo "Testing imports..."
	@python -c "from plum_sdk import PlumClient, TrainingExample; print('Import test successful!')"
	@echo "Testing complete, uninstalling development package..."
	@pip uninstall -y plum-sdk
	@echo "Development package uninstalled."

all: test build install-dev
	@echo "All tasks completed successfully!"
	@echo "You can now publish the package to PyPI with: make publish"
	@echo "Or install it in development mode with: make install-dev"
	@echo "To run tests, use: make test"
	@echo "To build the package, use: make build"
	@echo "To see all available commands, use: make help"