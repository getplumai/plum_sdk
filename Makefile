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
	@echo "make all          test and build"

.PHONY: test build publish

test:
	@pytest -v tests

build:
	@python -m black .
	@mkdir -p dist
	@rm -f dist/*.tar.gz || true
	@python -m build --sdist
	@echo "Built package. Check that it contains all required files with: tar -tvf dist/*.tar.gz"

publish:
	@echo "Publishing to PyPI..."
	@python -m twine upload dist/*

all: test build
