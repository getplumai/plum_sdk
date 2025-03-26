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
	@python setup.py sdist

publish:
	@twine upload dist/*

all: test build
