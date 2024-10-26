# Makefile
.PHONY: test test-all lint

test:
	pytest -v --cov=promptwright --cov-report=xml

test-all:
	pytest -v

lint:
	ruff check .
	ruff format --check .
