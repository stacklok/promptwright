# Makefile
.PHONY: test test-all lint

test:
	pytest -v --cov=promptsmith --cov-report=xml

test-all:
	pytest -v

lint:
	ruff check .
	ruff format --check .
