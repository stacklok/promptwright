# Makefile
.PHONY: test test-all lint

test:
	pytest -v --cov=promptweaver --cov-report=xml

test-all:
	pytest -v

lint:
	ruff check .
	ruff format --check .
