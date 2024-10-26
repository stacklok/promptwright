# Makefile
.PHONY: test test-integration test-all lint

test:
	pytest -v -m "not integration" --cov=promptsmith --cov-report=xml

test-integration:
	pytest -v -m integration

test-all:
	pytest -v

lint:
	ruff check .
	ruff format --check .
