# Makefile for Gradio Q&A App

# Variables
APP = src/app/main.py
PYTHON = python3
VENV = venv
REQ = requirements.txt

.PHONY: help venv install run test lint format clean

help:
	@echo "Available commands:"
	@echo "  make venv      - Create virtual environment"
	@echo "  make install   - Install dependencies"
	@echo "  make run       - Run the Gradio app"
	@echo "  make test      - Run tests with pytest + coverage"
	@echo "  make lint      - Run linting with ruff"
	@echo "  make format    - Auto-format with ruff"
	@echo "  make clean     - Remove virtual environment and cache"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r $(REQ) && pip install ruff pytest pytest-cov

run:
	. $(VENV)/bin/activate && $(PYTHON) $(APP)

test:
	. $(VENV)/bin/activate && PYTHONPATH=src pytest --cov=src --cov-report=term-missing -v

lint:
	. $(VENV)/bin/activate && ruff check .

format:
	. $(VENV)/bin/activate && ruff check --fix .

clean:
	rm -rf $(VENV) __pycache__ *.pyc .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
