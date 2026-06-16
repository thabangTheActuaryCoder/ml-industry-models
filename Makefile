.PHONY: install generate train train-banking train-insurance train-retail train-mining clean all

PYTHON ?= python3
BASE_DIR := $(shell pwd)

install:
	$(PYTHON) -m pip install -r requirements.txt

# --- Data Generation ---
generate-banking:
	$(PYTHON) banking/src/generate_data.py

generate-insurance:
	$(PYTHON) insurance/src/generate_data.py

generate-retail:
	$(PYTHON) retail/src/generate_data.py

generate-mining:
	$(PYTHON) mining/src/generate_data.py

generate: generate-banking generate-insurance generate-retail generate-mining

# --- Training ---
train-banking:
	$(PYTHON) banking/src/train.py

train-insurance:
	$(PYTHON) insurance/src/train.py

train-retail:
	$(PYTHON) retail/src/train.py

train-mining:
	$(PYTHON) mining/src/train.py

train: train-banking train-insurance train-retail train-mining

# --- Full Pipeline ---
all: install generate train

# --- Clean ---
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f banking/data/*.csv banking/artefacts/*
	rm -f insurance/data/*.csv insurance/artefacts/*
	rm -f retail/data/*.csv retail/artefacts/*
	rm -f mining/data/*.csv mining/artefacts/*
