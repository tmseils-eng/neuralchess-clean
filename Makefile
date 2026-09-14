.PHONY: help test lint perft train tiny play serve uci bench clean docker

PY ?= python3

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

test:  ## run the test suite (pytest if available, bundled runner otherwise)
	@$(PY) -c "import pytest" 2>/dev/null && $(PY) -m pytest -q || $(PY) scripts/run_tests.py

lint:  ## static checks
	@$(PY) -m ruff check neuralchess tests scripts 2>/dev/null || echo "ruff not installed; skipping"
	@$(PY) -m compileall -q neuralchess tests scripts

perft:  ## move generator correctness and speed
	$(PY) -m neuralchess perft --depth 5

tiny:  ## a few-minute training run to check the pipeline end to end
	$(PY) -m neuralchess train --config configs/tiny.json

train:  ## the laptop-scale run used for the reported results
	$(PY) -m neuralchess train --config configs/laptop.json

play:  ## play against the trained engine in the terminal
	$(PY) -m neuralchess play --checkpoint runs/laptop/checkpoints/best.npz

serve:  ## play against the trained engine in a browser
	$(PY) -m neuralchess serve --checkpoint runs/laptop/checkpoints/best.npz --open

uci:  ## run as a UCI engine
	$(PY) -m neuralchess uci --checkpoint runs/laptop/checkpoints/best.npz

bench:  ## inference and search throughput on this machine
	$(PY) -m neuralchess bench

docker:  ## build the container image
	docker build -t neuralchess .

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf build dist *.egg-info
