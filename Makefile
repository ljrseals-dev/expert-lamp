.PHONY: install smoke test dry clean

install:
	python -m pip install -e ".[dev]"

dry:
	python -m training.train --dry-run

smoke:
	bash scripts/smoke.sh

test:
	python -m pytest tests -q

clean:
	rm -rf checkpoints runs .pytest_cache **/__pycache__
