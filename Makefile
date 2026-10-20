PYTHON ?= python
OUT ?= runs/bench

.PHONY: install rust test lint bench smoke train demo reproduce archive
install:
	$(PYTHON) -m pip install -e '.[train,bench,export,demo,dev]'
rust:
	$(PYTHON) -m pip install ./rust
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check src tests scripts examples
	$(PYTHON) -m ruff format --check src tests scripts examples
bench:
	$(PYTHON) -m noiseroute bench --config configs/bench-small20.yaml --out $(OUT)
smoke:
	$(PYTHON) scripts/reproduce.py --out $(OUT) --smoke
reproduce:
	$(PYTHON) scripts/reproduce.py --out $(OUT)
train:
	$(PYTHON) -m noiseroute imitate --config configs/train-laptop.yaml --out runs/laptop-bc
	$(PYTHON) -m noiseroute train --config configs/train-laptop.yaml --checkpoint runs/laptop-bc/imitation.pt --out runs/laptop-ppo
demo:
	$(PYTHON) -m streamlit run examples/demo.py
archive:
	$(PYTHON) scripts/archive.py
