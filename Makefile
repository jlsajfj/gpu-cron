# bin/py sets LD_LIBRARY_PATH for the manylinux torch wheel on NixOS.

PY := bin/py
DATA := data/out

.PHONY: setup demo data canonical paraphrase assemble train train-mini train-micro train-nano train-pico train-femto eval eval-all web fixture-web test test-gpu test-demo conformance clean

setup: ## create the venv and install CPU torch + numpy
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/pip install -r requirements.txt
	npm install

canonical: ## cron -> canonical English (cronstrue), validated by cron-parser
	node data/build-canonical.mjs

paraphrase: canonical ## canonical English -> many user phrasings (LLM; resumable, costs money)
	node data/paraphrase.mjs

augment: ## deterministic coverage of number spellings (no API, seeded)
	node data/augment.mjs

assemble: paraphrase augment ## build train/val/test/holdout splits
	$(PY) data/assemble.py

data: assemble

train: ## the default CPU run (configs/default.json)
	$(PY) train/train.py --config configs/default.json --out runs/default

train-mini: ## the browser-sized model
	$(PY) train/train.py --config configs/mini.json --out runs/mini

train-micro: ## 486k params
	$(PY) train/train.py --config configs/micro.json --out runs/micro --bf16
train-nano: ## 226k params
	$(PY) train/train.py --config configs/nano.json --out runs/nano --bf16
train-femto: ## 86k params
	$(PY) train/train.py --config configs/femto.json --out runs/femto --bf16
train-pico: ## 45k params -- the model the browser demo ships
	$(PY) train/train.py --config configs/pico.json --out runs/pico --bf16

CKPT ?= runs/pico/checkpoint.pt

eval: ## exact + semantic match on the held-out splits (CKPT=runs/<run>/checkpoint.pt)
	$(PY) eval/evaluate.py --checkpoint $(CKPT) --out eval/results/$(notdir $(patsubst %/,%,$(dir $(CKPT)))).json

eval-all: ## evaluate every run, then re-render the README's results table
	@for ckpt in runs/*/checkpoint.pt; do \
		name=$$(basename $$(dirname $$ckpt)); \
		echo "== $$name"; \
		$(PY) eval/evaluate.py --checkpoint $$ckpt --out eval/results/$$name.json || exit 1; \
	done
	$(PY) eval/render_table.py eval/results/*.json

demo: ## build and serve the demo, opening a browser (the thing to run after a clone)
	npm run demo

web: ## export the shipped model and bundle dist/ + demo/dist/ (no inference runtime)
	$(PY) export/export_js.py --checkpoint $(CKPT) --out weights/model --dtype int8
	npm run build

fixture-web: ## regenerate the fixture the GPU conformance harness pins against
	$(PY) export/make_test_fixture.py

conformance: ## regenerate the Python/JS agreement fixture
	$(PY) grammar/generate_conformance.py

test: ## python tests + typecheck + unit tests
	$(PY) -m unittest discover -s tests -v
	npm run typecheck
	npm test

test-gpu: ## WGSL forward pass vs the numpy reference, in headless chromium
	npm run test:gpu

test-demo: ## drive real prompts through the built demo, end to end
	npm run test:demo

clean:
	rm -rf $(DATA) runs dist demo/dist test/gpu-bundle.js
