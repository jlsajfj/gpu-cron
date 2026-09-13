# bin/py sets LD_LIBRARY_PATH for the manylinux torch wheel on NixOS.

PY := bin/py
DATA := data/out

.PHONY: setup data canonical paraphrase assemble train train-mini train-nano eval web fixture-web test conformance clean

setup: ## create the venv and install CPU torch + numpy
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/pip install -r requirements.txt
	cd web && npm install

canonical: ## cron -> canonical English (cronstrue), validated by cron-parser
	node data/build-canonical.mjs

paraphrase: canonical ## canonical English -> many user phrasings (LLM; resumable, costs money)
	node data/paraphrase.mjs

assemble: paraphrase ## build train/val/test/holdout splits
	$(PY) data/assemble.py

data: assemble

train: ## the default CPU run (configs/default.json)
	$(PY) train/train.py --config configs/default.json --out runs/default

train-mini: ## the browser-sized model
	$(PY) train/train.py --config configs/mini.json --out runs/mini

train-nano: ## the model the browser demo ships (a few hundred thousand params)
	$(PY) train/train.py --config configs/nano.json --out runs/nano --bf16

eval: ## exact + semantic match on the held-out splits
	$(PY) eval/evaluate.py --checkpoint runs/default/checkpoint.pt --out eval/results/default.json

web: ## export the browser weights and bundle the demo (no inference runtime)
	$(PY) export/export_js.py --checkpoint runs/nano/checkpoint.pt --out web/weights/model
	cd web && npm run build

fixture-web: ## regenerate the fixture that pins web/src/forward.ts to the reference
	$(PY) export/make_test_fixture.py

conformance: ## regenerate the Python/JS agreement fixture
	$(PY) grammar/generate_conformance.py

test: ## python property tests + browser unit tests
	$(PY) -m unittest discover -s tests -v
	cd web && npm run typecheck && npm test

clean:
	rm -rf $(DATA) runs web/dist
