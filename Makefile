# bin/py sets LD_LIBRARY_PATH for the manylinux torch wheel on NixOS.

PY := bin/py
DATA := data/out

.PHONY: setup data canonical paraphrase assemble train train-mini eval web test conformance clean

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

eval: ## exact + semantic match on the held-out splits
	$(PY) eval/evaluate.py --checkpoint runs/default/checkpoint.pt --out eval/results/default.json

web: ## export int8 ONNX and bundle the browser demo
	$(PY) export/export_onnx.py --checkpoint runs/mini/checkpoint.pt --int8 \
		--out web/weights/model.int8.onnx
	cd web && npm run build

conformance: ## regenerate the Python/JS agreement fixture
	$(PY) grammar/generate_conformance.py

test: ## python property tests + browser unit tests
	$(PY) -m unittest discover -s tests -v
	cd web && npm run typecheck && npm test

clean:
	rm -rf $(DATA) runs web/dist
