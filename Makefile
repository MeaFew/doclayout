.PHONY: all setup samples download detect evaluate visualize dashboard \
        verify lint format format-check test audit clean

PYTHON := python

# ── Pipeline ─────────────────────────────────────────────────────
# evaluate is excluded from `all` because it requires PubLayNet val data
# + pycocotools and currently raises NotImplementedError.
all: samples detect visualize

samples:
	$(PYTHON) scripts/make_samples.py

download:
	$(PYTHON) scripts/download_data.py

detect:
	$(PYTHON) scripts/detect.py --batch samples

evaluate:
	$(PYTHON) scripts/evaluate.py

visualize:
	$(PYTHON) scripts/visualize.py

# ── Dashboard ────────────────────────────────────────────────────
dashboard:
	streamlit run dashboard/app.py

# ── Setup ────────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt
	pre-commit install

# ── Quality gates ────────────────────────────────────────────────
lint:
	ruff check scripts/ tests/ dashboard/

format:
	ruff format scripts/ tests/ dashboard/

format-check:
	ruff format --check scripts/ tests/ dashboard/

test:
	pytest tests/ -v --tb=short

audit:
	$(PYTHON) scripts/audit_consistency.py

verify: lint format-check test audit
	@echo "All quality gates passed"

# ── Utilities ────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
