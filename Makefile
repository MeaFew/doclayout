.PHONY: all setup samples download detect evaluate visualize dashboard \
        verify lint format format-check test typecheck audit clean

PYTHON := python

# ── Pipeline ─────────────────────────────────────────────────────
all: samples detect visualize

samples:
	$(PYTHON) -m doclayout.make_samples

download:
	$(PYTHON) -m doclayout.download_data

detect:
	$(PYTHON) -m doclayout.detect --batch samples

evaluate:
	$(PYTHON) -m doclayout.evaluate

visualize:
	$(PYTHON) -m doclayout.visualize

# ── Dashboard ────────────────────────────────────────────────────
dashboard:
	streamlit run dashboard/app.py

# ── Setup ────────────────────────────────────────────────────────
# Note: requirements.txt contains platform-specific deps (paddlepaddle,
# pycocotools-windows). No cross-platform lock file is generated for this
# reason. Light runtime deps (numpy/Pillow/pycocotools-windows) are declared
# in pyproject [project].dependencies, so `pip install -e ".[dev]"` alone
# is enough to run the tests — CI installs exactly that (no paddle).
setup:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# ── Quality gates ────────────────────────────────────────────────
lint:
	ruff check src/ tests/ dashboard/

format:
	ruff format src/ tests/ dashboard/

format-check:
	ruff format --check src/ tests/ dashboard/

test:
	pytest tests/ -v --tb=short --cov=doclayout --cov-report=term-missing --cov-fail-under=60

typecheck:
	mypy src/doclayout

audit:
	$(PYTHON) -m doclayout.audit_consistency

verify: lint format-check typecheck test audit
	@echo "All quality gates passed"

# ── Utilities ────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
