PYTHON ?= python
AUDIT_DIR ?= build/audit

.PHONY: help install test verify-freeze oria-preflight reproduce research-status control-structural-recovery release-check diff-check preflight clean-audit

help:
	@printf '%s\n' \
	  'PCC Poker v0.8.0 safe developer commands' \
	  '' \
	  '  make install          Install the package in editable mode' \
	  '  make test             Run the full pytest suite' \
	  '  make verify-freeze    Verify immutable v0.8.0 scientific artifacts' \
	  '  make oria-preflight    Run synthetic-only HandHQ ingestion safety checks' \
	  '  make reproduce        Write a NON-FROZEN audit manifest under build/audit/' \
	  '  make research-status  Regenerate status reports under build/audit/' \
	  '  make control-structural-recovery  Run post-freeze synthetic Control structure test' \
	  '  make release-check    Run read-only release metadata/hygiene checks' \
	  '  make diff-check       Run git diff --check' \
	  '  make preflight        Run all safe pre-release checks' \
	  '  make clean-audit      Remove generated build/audit files'

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

verify-freeze:
	$(PYTHON) -m pcc_poker verify-freeze

oria-preflight:
	$(PYTHON) -m pcc_poker oria-ingestion-preflight

reproduce:
	@mkdir -p $(AUDIT_DIR)
	$(PYTHON) -m pcc_poker reproduce --output $(AUDIT_DIR)/reproducibility-manifest.json

research-status:
	@mkdir -p $(AUDIT_DIR)
	$(PYTHON) -m pcc_poker research-status \
		--json-output $(AUDIT_DIR)/research-status.json \
		--csv-output $(AUDIT_DIR)/research-status.csv \
		--markdown-output $(AUDIT_DIR)/RESEARCH_STATUS.md

control-structural-recovery:
	$(PYTHON) -m pcc_poker control-structural-recovery --output validation/control-structural-recovery.json

release-check:
	$(PYTHON) -m pcc_poker release-check

diff-check:
	@if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
		git diff --check; \
	else \
		printf '%s\n' 'git diff --check skipped (not a git work tree)'; \
	fi

preflight: test verify-freeze oria-preflight reproduce research-status release-check diff-check
	@printf '%s\n' 'Preflight passed: tests, frozen hashes, ORIA-safe ingestion checks, audit outputs, release metadata, and diff checks are clean.'

clean-audit:
	rm -rf $(AUDIT_DIR)
