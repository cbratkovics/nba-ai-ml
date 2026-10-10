.PHONY: help install test lint lint-sql dbt-load dbt-build dbt-full dbt-nightly dbt-docs dbt-export policy drift-reference drift-calibrate check-docs frontend-build

PY ?= .venv/bin/python
DBT ?= .venv/bin/dbt
DBT_FLAGS = --project-dir dbt --profiles-dir dbt
export DBT_TARGET ?= local
export NBA_DUCKDB_PATH ?= .duckdb/nba.duckdb

help:
	@echo "install      - pip install -c constraints.txt -e .[dev]"
	@echo "test         - ruff + sqlfluff + pytest (skips the two LightGBM test files where libomp is missing)"
	@echo "lint-sql     - sqlfluff over the dbt models, tests and snapshots (.sqlfluff)"
	@echo "dbt-load     - pull the warehouse sources from Hugging Face into data/warehouse"
	@echo "dbt-build    - dbt build on the local DuckDB file (.duckdb/nba.duckdb)"
	@echo "dbt-full     - dbt build --full-refresh (same target)"
	@echo "dbt-nightly  - the nightly selection (dbt/selectors.yml): every model, seed and the snapshot, incremental"
	@echo "dbt-docs     - dbt docs generate --static, then the description check"
	@echo "dbt-export   - export gold marts to data/warehouse/export"
	@echo "policy       - write reports/policy_<season>.json from the built gold marts, then rebuild (dbt-full) so the policy marts reconcile"
	@echo "drift-reference  - write reports/drift_reference_<feature>_<model>.json from gold.fct_player_game"
	@echo "drift-calibrate  - write reports/drift_calibration_<season>.json over the replay season's dates"
	@echo "Set DBT_TARGET=motherduck and MOTHERDUCK_TOKEN to build on MotherDuck (database nba)."

install:
	uv pip install --python $(PY) -c constraints.txt -e ".[dev]"

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

lint-sql:
	mkdir -p .duckdb
	.venv/bin/sqlfluff lint dbt/models dbt/tests dbt/snapshots

test: lint lint-sql
	$(PY) -m pytest -q

dbt-load:
	$(PY) -m nba.warehouse.load --root data/warehouse

dbt-build:
	mkdir -p .duckdb
	$(DBT) build $(DBT_FLAGS)

dbt-full:
	mkdir -p .duckdb
	$(DBT) build $(DBT_FLAGS) --full-refresh

dbt-nightly:
	$(DBT) build $(DBT_FLAGS) --selector nightly

policy:
	$(PY) -m nba.decisions.evaluate --duckdb $(NBA_DUCKDB_PATH)

drift-reference:
	$(PY) -m nba.drift.reference --duckdb $(NBA_DUCKDB_PATH)

drift-calibrate:
	$(PY) -m nba.drift.calibrate --duckdb $(NBA_DUCKDB_PATH)

dbt-docs:
	$(DBT) docs generate $(DBT_FLAGS) --static
	$(PY) scripts/check_dbt_descriptions.py

dbt-export:
	mkdir -p data/warehouse/export
	$(DBT) run-operation export_gold $(DBT_FLAGS)

frontend-build:
	cd frontend && npm run build
