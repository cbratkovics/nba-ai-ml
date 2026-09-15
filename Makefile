.PHONY: help install test lint dbt-load dbt-build dbt-full dbt-nightly dbt-docs dbt-export check-docs frontend-build

PY ?= .venv/bin/python
DBT ?= .venv/bin/dbt
DBT_FLAGS = --project-dir dbt --profiles-dir dbt
export DBT_TARGET ?= local
export NBA_DUCKDB_PATH ?= .duckdb/nba.duckdb

help:
	@echo "install      - pip install -c constraints.txt -e .[dev]"
	@echo "test         - ruff + pytest (skips the two LightGBM test files where libomp is missing)"
	@echo "dbt-load     - pull the warehouse sources from Hugging Face into data/warehouse"
	@echo "dbt-build    - dbt build on the local DuckDB file (.duckdb/nba.duckdb)"
	@echo "dbt-full     - dbt build --full-refresh (same target)"
	@echo "dbt-nightly  - the nightly selection: incremental silver with parents and children"
	@echo "dbt-docs     - dbt docs generate --static, then the description check"
	@echo "dbt-export   - export gold marts to data/warehouse/export"
	@echo "Set DBT_TARGET=motherduck and MOTHERDUCK_TOKEN to build on MotherDuck (database nba)."

install:
	uv pip install --python $(PY) -c constraints.txt -e ".[dev]"

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

test: lint
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
	$(DBT) build $(DBT_FLAGS) --select "+slv_game_logs+" "+slv_predictions+" "+slv_residuals+" "+slv_daily_reports+" "brz_metrics+" "brz_replay_report+" "brz_replay_all_rows+"

dbt-docs:
	$(DBT) docs generate $(DBT_FLAGS) --static
	$(PY) scripts/check_dbt_descriptions.py

dbt-export:
	mkdir -p data/warehouse/export
	$(DBT) run-operation export_gold $(DBT_FLAGS)

frontend-build:
	cd frontend && npm run build
