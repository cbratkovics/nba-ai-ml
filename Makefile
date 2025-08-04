.PHONY: help install setup train serve test deploy clean lint format type-check migrate monitor

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make setup      - Initialize database and collect data"
	@echo "  make train      - Train ML models"
	@echo "  make serve      - Run API server"
	@echo "  make test       - Run all tests"
	@echo "  make deploy     - Deploy to production"
	@echo "  make clean      - Clean cache and temp files"
	@echo "  make lint       - Run code linters"
	@echo "  make format     - Format code"
	@echo "  make monitor    - Start monitoring stack"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pre-commit install

setup:
	python scripts/setup.py
	python scripts/collect_historical_data.py --seasons 2021-2025

train:
	python -m ml.training.pipeline --config config/training.yaml
	@echo "Training complete. Check MLflow UI at http://localhost:5000"

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

test:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report available at htmlcov/index.html"

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-load:
	locust -f tests/load/test_api_load.py --host http://localhost:8000

lint:
	ruff . --fix
	black . --check
	mypy .

format:
	black .
	ruff . --fix

type-check:
	mypy . --strict

migrate:
	alembic upgrade head

rollback:
	alembic downgrade -1

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

deploy-staging:
	./scripts/deploy.py staging

deploy-production:
	./scripts/deploy.py production

monitor:
	docker-compose -f infrastructure/docker/docker-compose.monitoring.yml up -d
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist htmlcov .coverage

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

retrain:
	python scripts/retrain_models.py --auto-deploy

backup-models:
	python scripts/backup_models.py --destination s3://nba-ml-models/backup/

validate-data:
	python -m ml.data.processors.data_validator --latest

benchmark:
	python scripts/benchmark_models.py --output reports/benchmark.html