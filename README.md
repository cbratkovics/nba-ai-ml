# NBA Performance Prediction System

An applied data-science portfolio project for exploring NBA player-performance forecasting, feature engineering, evaluation, and application development. The repository contains substantive historical-data pipelines and several experimental model implementations, plus a fixture-driven Next.js demonstration. It does **not** publish a verified forecast-quality, traffic, latency, uptime, or deployment result.

> **Demo status:** the configured frontend URL and Railway templates are retained, but this cleanup did not deploy them. A local build proves only that the checkout compiles. Example UI predictions are deterministic synthetic fixtures and are labeled as such.

## Capability and evidence

| Capability | Implementation | Verification in this checkout | Limitation |
| --- | --- | --- | --- |
| Historical feature engineering | `ml/data/processors/feature_engineer.py`, `api/features/player_features.py` | Source inspected; focused serving-contract tests | Requires historical data/database; prediction-time causality has not been comprehensively evaluated |
| Multi-model ensemble | `ml/models/ensemble.py` | Implementation present | Experimental: its meta-learner uses base-model predictions from their fitting rows, not chronological out-of-fold stacking |
| Saved-artifact inference | `ml/serving/predictor_v2.py` → `api/endpoints/predictions.py` | Missing-artifact, response, and cache boundaries tested | Existing serialized artifact provenance is unknown; no binaries are committed in this checkout |
| Synthetic smoke artifacts | `scripts/create_dummy_model.py` | Deterministic generation tested in a temporary directory | Synthetic data; not evidence of NBA forecast quality and not selected by serving automatically |
| Interactive UI | `frontend/` | lint/build commands documented below | Fixture presentation is not a deployed or live-model verification |
| Optional monitoring/experimentation | `monitoring/`, `api/ml/experiments.py`, `scripts/automated_retraining_pipeline.py` | Implementation paths inspected | Code presence does not establish operated monitoring, A/B testing, or automated promotion |

The detailed route map, evidence definitions, limitations, and external follow-ups are in [`docs/PORTFOLIO_EVIDENCE.md`](docs/PORTFOLIO_EVIDENCE.md).

## Actual data, model, and serving paths

1. **Historical modeling:** NBA client/database records → feature processors → training scripts (`api/ml/train_models.py`, `scripts/train_ensemble_models.py`, or `scripts/train_production_models.py`) → script-specific artifacts. These scripts are experimental and do not together establish one reproduced benchmark.
2. **Serving:** database/NBA client → `api.features.player_features.FeatureEngineer` → trusted `models/*.pkl` selected by `ml.serving.predictor_v2.ModelRegistry` → `/v1/predict` → API client. Missing artifacts now remain errors; no in-memory synthetic model impersonates inference.
3. **Synthetic artifact smoke test:** deterministic generated rows → a preprocessing/model `Pipeline` → `artifacts/synthetic-demo/` plus `provenance.json`. This isolated route is not a default serving route.
4. **Frontend fixture route:** constants in UI components → labeled example cards. It bypasses the prediction API and has no model-quality meaning.

The Random Forest serving convention and the RF/XGBoost/LightGBM Bayesian-ridge ensemble are distinct implementations. A Random Forest is itself a tree ensemble, but that does not mean the multi-library stack is the artifact serving `/v1/predict`.

## Quick start: offline fixture demonstration

Prerequisites: Node.js 18+ and the committed npm lockfile.

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000`. No database, Redis, credentials, or paid service is required for the labeled fixture UI.

## Backend and real-data path

Python 3.10+ is expected.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

`DATABASE_URL` is required for database-backed features. `REDIS_URL` is optional; without it inference runs uncached. `CLERK_SECRET_KEY` is needed only to validate Clerk bearer tokens; anonymous demo-tier requests are permitted by the current auth dependency. The liveness endpoint does not imply model readiness. Prediction calls fail when data or a trusted model is unavailable rather than returning hidden synthetic output.

Historical collection/training scripts may call the NBA API, write data, or overwrite script-specific outputs. Inspect configuration and choose an isolated output before running them. They were not executed during this bounded cleanup.

### Synthetic artifact smoke check

```bash
python scripts/create_dummy_model.py --output-dir /tmp/nba-synthetic-artifacts --samples 40
```

The command refuses a directory named `models` and refuses a non-empty output unless `--overwrite` is explicit. `provenance.json` records source kind, command, seed, timestamp, feature order, preprocessing, source commit/dirty state, and SHA-256 hashes.

## Evaluation semantics

Report MAE and RMSE in the target unit (points, rebounds, or assists); report R² as a dimensionless coefficient, never “percent accuracy.” Undefined R² and non-finite values must remain unavailable. A tolerance hit rate must state its threshold, target, eligible denominator, dataset, and split. Model disagreement or heuristic ranges are not calibrated prediction intervals without held-out coverage evidence.

A publishable evaluation needs the dataset/input hash, provenance and date range, inclusion rules, chronological train/validation/test boundaries, prediction horizon, feature and preprocessing identity, model configuration/artifact hash, seed, dependency versions, per-target denominators, causal baseline, reproduction command, commit and dirty state, and permitted prediction-level output. No qualifying evaluation artifact is committed today, so the dashboard renders an empty state.

## Checks

```bash
pytest -q tests/test_provenance_contract.py
cd frontend && npm run lint
cd frontend && npm run build
cd dashboard && npm run build
python scripts/create_dummy_model.py --output-dir /tmp/nba-synthetic-artifacts --samples 40
python -m compileall api ml scripts/create_dummy_model.py
```

## Repository map and deployment

- `api/`: FastAPI routes, schemas, database-backed features, and optional services.
- `ml/`: feature processing, model experiments, evaluation helpers, and serving.
- `frontend/`: maintained Next.js portfolio UI (`package-lock.json` governs installs).
- `dashboard/`: retained legacy Create React App UI, explicitly labeled in-app.
- `scripts/`: ingestion, training, deployment checks, and synthetic smoke generation.
- `docs/PORTFOLIO_EVIDENCE.md`: evidence ledger, limitations, and external copy.

`Dockerfile` and `railway.json` configure the FastAPI container and `/health` liveness probe. They are deployment templates, not proof that a current public service was deployed or inspected. After merge, an authorized operator must deploy, check readiness and inference separately, and then verify desktop/mobile fixture and failure states at the public URL.
