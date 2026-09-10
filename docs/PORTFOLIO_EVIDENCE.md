# Portfolio evidence and model-path map

## Evidence rules

Implementation, execution, evaluation, artifact selection, and deployment verification are recorded separately. “Present” below means source exists, not that it ran successfully in a hosted environment.

## Route map

| Route | Source → features → training/artifact → loader/endpoint → UI | Status |
| --- | --- | --- |
| Historical experiment | NBA API/database → `ml/data/processors/feature_engineer.py` or training-script transforms → `api/ml/train_models.py`, `scripts/train_ensemble_models.py`, `scripts/train_production_models.py` → script-specific outputs | Implemented in multiple experimental paths; no qualifying evaluation record inspected |
| Serving | database/NBA client → `api/features/player_features.py` → trusted `models/rf_{target}_model.pkl`, `ensemble_{target}_{version}.pkl`, or `rf_model.pkl` → `ModelRegistry` → `/v1/predict` → `frontend/lib/api-client.ts` | Route traced and error/cache contracts tested; current artifact provenance unknown/absent |
| Synthetic smoke | seeded NumPy rows → `Pipeline(StandardScaler, RandomForestRegressor)` → isolated `artifacts/synthetic-demo` + metadata | Executed only as an offline smoke path; never NBA evidence or an automatic fallback |
| UI fixtures | static component constants → example cards | Deterministic, labeled synthetic fixture; no backend inference |

The substantive `NBAEnsemble` in `ml/models/ensemble.py` combines Random Forest, XGBoost, and LightGBM base estimators with a Bayesian ridge meta-learner and contains optional neural-network/MLflow paths. It is preserved as experimental modeling work. Its current fit path trains the meta-learner on predictions from rows used to fit base models, so it must not be called out-of-fold stacking or used as validated forecast evidence without redesign and a chronological untouched test set.

The serving registry instead searches conventional Random Forest and versioned ensemble filenames. No inspectable metadata binds the absent/unknown serialized serving artifacts to the multi-model implementation. Serialized joblib/pickle files must be trusted before loading because deserialization can execute code.

## Data and methodology limitations

- Multiple feature/training implementations exist and do not demonstrate one shared end-to-end preprocessing contract.
- Rolling features must be ordered per player and shifted before the target game. This has not been comprehensively proven for every training script.
- Random cross-validation and tuning on a final test set would not support pregame forecasting claims; use date-group chronological boundaries and a separate validation period.
- Compare against a prior-games mean using only eligible earlier games, with window and cold-start policy stated.
- Heuristic “confidence” and interval widths in legacy code are not calibrated held-out coverage. API evaluation metrics remain empty until backed by a record.
- Redis is optional. Cache keys are partitioned by source kind and model version; cached results carry provenance and observation time.

## Deployment and external surfaces

The repository contains a Railway backend template and refers to a Vercel URL in a deployment-check script. This cleanup did not push, deploy, inspect GitHub settings, or verify that either public surface reflects this commit.

After merge, an authorized maintainer should:

1. Deploy the backend/frontend and verify API liveness, model readiness, missing-model errors, and fixture labels separately.
2. Inspect desktop and mobile loading, fixture, and failure states in a browser.
3. Update GitHub About/topics if desired; do not imply deployed ensemble serving or measured performance.
4. Update résumé/portfolio copy independently and link only to a deployment verified at the merged commit.

**Proposed GitHub About (under 350 characters):** NBA player-performance forecasting portfolio featuring historical-data feature engineering, experimental tree-model ensembles, reproducible artifact provenance, a FastAPI serving path, and a labeled Next.js fixture demonstration.

Suggested topics: `data-science`, `sports-analytics`, `machine-learning`, `feature-engineering`, `fastapi`, `nextjs`, `xgboost`, `lightgbm`. These describe code present, not operated services.

**Defensible résumé/site bullets:**

- Developed NBA player-performance feature pipelines and experimental Random Forest/XGBoost/LightGBM ensemble code, with chronological evaluation requirements documented to prevent unsupported quality claims.
- Built a FastAPI/Next.js portfolio demonstration with explicit fixture and inference provenance, model-version-aware caching, and honest unavailable states when data, artifacts, or evaluations are missing.
