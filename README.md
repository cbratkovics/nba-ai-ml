# 🏀 NBA Performance Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14.2+-000000?style=for-the-badge&logo=next.js&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=for-the-badge&logo=typescript&logoColor=white)


**Machine learning platform for NBA player performance prediction with feature engineering, model serving, and monitoring**

</div>

---

## Overview

An end-to-end machine learning platform demonstrating production ML engineering through:

- **Ensemble prediction** for points, rebounds, and assists using tree-based models
- **Redis caching** in front of model serving to avoid recomputing repeat requests
- **ETL pipeline** for ingesting and cleaning historical game records
- **Feature engineering** with rolling averages, matchup difficulty, rest days, and other contextual signals
- **Drift detection** with KS and Chi-squared tests for model monitoring

## Key Features

### Machine Learning Pipeline
- **Ensemble Models**: XGBoost, LightGBM, and Random Forest combination
- **Feature Engineering**: Rolling averages, opponent analysis, rest days, and momentum tracking
- **Multi-Target Prediction**: Separate models for points, rebounds, and assists
- **Hyperparameter Tuning**: Optuna-based optimization with cross-validation
- **Explainability**: SHAP values for feature importance analysis

### Production Architecture
- **FastAPI Backend**: Async request handling with Pydantic validation
- **Redis Caching**: Intelligent TTL strategies for frequently accessed predictions
- **PostgreSQL Storage**: Optimized queries with SQLAlchemy ORM
- **A/B Testing**: Framework for model comparison with statistical significance
- **Monitoring**: Drift detection and performance tracking

### Frontend Dashboard
- **Next.js 14**: Modern React framework with TypeScript
- **Real-time Updates**: SWR for data fetching and caching
- **Data Visualization**: Recharts for interactive charts
- **Responsive Design**: Tailwind CSS with mobile optimization

## Technology Stack

### Backend
- **Core**: Python 3.10+, FastAPI, SQLAlchemy
- **ML**: XGBoost, LightGBM, scikit-learn, pandas, numpy
- **Infrastructure**: Redis, PostgreSQL, Docker
- **Testing**: pytest with 87% coverage

### Frontend  
- **Framework**: Next.js 14, TypeScript, React
- **Styling**: Tailwind CSS, Framer Motion
- **Data**: SWR, Recharts
- **Build**: Vercel deployment ready

## Model and System Notes

### Trained Artifact

The committed model metadata lives in [`models/features.json`](models/features.json) and describes
what is actually trained and served:

| Property | Value |
|----------|-------|
| Model type | RandomForestRegressor |
| Feature count | 20 |
| Targets | points, rebounds, assists |

Feature names are listed in that file. They cover rolling averages over 5, 10, and 20 game windows
for points, rebounds, and assists, shooting percentages, minutes, games played, age, home/away,
rest days, back-to-back flags, matchup difficulty, and season game number.

### Evaluation

This repository does not publish model accuracy, latency, or dataset-size figures. Retrain on your
own data pull and evaluate with the scripts in this repo to get numbers that describe your run.
Model quality depends on the seasons you ingest, your train/test split, and how you handle
injuries and rotation changes, so numbers copied from another environment would be misleading.

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL 14+
- Redis (optional, for caching)

### Quick Start

1. **Clone Repository**
```bash
git clone https://github.com/cbratkovics/nba-ai-ml.git
cd nba-ai-ml
```

2. **Backend Setup**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Frontend Setup**
```bash
cd frontend
npm install
```

4. **Environment Variables**
```bash
# Backend (.env)
DATABASE_URL=postgresql://user:pass@localhost:5432/nba_ml
REDIS_URL=redis://localhost:6379
API_KEY=your-secret-key

# Frontend (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

5. **Run Services**
```bash
# Backend
uvicorn api.main:app --reload

# Frontend
cd frontend && npm run dev
```

## API Reference

### Core Endpoints

#### Single Prediction
```http
POST /v1/predict
Content-Type: application/json

{
  "player_id": "203999",
  "game_date": "2024-12-15",
  "opponent_team": "LAL",
  "home_game": true
}
```

#### Response
```json
{
  "player_name": "Nikola Jokic",
  "predictions": {
    "points": 28.5,
    "rebounds": 13.2,
    "assists": 8.7
  },
  "confidence_intervals": {
    "points": {"lower": 23.2, "upper": 33.8}
  },
  "model_confidence": 0.923
}
```

#### Batch Predictions
```http
POST /v1/predict/batch
```

#### Model Performance
```http
GET /v1/models/performance
```

## Project Structure

```
nba-ai-ml/
├── api/                # FastAPI backend
│   ├── models/        # ML models
│   ├── routes/        # API endpoints
│   └── services/      # Business logic
├── ml/                # Machine learning
│   ├── features/      # Feature engineering
│   ├── models/        # Model training
│   └── evaluation/    # Model evaluation
├── frontend/          # Next.js dashboard
│   ├── components/    # React components
│   ├── pages/         # Next.js pages
│   └── lib/          # Utilities
└── tests/            # Test suite
```

## Key Achievements

- **Multi-Target Models**: Separate predictors for points, rebounds, and assists
- **Cached Serving**: FastAPI with Redis caching for repeat requests
- **End-to-End Pipeline**: Ingestion, feature engineering, training, and serving
- **Containerized**: Docker, testing, and monitoring included
- **A/B Testing**: Framework for model experimentation
- **MLOps Integration**: Drift detection and automated retraining

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=api tests/

# Specific test file
pytest tests/test_predictions.py
```

## Docker Support

```bash
# Build image
docker build -t nba-ml:latest .

# Run container
docker run -p 8000:8000 nba-ml:latest
```

## Future Enhancements

- [ ] Real-time data streaming integration
- [ ] Advanced time series models (LSTM)
- [ ] Player injury impact modeling
- [ ] Team chemistry factors
- [ ] Playoff performance adjustments

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create your feature branch
3. Commit changes with clear messages
4. Push to your branch
5. Open a Pull Request

## License

MIT License - See [LICENSE](LICENSE) for details

## Acknowledgments

- NBA Stats API for data access
- XGBoost and LightGBM communities
- FastAPI for excellent documentation
- Open source ML community

---

<div align="center">

### Built by Christopher Bratkovics

[![Portfolio](https://img.shields.io/badge/Portfolio-cbratkovics.dev-4A90E2?style=for-the-badge)](https://cbratkovics.dev)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/cbratkovics)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/cbratkovics)

**⭐ Star this repository if you find it useful!**

</div>