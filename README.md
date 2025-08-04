# NBA AI/ML Prediction Platform

A production-ready NBA player performance prediction system deployed on Railway with real-time API access.

## Current Status

**Live Production URL**: https://nba-ai-ml-production.up.railway.app

### Working Features
- ✅ FastAPI backend with prediction endpoints
- ✅ React frontend with interactive UI
- ✅ Redis caching for performance
- ✅ Multiple ML models (Random Forest, XGBoost, LightGBM)
- ✅ Player prediction interface
- ✅ Model A/B testing framework
- ✅ Health monitoring endpoints
- ✅ Deployment health checks
- ✅ Graceful error handling

### Current Limitations
- 🔄 Using dummy/mock data (not connected to real NBA API yet)
- 🔄 Models trained on synthetic data
- 🔄 No real-time data updates
- 🔄 Limited to 4 demo players

## Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.10+)
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, PyTorch
- **Data Processing**: pandas, numpy, shap
- **Caching**: Redis
- **Database**: SQLAlchemy with PostgreSQL support
- **Deployment**: Railway with automatic restarts

### Frontend
- **Framework**: React with TypeScript
- **UI Library**: Tailwind CSS
- **State Management**: React Hooks
- **API Client**: Axios
- **Charts**: Chart.js

### Infrastructure
- **Hosting**: Railway (PaaS)
- **CI/CD**: GitHub Actions ready
- **Monitoring**: Prometheus metrics exposed
- **Logging**: Structured logging with Python logging

## Project Structure

```
nba-ai-ml/
├── api/                    # FastAPI backend
│   ├── main.py            # Application entry point
│   ├── endpoints/         # API endpoints
│   │   ├── predictions.py # Prediction endpoints
│   │   ├── experiments.py # A/B testing endpoints
│   │   └── health.py      # Health check endpoints
│   └── enterprise/        # Enterprise features
├── ml/                    # Machine learning modules
│   ├── serving/          # Model serving
│   │   └── predictor.py  # Prediction service (uses dummy data)
│   ├── training/         # Training pipelines
│   ├── models/           # Model implementations
│   └── data/             # Data processing
├── frontend/             # React frontend
│   ├── src/
│   │   ├── App.tsx       # Main application
│   │   ├── api.ts        # API client
│   │   └── components/   # React components
│   └── package.json
├── scripts/              # Utility scripts
│   ├── deployment_health_check.py
│   └── setup_project.py
├── requirements.txt      # Python dependencies
├── railway.json         # Railway deployment config
└── .env.example         # Environment variables template
```

## API Endpoints

### Base URL
- Production: `https://nba-ai-ml-production.up.railway.app`
- Local: `http://localhost:8000`

### Available Endpoints

#### Health Check
```bash
GET /health
```
Returns system health status.

#### Predict Player Performance
```bash
POST /v1/predict
Content-Type: application/json

{
    "player_id": "203999",
    "game_date": "2025-01-15",
    "opponent_team": "LAL",
    "include_explanation": true
}
```

**Available Demo Players**:
- `203999` - Nikola Jokic
- `2544` - LeBron James
- `201939` - Stephen Curry
- `1628369` - Jayson Tatum

#### Get Next Game Prediction
```bash
GET /v1/predict/next-game/{player_id}
```

#### A/B Testing
```bash
POST /v1/experiments/predict
Content-Type: application/json

{
    "experiment_id": "model_v2_test",
    "player_id": "203999",
    "game_date": "2025-01-15",
    "opponent_team": "LAL"
}
```

## Local Development

### Prerequisites
- Python 3.10+
- Node.js 16+
- Redis (optional, for caching)

### Backend Setup
```bash
# Clone repository
git clone https://github.com/cbratkovics/nba-ai-ml.git
cd nba-ai-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn api.main:app --reload --port 8000
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Running Health Checks
```bash
# Check all critical imports
python scripts/deployment_health_check.py

# Test API imports
python -c "import api.main; print('✓ API imports successful')"
```

## Environment Variables

Create a `.env` file based on `.env.example`:

```env
# API Configuration
API_KEY=your-secret-api-key
ENVIRONMENT=development

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379

# Model Configuration
MODEL_PATH=./models
MODEL_VERSION=latest

# Monitoring
ENABLE_METRICS=true
```

## Current Model Performance (Mock Data)

Since the system uses dummy data, these are simulated metrics:

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Points | 0.942 | 3.1 | 4.2 |
| Rebounds | 0.728 | 2.4 | 3.1 |
| Assists | 0.715 | 1.8 | 2.3 |

## Dependencies

### Core ML Dependencies
- `scikit-learn==1.3.2` - Base ML algorithms
- `xgboost==2.0.2` - Gradient boosting
- `lightgbm==4.1.0` - Fast gradient boosting
- `torch==2.2.0` - Neural networks
- `shap==0.44.0` - Model explainability (optional)

### API Dependencies
- `fastapi==0.104.1` - Web framework
- `uvicorn==0.24.0` - ASGI server
- `redis==5.0.1` - Caching
- `pydantic==2.5.0` - Data validation

### Additional Tools
- `pandas==2.1.3` - Data manipulation
- `numpy==1.26.0` - Numerical computing
- `mlflow==2.8.1` - Model versioning
- `prometheus-fastapi-instrumentator==6.1.0` - Metrics

## Deployment

### Railway Deployment

The project is configured for Railway deployment:

1. **Automatic Restarts**: Configured in `railway.json` with up to 10 retries
2. **Health Checks**: Railway monitors the `/health` endpoint
3. **Environment Variables**: Set in Railway dashboard
4. **Build Command**: Automatically installs from `requirements.txt`
5. **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### Manual Deployment Check
```bash
# Verify all dependencies
pip install -r requirements.txt

# Run health check
python scripts/deployment_health_check.py

# Start server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Next Steps

### Phase 1: Real Data Integration
- [ ] Connect to NBA API for live data
- [ ] Implement data collection pipeline
- [ ] Build historical data database
- [ ] Create feature engineering from real stats

### Phase 2: Model Training
- [ ] Train models on real NBA data
- [ ] Implement backtesting framework
- [ ] Add model versioning
- [ ] Create automated retraining pipeline

### Phase 3: Production Features
- [ ] Add authentication system
- [ ] Implement rate limiting
- [ ] Add comprehensive logging
- [ ] Create admin dashboard
- [ ] Set up monitoring alerts

### Phase 4: Advanced Features
- [ ] Real-time game predictions
- [ ] Player injury risk assessment
- [ ] Team chemistry analysis
- [ ] Fantasy sports integration

## Troubleshooting

### Common Issues

1. **Module Import Errors**
   ```bash
   # Install all dependencies
   pip install -r requirements.txt
   
   # Run health check
   python scripts/deployment_health_check.py
   ```

2. **Redis Connection Failed**
   - The app works without Redis, just with warnings
   - To use Redis: `export REDIS_URL=redis://localhost:6379`

3. **Port Already in Use**
   ```bash
   # Use a different port
   uvicorn api.main:app --port 8001
   ```

## Contributing

This project is under active development. Current priorities:
1. Connecting to real NBA data sources
2. Training models on actual game statistics
3. Improving prediction accuracy
4. Adding more features to the frontend

## License

MIT License - see [LICENSE](LICENSE) file for details.