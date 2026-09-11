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

This is a batch NBA player-stat prediction pipeline (points, rebounds, assists) that is
currently being rebuilt. The current tree contains earlier prototype code: a FastAPI service,
a Next.js dashboard, data-collection scripts, and several training experiments. That code is
not wired end to end, and the dashboard pages render sample data (see the demo banner on each
page). Treat the repository as a work in progress rather than a deployed system.

## Data source

Game logs come from `nba_api`, an unofficial client for `stats.nba.com`. This is a
non-commercial personal portfolio project. Requests are rate-limited politely (a fixed delay
between calls, no parallel scraping), and no data is redistributed from this repository.

## Model and System Notes

### Trained Artifact

Model artifacts are not committed to this repository; they live on Hugging Face. The local
metadata file `models/features.json` (also not committed) describes the artifact layout:

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

## Docker Support

```bash
# Build image
docker build -t nba-ml:latest .

# Run container (start.py listens on $PORT, default 8080)
docker run -p 8080:8080 nba-ml:latest
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
