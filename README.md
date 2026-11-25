# ✈️ Airline No-Show Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)
![React](https://img.shields.io/badge/React-18-blue?style=for-the-badge&logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5-blue?style=for-the-badge&logo=typescript)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)

**An ML-powered airline passenger no-show prediction prototype with analytics, explainability, and decision-support tools.**

</div>

---

## ✨ Features

- 🔮 **ML Predictions** — Random Forest, XGBoost, LightGBM, Gradient Boosting, Logistic Regression
- 📊 **Analytics Dashboard** — Executive KPIs, route intelligence, channel performance, time trends
- 💰 **Overbooking Simulator** — Monte Carlo scenario risk & cost analysis
- 🔍 **Drift Monitoring** — Feature-distribution drift detection (PSI & KS tests)
- 🧠 **Explainable AI** — SHAP-based local and global explanations
- 🔒 **Secure by Design** — JWT auth with HttpOnly cookies, CSRF protection, RBAC

## 🏗️ Architecture

```
airline-no-show-prediction/
├── backend/      # Flask REST API
├── frontend/     # React + TypeScript
├── ml/           # ML Pipeline
├── sql/          # Database schema & views
├── docker/       # Container configs
├── models/       # Trained artifacts
└── tests/        # Unit & integration tests
```

## 🚀 Quick Start

### Docker
```bash
cp .env.example .env
docker compose up --build -d
```

### Manual
```bash
# Backend
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python backend\run.py

# Frontend
npm --prefix frontend ci
npm --prefix frontend run dev
```

## 📖 API Highlights

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | Authenticate user |
| POST | `/api/predict` | Single prediction |
| POST | `/api/bulk-predict` | Batch prediction (max 1000) |
| GET  | `/api/analytics/overview` | Executive dashboard |
| GET  | `/api/analytics/routes` | Route analysis |
| POST | `/api/overbooking/simulate` | Overbooking simulation |

## 🧪 Testing

```bash
python -m pytest -q
npm --prefix frontend run build
```

## 📄 License

MIT License — see LICENSE file.

---

<div align="center">
Built with ❤️ for the aviation industry
</div>
