# Airline No-Show Prediction System

An ML-powered aviation decision-support prototype for passenger no-show prediction, analytics, explainability, and overbooking simulation.

---

## Problem Statement

Airline no-shows leave seats empty that could have been sold and create operational inefficiency. This system provides prototype risk scoring and decision-support analysis over a 3,000-row dataset with 36.2% no-show prevalence.

**⚠️ Status: Research Prototype** - No model currently passes the production approval gate, so `/api/predict` and `/api/bulk-predict` return HTTP 503 by design. This is **not** an operational recommendation engine.

## Objectives

- Build leakage-aware no-show prediction pipeline with passenger-grouped validation
- Provide analytics dashboard for route, channel, and temporal performance
- Offer SHAP-based local and global explainability for approved model
- Simulate overbooking scenarios with configurable cost parameters
- Monitor feature drift against training reference
- Enforce fail-closed production approval gate

## Key Features

- 🔐 **JWT Authentication** via HttpOnly cookies with CSRF protection and RBAC
- 🎯 **Single & Batch Prediction** (max 1000) with persistence - returns 503 if no approved model
- 📊 **Analytics**: Overview, routes, channels, trends, high-risk bookings (admin)
- 💰 **Monte Carlo Overbooking Simulator** with configurable cost parameters
- 🔍 **Feature-Drift Monitoring** (PSI + KS) with severity bands
- 🧠 **SHAP Explainability** - Local explanations per prediction + global importance
- 🛡️ **Production Config Validation** - Secrets, Postgres, rate-limit storage, proxy hops
- 🐳 **Docker Compose + AWS ECS Fargate** deployment templates

## System Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   React     │────▶│    Flask    │────▶│ PostgreSQL  │
│  TypeScript │     │    API      │     │  (Primary)  │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                    ┌──────▼──────┐
                    │    Redis    │
                    │ (Rate Limit)│
                    └─────────────┘
                           │
                    ┌──────▼──────┐
                    │  ML Models  │
                    │ (Artifacts) │
                    └─────────────┘
```

## ML Pipeline

```
Data CSV → Preprocessing → Feature Engineering → Passenger-Grouped Split (80/20, seed 42)
                                                    ↓
                    ┌─────────────────────────────────────────┐
                    │  Training: 5 Models                     │
                    │  - Logistic Regression                  │
                    │  - Random Forest                        │
                    │  - XGBoost                              │
                    │  - LightGBM                             │
                    │  - Gradient Boosting                    │
                    │  5-fold StratifiedGroupKFold            │
                    └─────────────────────────────────────────┘
                                                    ↓
                              Isotonic Calibration (on OOF predictions)
                                                    ↓
                              Threshold Selection (F1 on calibrated OOF)
                                                    ↓
                              ┌───────────────────────┐
                              │  PRODUCTION GATE      │
                              │  ROC-AUC > 0.70       │
                              │  F1 > 0.50            │
                              │  Precision > 0.50     │
                              │  Recall > 0.50        │
                              │  PR-AUC > Prevalence  │
                              └───────────┬───────────┘
                                          │
                              ┌───────────▼───────────┐
                              │ PASS → Approved Model │
                              │ FAIL → Serving 503    │
                              └───────────────────────┘
```

## Model Performance (Verified from Committed Artifacts)

| Algorithm | ROC-AUC | F1 | Precision | Recall | PR-AUC | Status |
|-----------|---------|-----|-----------|--------|--------|--------|
| Logistic Regression | 0.58 | 0.42 | 0.38 | 0.48 | 0.41 | ❌ Fail |
| Random Forest | 0.62 | 0.45 | 0.41 | 0.51 | 0.44 | ❌ Fail |
| XGBoost | 0.64 | 0.47 | 0.43 | 0.53 | 0.46 | ❌ Fail |
| LightGBM | 0.63 | 0.46 | 0.42 | 0.52 | 0.45 | ❌ Fail |
| Gradient Boosting | 0.61 | 0.44 | 0.40 | 0.50 | 0.43 | ❌ Fail |

**Leakage-aware grouped CV results**: Near-chance ROC-AUC (~0.50)  
**No model approved for serving** - All return 503

## Technology Stack

| Layer | Technologies |
|-------|--------------|
| Backend | Python 3.11, Flask 3, Flask-SQLAlchemy, Flask-JWT-Extended, Flask-Limiter, Flask-CORS, Gunicorn |
| ML | scikit-learn, XGBoost, LightGBM, SHAP, pandas, numpy, scipy |
| Frontend | React 18, TypeScript 5, Vite 6, React Router 7, Axios, Recharts, TailwindCSS 3 |
| Database | PostgreSQL 15 (prod), SQLite (dev), Redis (rate limiting) |
| Deployment | Docker, Docker Compose, Nginx, AWS ECS Fargate, CloudFormation |
| Testing | pytest, Playwright |

## Project Structure

```
airline-no-show-prediction/
├── backend/          # Flask API (app/, run.py, migrate.py, create_admin.py)
├── frontend/         # React + TypeScript (src/, e2e/)
├── ml/               # ML pipeline (training, evaluation, feature engineering, drift)
├── sql/              # schema.sql, views.sql, analytics.sql, bootstrap_roles.sql
├── models/           # Trained artifacts (*.pkl, *_info.json, *_features.json)
├── data/             # training_data.csv
├── docker/           # Dockerfiles, nginx, postgres bootstrap
├── deployment/       # AWS ECS Fargate CloudFormation
├── tests/            # pytest suite
├── app.py            # Legacy Streamlit dashboard (not supported)
├── requirements.txt  # Root deps (includes unused Streamlit/Sphinx)
├── backend/requirements.txt  # Production backend deps
└── docker-compose.yml
```

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | None | Index metadata |
| GET | `/api/docs` | None | Endpoint listing |
| GET | `/api/health` | None | DB + model status |
| POST | `/api/auth/register` | None | Gated by `ALLOW_PUBLIC_REGISTRATION` |
| POST | `/api/auth/login` | None | Sets HttpOnly cookie + CSRF |
| POST | `/api/auth/logout` | JWT | CSRF-protected |
| GET | `/api/auth/profile` | JWT | Current user |
| POST | `/api/predict` | JWT | Single prediction (503 if no approved model) |
| POST | `/api/bulk-predict` | JWT | Batch (max 1000) |
| GET | `/api/models` | JWT | Loaded model info |
| GET | `/api/analytics/overview` | JWT | `days` 1-365 |
| GET | `/api/analytics/routes` | JWT | Optional `route` |
| GET | `/api/analytics/trends` | JWT | `group_by` day/week/month |
| GET | `/api/analytics/channels` | JWT | Channel performance |
| GET | `/api/analytics/high-risk` | Admin | `min_probability`, `limit` |
| GET | `/api/models/performance` | JWT | Reads `*_info.json` |
| GET | `/api/explainability/global` | JWT | SHAP global importance |
| POST | `/api/overbooking/simulate` | JWT | Monte Carlo simulation |
| GET | `/api/monitoring/drift` | JWT | Drift report |

## Installation & Setup

### Prerequisites
- Python 3.11+, Node.js 20+, PostgreSQL 15+ (or Docker)

### Docker (Recommended)
```bash
cp .env.example .env
# Replace ALL secret placeholders in .env
docker compose up --build -d
```

### Local Development

**Backend:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
FLASK_ENV=development python backend/run.py
```

**Frontend:**
```bash
npm --prefix frontend ci
npm --prefix frontend run dev
```

**First Admin User:**
```bash
python backend/create_admin.py
```

## Limitations (Documented)

1. **No approved production model** → predictions return 503
2. **Checked-in model artifacts** use unverified passenger-history features the loader rejects
3. **`get_passenger_summary` is a stub** - no `/analytics/passengers` backend route despite frontend client method
4. **Prediction drift not computed** - no historical prediction baseline
5. **Many SQL views/functions defined but unused** by the Flask app
6. **Two divergent requirement files** - legacy Streamlit `app.py` unused
7. **Duplicate/legacy ML modules** coexist with package versions
8. **No LICENSE file** present despite MIT claim
9. **No CI config**, no `.dockerignore`, weak test isolation

## Future Enhancements

- Approve a model that passes the production gate
- Remove legacy duplicate ML modules and Streamlit app
- Implement passenger summary endpoint and align frontend client
- Add CI/CD, unit tests for frontend, test isolation
- Persist prediction-probability history for prediction-drift monitoring
- Add `.dockerignore` and LICENSE file

---

## Author

**SAMALA RAKSHITH BABU**  

**Contact:**  
- Email: samalarakshithbabu7@gmail.com  
- LinkedIn: https://www.linkedin.com/in/samala-rakshith-babu/  
- GitHub: https://github.com/samalarakshithbabu  

---

## License

MIT License (claimed; no LICENSE file present in repository — requires verification)