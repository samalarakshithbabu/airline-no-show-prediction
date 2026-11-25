# Runtime Validation Report

**Generated:** 2026-11-22  
**Target:** `C:\Users\samal\Downloads\Airline-No-Show-Prediction-System-main`  
**Status:** Research prototype; production deployment and predictions are not approved.

This report replaces the stale 2026-11-21 report, which recorded 43 Python tests.

## Current verification

| Check | Result | Evidence |
|---|---|---|
| Local startup and health | PASS | Flask `/api/health` returned HTTP 200; database reported healthy and `prediction_enabled: false`. Vite returned HTTP 200. Local test servers were stopped after browser testing. |
| Python suite | PASS | `python -m pytest -q` - 52 passed. |
| Frontend build and TypeScript | PASS | `npm --prefix frontend run build` - `tsc` and Vite production build succeeded. |
| Browser E2E | PASS | `npm --prefix frontend run test:e2e -- --reporter=list` - 3 passed. |
| Python syntax | PASS | `python -m compileall -q backend ml tests`. |
| Compose/AWS template syntax | PASS | `docker-compose.yml` and `deployment/aws/ecs-fargate.yml` parsed as YAML; CloudFormation intrinsic tags were preserved by a custom parser. |
| Python dependency CVEs | PASS | `python -m pip_audit -r requirements.txt` and `python -m pip_audit -r backend/requirements.txt` reported no known vulnerabilities. |
| Frontend production dependency CVEs | PASS | `npm --prefix frontend audit --omit=dev` reported 0 vulnerabilities. |
| Shared Python environment consistency | FAIL | `python -m pip check` reports conflicts in globally installed packages, including protobuf, numpy, pydantic, Starlette, dacite, pymongo, and TensorFlow. This is not an isolated project environment; the project test suite passed. |
| Docker/AWS validation | NOT RUN | Docker, `cfn-lint`, and AWS CLI are unavailable. No image build or AWS deployment was performed. |

The test suite includes model approval, disabled prediction, feature-contract, authentication, authorization, security-control, and API checks. `tests/test_runtime_analytics_flow.py` verifies ordinary-user denial (HTTP 403) and admin access (HTTP 200) for restricted high-risk analytics. Browser journeys cover ordinary-user analytics/overbooking, authenticated page navigation, and the disabled prediction flow. The browser test verifies the prediction form collects booking date, flight date, departure airport, and arrival airport.

## Model and serving status

There is no `models/production_model.json`. The model directory contains legacy research artifacts, none approved for serving. The API and UI keep predictions disabled; the health endpoint and non-prediction functionality remain available. Approval checks require an explicit supported-model pointer, passing quality-gate metadata, matching artifact metadata, and a valid feature contract before artifact deserialization. Prediction endpoints return HTTP 503 when no approved model is available. Requests cannot select arbitrary model identifiers, and unverified passenger-history features are excluded.

The active passenger-grouped evaluation used 3,000 rows, 1,872 passengers, and a 600-row passenger-disjoint holdout. XGBoost was selected by five-fold grouped-CV PR-AUC:

| Metric | Grouped CV | Held-out at 0.50 threshold |
|---|---:|---:|
| ROC-AUC | 0.5346 | 0.5036 |
| PR-AUC | 0.3814 | 0.3632 |
| Brier score | 0.2495 | 0.2322 |
| Precision / recall / F1 | 0.3870 / 0.4787 / 0.4280 | 0 / 0 / 0 |

At the training-OOF-selected 0.3179 threshold, held-out precision was 0.3647, recall 0.9816, F1 0.5318, and specificity 0.0313 (371 false positives and 12 true negatives). These are not operational recommendations. The discrimination and precision gates fail; no candidate is approved. Metrics were not changed to pass the gate.

These are the existing recorded grouped-validation results, not a training rerun during this fix. The recent preprocessing correction only derives a missing inference feature from encoded cabin input; no model was promoted or retrained.

## Security and feature contract

- Browser authentication uses an HttpOnly access cookie, CSRF protection, SameSite=Lax, and Secure cookies in production. Bearer authentication remains accepted for compatibility.
- Admin authorization checks the current active database user and role.
- Production rate limiting requires shared Redis-compatible storage; in-memory storage is rejected for production. Trusted proxy hops must be configured.
- The backend container is configured to run as a non-root user; the image was not built here.
- The prediction form/API contract now collects dates and airport categories required by active training features. Missing and non-finite required features are rejected, and unknown airports are not silently encoded. Passenger-history aggregates are not used.
- A feature-contract test exposed a missing `is_first_class` value when cabin class was supplied as an encoded value. The encoded preprocessing path now derives it; the full suite passes.
- The supplied dataset has no verified provenance, license, or operational target definition. Passenger-history feature timing is unknown and remains excluded. Validation is grouped by passenger to avoid cross-split passenger leakage.

## Remaining gaps and blockers

- No model meets the approval gate, so production prediction remains unavailable by design.
- No live PostgreSQL, Redis production service, Docker runtime, or AWS account/environment was available. Database migrations, role setup, container networking/health, secrets/IAM, TLS, and the ECS deployment remain unverified.
- Booking management lacks a dedicated frontend page and booking CRUD API.
- Monitoring is limited to feature-distribution drift; without approved predictions and labeled production outcomes, prediction drift and model performance cannot be measured.
- The shared global Python environment has dependency conflicts; reproduce tests and `pip check` in an isolated environment before deployment.
- Compose and CloudFormation syntax parsing is not equivalent to `docker compose config`, `cfn-lint`, an image scan, or a live deployment.

## Startup commands

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:FLASK_ENV = "development"
python backend\run.py
```

In another PowerShell terminal:

```powershell
npm --prefix frontend ci
npm --prefix frontend run dev
```

Compose startup, which was not validated because Docker is unavailable:

```powershell
docker compose up --build -d
```

Do not manually create an approval pointer or serve any of the legacy model artifacts.
