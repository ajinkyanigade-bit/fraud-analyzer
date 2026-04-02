# FraudSense AI — Cloud Banking Fraud Analyzer

A single-file, production-ready web application that uses a machine learning ensemble to detect fraudulent banking transactions in real time. Built with Flask and scikit-learn, deployable locally or on Render.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
- [API Reference](#api-reference)
- [ML Model Details](#ml-model-details)
- [Deployment (Render)](#deployment-render)
- [Configuration](#configuration)

---

## Overview

FraudSense AI analyzes banking transactions across six key dimensions — amount, location, device type, transaction type, time of day, and velocity — and returns a risk score, risk level, decision (APPROVE / REVIEW / BLOCK), and a plain-English explanation of the risk factors involved.

On startup, the app trains its model on a synthetic dataset of 20,000 transactions (4% fraud rate) and seeds the dashboard with 40 sample transactions so the UI is immediately usable.

---

## Features

- **Real-time fraud scoring** — POST a transaction and receive a risk score (0–99), risk level (low / medium / high), and a recommended action (APPROVE / REVIEW / BLOCK).
- **Ensemble ML model** — Combines Gradient Boosting (55%), Random Forest (30%), and Isolation Forest (15%) for robust predictions.
- **Explainable decisions** — Every prediction includes a list of human-readable risk factors (e.g., "Tor/anonymising network detected", "Late-night transaction").
- **Live dashboard** — A single-page HTML dashboard with transaction history, active alerts, hourly volume charts, and model metrics.
- **Alert management** — Medium and high-risk transactions automatically generate alerts that can be dismissed via the UI or API.
- **Model retraining** — Trigger a full model retrain at any time via a POST endpoint.
- **CORS enabled** — All responses include permissive CORS headers for easy frontend integration.

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| Web framework | Flask >= 3.0.3 |
| ML models | scikit-learn >= 1.5.0 |
| Numerical ops | NumPy >= 2.0.0 |
| Data handling | pandas >= 2.2.2 |
| Production server | Gunicorn >= 21.2.0 |
| Cloud deployment | Render (render.yaml included) |

---

## Project Structure

```
fraud_analyzer - web/
├── app.py                 # Entire backend: feature engineering, ML model, Flask routes
├── templates/
│   └── index.html         # Single-page dashboard UI
├── requirements.txt       # Python dependencies
├── Procfile               # Gunicorn start command (Heroku-compatible)
├── render.yaml            # Render.com deployment config
└── gitignore              # Git ignore rules
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### Local Setup

1. **Clone / unzip the project** and navigate to the project folder:
   ```bash
   cd "fraud_analyzer - web"
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate          # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```bash
   python app.py
   ```

   The app will automatically open `http://127.0.0.1:5000/dashboard` in your browser. On first run, model training takes roughly 30–60 seconds.

---

## API Reference

### `GET /health`
Returns service status and full model metrics.

### `POST /api/analyze`
Analyze a single transaction without storing it.

**Request body:**
```json
{
  "amount": 4500.00,
  "transaction_type": "wire_transfer",
  "location": "Lagos",
  "device_type": "tor",
  "hour_of_day": 2,
  "txns_last_hour": 8
}
```

**Accepted values:**
| Field | Options |
|---|---|
| `transaction_type` | `bill_payment`, `pos_payment`, `atm_withdrawal`, `online_transfer`, `wire_transfer`, `crypto_exchange` |
| `location` | `Mumbai`, `Delhi`, `New York`, `London`, `Toronto`, `Singapore`, `Dubai`, `Different State`, `Foreign Country`, `Lagos`, `Bucharest`, `Unknown IP`, `High-Risk Region` |
| `device_type` | `known`, `new`, `vpn`, `tor` |
| `hour_of_day` | Integer 0–23 |
| `txns_last_hour` | Integer ≥ 1 |

**Response:**
```json
{
  "risk_score": 87,
  "risk_level": "high",
  "decision": "BLOCK",
  "confidence": 96,
  "risk_factors": ["High-risk geographic location", "Tor/anonymising network detected", "Late-night transaction (hour 2:00)"],
  "model_scores": {
    "gradient_boosting": 91.2,
    "random_forest": 85.4,
    "isolation_forest": 72.1,
    "ensemble": 87.0
  }
}
```

### `GET /api/transactions`
List stored transactions. Supports query params: `risk` (low/medium/high), `search`, `page`, `limit`.

### `POST /api/transactions`
Submit and store a new transaction (same body as `/api/analyze`). Returns the full transaction record with an auto-generated ID.

### `GET /api/transactions/<tid>`
Get a single transaction by ID.

### `POST /api/transactions/<tid>/block`
Remove a transaction from the store (simulate blocking).

### `GET /api/alerts`
List all active (non-dismissed) alerts.

### `POST /api/alerts/<aid>/dismiss`
Dismiss an alert by ID.

### `GET /api/metrics`
Returns aggregate dashboard stats: total transactions, flagged count, review count, clean count, model accuracy, total savings blocked, average risk score, and active alert count.

### `GET /api/model`
Returns model version, training status, per-model accuracy/precision/recall/F1/AUC-ROC, and feature importances.

### `POST /api/model/retrain`
Triggers a full model retrain on a freshly generated synthetic dataset.

### `GET /api/stats/hourly`
Returns transaction volumes and fraud rates broken down by hour of day (0–23).

---

## ML Model Details

### Feature Engineering (16 features)

| Feature | Description |
|---|---|
| `amount_log` | Log-transformed transaction amount |
| `amount_bin` | Bucketed amount (0–4 tiers up to $10,000+) |
| `location_risk` | Numeric risk score per location (0=safe, 3=high-risk) |
| `device_risk` | Numeric risk per device type (0=known, 3=tor) |
| `txn_type_encoded` | Encoded transaction category |
| `is_crypto` / `is_wire` | Binary flags for irreversible transaction types |
| `hour_of_day` | Raw hour (0–23) |
| `is_late_night` | 1 if hour is between 1–5 AM |
| `is_business_hours` | 1 if hour is between 9 AM–5 PM |
| `txns_last_hour` | Transaction velocity |
| `high_velocity` | 1 if velocity > 5 |
| `combined_risk` | location_risk × device_risk interaction |
| `amount_x_velocity` | amount_log × velocity interaction |
| `late_night_foreign` | 1 if late night AND high-risk location |
| `crypto_foreign` | 1 if crypto transaction AND high-risk location |

### Ensemble Logic

| Model | Weight | Purpose |
|---|---|---|
| Gradient Boosting (200 estimators) | 55% | Primary classifier |
| Random Forest (150 estimators) | 30% | Secondary classifier, scaled features |
| Isolation Forest (unsupervised) | 15% | Anomaly detection |

### Decision Thresholds

| Risk Score | Level | Action |
|---|---|---|
| 0–39 | Low | APPROVE |
| 40–69 | Medium | REVIEW |
| 70–84 | High | REVIEW |
| 85–99 | High | BLOCK |

The training dataset uses oversampling of the minority (fraud) class to handle the 4% fraud rate imbalance.

---

## Deployment (Render)

The project includes a ready-to-use `render.yaml`. To deploy:

1. Push the project to a GitHub repository.
2. Go to [Render](https://render.com), create a new **Web Service**, and connect the repository.
3. Render will automatically detect `render.yaml` and configure the service.

The service is configured for:
- Python 3.11.0
- 2 Gunicorn workers
- 120-second worker timeout (to allow model training on cold start)

Alternatively, deploy via the `Procfile` on any Heroku-compatible platform:
```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

---

## Configuration

All configuration is handled via environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Port the server listens on |
| `RENDER` | *(unset)* | Set by Render automatically; disables auto browser-open and debug mode |

When `RENDER` is not set (i.e., running locally), the app runs in Flask debug mode and opens the dashboard in a browser automatically.
