# E-commerce Fraud Detection Project

```text
This project implements a fraud detection pipeline using:
	•	FastAPI – backend API for submitting transactions and reading results
	•	Redis Streams – message broker for queuing transactions and results
	•	Worker – separate process for feature engineering and running inference
	•	Streamlit – frontend for simulating and viewing transaction predictions
	•	Scikit-learn – Logistic Regression model
```

Ensure that you are in /ecommerce-fraud directory.

```text
ecommerce-fraud/
├── api/                          # Online / real-time components
│   ├── main.py                   # FastAPI application (API entrypoint)
│   ├── producer.py               # Simulates real-time transactions (CSV → Redis stream)
│   └── consumer.py               # Consumes transactions, computes features, runs inference
│
├── src/                          # Core ML + data pipeline logic
│   ├── feature_engineering/
│   │   └── engineer.py           # Feature computation using Redis-backed device state
│   │
│   ├── models/
│   │   └── models_v2.py          # Fraud models & hyperparameter tuning
│   │
│   ├── preprocessing.py          # Scaling & preprocessing logic
│   │
│   └── state/                    # Redis-backed state stores
│       ├── redis_store.py        # Device state (Redis hash)
│       └── serializers.py        # Serialize / deserialize Redis payloads
│
├── training/
│   └── train_v2.py               # Initial offline training on time-ordered data
│
├── models/                       # Persisted artifacts
│   ├── feature_engineer.pkl
│   ├── preprocessor.pkl
│   └── fraud_model.pkl
│
├── data/                         # Raw datasets (offline)
│   ├── Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│   └── gdp_usd.xlsx
│
├── app.py                        # Streamlit UI (results visualization)
│
├── docker/                       # Docker assets (planned)
│   ├── Dockerfile.backend        # FastAPI backend (needs update)
│   ├── Dockerfile.frontend       # Streamlit frontend (needs update)
│   └── compose.yaml              # Docker Compose (WIP)
│
├── requirements.txt              # Python dependencies
└── README.md
```

# Installation

### Step 1: Create the Virtual Environmet
```bash
python3 -m venv .ecommerce-venv
```

### Step 2: Activate Virtual Environment

Mac:
```bash
source ecommerce-venv/bin/activate
```

### Step 3: Verify Installation
Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

# Usage


## Run the Application Locally

### Step 1: Start Redis 
Start Redis Using Docker:
```bash
docker run --name fraud-redis  -p 6379:6379 -d redis:latest
```

### Step 2: Train model
Before running the app, you need to train the Logistic Regression model.
Run:
```bash
python3 -m training.train_v2 \
    --model logistic_regression \
    --save
```

```text
What this does:
This script performs initial offline training of the fraud detection model:

1. Fits and saves the `FraudDataPreprocessor`:
   - Learns scaling and preprocessing parameters from training data
   - Ensures consistent transformations during real-time inference

2. Trains a Logistic Regression model using the first 50,000 time-ordered transactions:
   - SMOTE to address class imbalance
   - Hyperparameter tuning via cross-validation
   - Performance evaluation on a held-out 5,000-transaction test set

3. Persists trained artifacts to the `models/` directory:
   - `models/preprocessor.pkl`
   - `models/logistic_regression_model.pkl`
```

### Step 3: Start FastAPI Backend
Run FastAPI server in a separate terminal:
```bash
uvicorn api.main:app --reload
```

### Step 4: Start worker
Run the worker process in another terminal:
```bash
python3 -m api.consumer
```

## FastAPI Endpoints

### 1. /stream – Start Transaction Simulation

**Description:**
Starts pushing transactions from the CSV to the Redis TRANSACTIONS_STREAM to simulate real-time events.

**Method: POST**

**Response Example:**

```bash
{
  "status": "queued",
  "transaction_ids": [
    "92d999d0-215a-4d7e-ac57-d6eec00d9111",
    "f31b2a8c-8f4b-4d2c-9c23-12a3b4e4d5f7",
    ...
  ]
}
```

These `transaction_ids` correspond to the entries in the transaction stream.


### 2. /get – Fetch Processed Transactions

**Description:**
Fetches the latest processed transactions and their predicted fraud classes from the RESULT_STREAM.

**Method: GET**

**Response Example:**

```bash
[
  {
    "transaction_id": "92d999d0-215a-4d7e-ac57-d6eec00d9111",
    "txn_count": 3,
    "device_ip_consistency": true,
    "device_source_consistency": true,
    "time_setup_to_txn_seconds": 1777456.0,
    "time_since_last_device_txn": 0.0,
    "purchase_deviation_from_device_mean": 0.0,
    "device_lifespan": 0.0,
    "predicted_class": 1,
    "fraud_probability": 0.6598676441447116
  },
  ...
]
```

Notes:
	•	predicted_class: 1 = fraud, 0 = non-fraud
	•	fraud_probability: probability of being fraudulent
	•	Returned transactions include all computed features used by the model
