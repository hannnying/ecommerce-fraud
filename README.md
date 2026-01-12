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

### Step 2: Train model (also ensures that device_state is updated for model inference)
Before running the app, you need to train the Logistic Regression model.
Run:
```bash
python3 -m training.train_v2 \
    --model dt \
    [--save]
```

```text
What this does:
This script performs initial offline training of the fraud detection model:

1. Fits and saves (or logs) the `FraudDataPreprocessor`:
   - Learns scaling and preprocessing parameters from training data
   - Ensures consistent transformations during real-time inference
   - Always logged to MLflow; saved to disk only if `--save` is specified

2. Trains a Decision Tree model using the first 50,000 time-ordered transactions:
   - Random Undersampling to address class imbalance
   - Hyperparameter tuning via cross-validation
   - Performance evaluation on a held-out 10,000-transaction test set
   - Training metrics and hyperparameters automatically logged to MLflow

3. Persists trained artifacts to the `models/` directory **only** if `--save` is provided:
   - `models/preprocessor.pkl`
   - `models/dt_model.pkl`
```

To launch the MLflow UI, run:
```bash
mlflow server --port 8080
```

### Step 3: Stream transactions as events
Run the producer process in another terminal:
```bash
python3 -m api.producer
```

### Step 4: Start worker
Run the worker process in another terminal:
```bash
python3 -m api.consumer
```

### Step 5: Start FastAPI Backend
Run FastAPI server in a separate terminal:
```bash
uvicorn api.main:app --reload
```


### Step 6: Start Streamlit UI
Run Streamlit server in a separate terminal:
```bash
streamlit run app.py
```

## Run the Application with Docker

Run:
```bash
docker compose up --build
```

visit: http://localhost:8501

## FastAPI Endpoints

### 2. /results/recent– Fetch the k Most Recent Processed Transactions

**Description:**
Fetches the latest processed transactions and their predicted fraud classes from the RESULT_STREAM.

**Method: GET**

**Response Example:**

```bash
{
  "count": 10,
  "results": [
    {
      "stream_id": "1768261775779-0",
      "transaction_id": "d3735538-7be5-42d6-891b-e612ddf8ab87",
      "txn_count": "1",
      "log_time_setup_to_txn_seconds": "14.666988811177923",
      "first_device_transaction": "1",
      "scaled_device_purchase_diff": "-1.0",
      "repeated_device_purchase": "0",
      "identity_changed": "0",
      "predicted_class": "0",
      "fraud_probability": "0.21608527131782945"
    },
    ...
  ]
}
```

Notes:
	•	predicted_class: 1 = fraud, 0 = non-fraud
	•	fraud_probability: probability of being fraudulent
	•	Returned transactions include all computed features used by the model

