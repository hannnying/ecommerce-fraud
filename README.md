# Current State (More in DRIFT_AND_RETRAINING.md)

Currently, the model is trained on the first 120,000 transactions, while the remaining 30,000 transactions are streamed, and use the trained model to predict whether they are fraudulent or not.

The model is a hybrid-model, consisting of a rule-based classifier and a Logistic Regression model that was trained on the first 120,000 transactions. The rule-based classifer acts as a filter, whereby if a transaction fits any of the criteria, it will be given a label. Transactions that do not pass this filter will be classified by the Logistic Regression model. 

While this hybrid approach is able to catch ~60% of fraud in the WHOLE dataset (), the rules overfit the earlier data. 40% of the first 20,000 transactions could be classified by the rule-based classifier, while only classifying 0.276% of the remaining transactions. Notably, while `fast_purchase` = `time_between_signup_purchase` <= 60 is a crucial characteristic of fraudulent transactions, subsequent transactions no longer behave this way, this is known as data drift. 

Additionally, in January, about 70% of the transactions were fraudulent, whereas from Febraury - December, 4.5% - 7% of the transaction were fraudulent. Hence, there is also a need to address label drift. 

The key changes to be made in this version aim to address data and label drift with ML monitoring, and they include:
1. Offline ML training on initial 10,000 transactions for the initial ML model.
2. Streaming from the 10,001st transaction through Redis.
3. Consumer runs inference on each streamed transaction, and labels arrive after (configurable) each transaction via LABELS_STREAM.
4. Performance monitoring on labelled transactions (and distribution of key model input features) 
5. Periodic retraining


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

Run 
```bash
python3 -m training.train
```

This step trains the Random Forest fraud model, logs the model to MLflow, saves a local model artifact, and exports Redis-backed state locally, which is used for feature engineering.
Skip this step if a trained model and saved Redis state already exist.

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
      "stream_id": "1768548424071-0",
      "transaction_id": "56ec435c-b0ed-4d4a-afae-a20c2dbf5c54",
      "device_txn_idx": "2",
      "first_device_txn": "0",
      "device_time_since_last": "1.0",
      "ip_switched": "0",
      "sex_changed": "0",
      "age_diff": "0",
      "scaled_age_diff": "0.0",
      "identity_changed": "0",
      "scaled_device_purchase_diff": "0.0",
      "device_txn_velocity_1m": "1",
      "device_txn_velocity_5m": "1",
      "device_txn_velocity_1h": "1",
      "device_txn_velocity_24h": "1",
      "fast_purchase": "1",
      "log_time_setup_to_txn_seconds": "0.6931471805599453",
      "ip_txn_idx": "2",
      "txn_velocity_1h": "30",
      "predicted_class": "1",
      "fraud_probability": "0.9123333333333333"
    },
    ...
  ]
}
```

Notes:
	•	predicted_class: 1 = fraud, 0 = non-fraud
	•	fraud_probability: probability of being fraudulent
	•	Returned transactions include all computed features used by the model

