# E-commerce Fraud Detection Project
This project implements a fraud detection pipeline using:
	•	FastAPI – backend API for submitting transactions and reading results
	•	Redis Streams – message broker for queuing transactions and results
	•	Worker – separate process for feature engineering and running inference
	•	Streamlit – frontend for simulating and viewing transaction predictions
	•	Scikit-learn – Logistic Regression model

Ensure that you are in /ecommerce-fraud directory.

# Directory Structure
ecommerce-fraud/
├── api/
|   ├── dependencies.py    
|   ├── inference.py    
│   ├── main.py           # FastAPI application
│   ├── schemas.py        # Request/response models
│   └── worker.py        
├── models/               # Saved model artifacts
│   └── (models saved here)
├── src/                  # Core modules
|   ├── config.py                  
│   ├── data_prep_v2.py
│   ├── evaluation.py
│   ├── models.py
│   ├── preprocessing.py
│   └── utils.py
├── data/                 # Training/test datasets
│   ├── Fraud_Data.csv
│   ├── gdp_usd.xlsx
│   └── IpAddress_to_Country.csv
├── app.py                # Streamlit 
├── compose.yaml.py                
├── Dockerfile.backend              
├── Dockerfile.frontend              
├── requirements.txt              
└── train.py              # Training script
              

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

Before running the app, you need to train the Logistic Regression model.
Run:
```bash
python3 train.py \
    --model logistic_regression \
    --resampling random_undersampling \
    --columns "txn_count_device_percentile,time_setup_to_txn_seconds,device_ip_consistency,source_device_consistency,time_since_last_device_id_txn,purchase_deviation_from_device_mean,sex_encoded" \
    --save
```

What this does:
1.	Fits and saves the FraudFeatureEngineer pipeline
2.	Fits and saves the FraudDataPreprocessor pipeline
3.	Trains a Logistic Regression model using:
•	Random undersampling for class imbalance
•	Only the 7 specified features
•	Custom hyperparameters (C=0.001, max_iter=1000, penalty=l2, solver=liblinear)
4.	Saves all components to the models/ directory:
•	models/feature_engineer.pkl
•	models/preprocessor.pkl
•	models/logistic_regression_model.pkl


## Run the Application Locally

### Step 1: Start Redis 
Start Redis Using Docker:
```bash
docker run --name fraud-redis  -p 6379:6379 -d redis:latest
```

### Step 2: Start FastAPI Backend
Run FastAPI server in a separate terminal:
```bash
uvicorn api.main:app --reload
```

### Step 3: Start worker
Run the worker process in another terminal:
```bash
python3 -m api.worker
```

### Step 4: Start Streamlit Simulation
Run Streamlit to simulate transactions and view predictions
```bash
stremlit run app.py
```

## Running on Docker

```bash
docker compose build
docker compose up
```