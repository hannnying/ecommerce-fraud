# API Deployment Guide

This guide walks you through training models and deploying them via FastAPI for real-time fraud detection inference.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Training and Saving Models](#training-and-saving-models)
3. [Running the FastAPI Service](#running-the-fastapi-service)
4. [Testing the API](#testing-the-api)
5. [Running the Transaction Demo UI](#running-the-transaction-demo-ui)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Activate Virtual Environment

```bash
cd /Users/han-ying/Downloads/ecommerce-fraud
source ecommece_venv/bin/activate
```

### 2. Verify Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Required packages for API:
- `fastapi`
- `uvicorn[standard]`
- `pydantic`

If not installed:

```bash
pip install fastapi uvicorn[standard] pydantic
```

### 3. Directory Structure

Your project should have this structure:

```
ecommerce-fraud/
├── api/
│   ├── main.py           # FastAPI application
│   ├── inference.py      # Model loading and prediction
│   └── schemas.py        # Request/response models
├── models/               # Saved model artifacts
│   └── (models saved here)
├── src/                  # Core modules
│   ├── models.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── config.py
├── data/                 # Training/test datasets
│   ├── train.csv
│   └── test.csv
├── train.py              # Training script
└── app.py                # Streamlit dashboard
```

---

## Training and Saving Models

### Basic Usage

Train the default model (Logistic Regression) and save all required components:

```bash
python3 train.py --save
```

**What this does**:
- Fits and saves `FraudFeatureEngineer` to `models/feature_engineer.pkl`
- Fits and saves `FraudDataPreprocessor` to `models/preprocessor.pkl`
- Trains and saves the model to `models/logistic_regression_model.pkl`

All three components are automatically fitted and saved when you run `train.py` with the `--save` flag, so you don't need to run any separate data preprocessing scripts.

---

### Advanced Training Options

#### 1. Train with Specific Model Type

```bash
# Logistic Regression
python3 train.py --model logistic_regression --save

# Random Forest
python3 train.py --model random_forest --save

# XGBoost
python3 train.py --model xgboost --save
```

#### 2. Train with Resampling Techniques

Handle class imbalance with different resampling methods:

```bash
# Random undersampling (default)
python3 train.py --model logistic_regression --resampling random_undersampling --save

# Edited Nearest Neighbors (ENN)
python3 train.py --model logistic_regression --resampling enn --save

# SMOTE (Synthetic Minority Over-sampling)
python3 train.py --model logistic_regression --resampling smote --save

# SMOTE + ENN combination
python3 train.py --model logistic_regression --resampling smoteenn --save
```

#### 3. Train with Feature Selection

Select specific features for training:

```bash
python3 train.py \
  --model logistic_regression \
  --resampling random_undersampling \
  --columns "txn_count_device_percentile,time_setup_to_txn_seconds,device_ip_consistency,source_device_consistency,time_since_last_device_id_txn,purchase_deviation_from_device_mean,sex_encoded" \
  --save
```

**Note**: Column names should be comma-separated without spaces around commas.

#### 4. Train with Custom Hyperparameters

Override default hyperparameters with custom values:

```bash
python3 train.py \
  --model logistic_regression \
  --resampling random_undersampling \
  --columns "txn_count_device_percentile,time_setup_to_txn_seconds,device_ip_consistency,source_device_consistency,time_since_last_device_id_txn,purchase_deviation_from_device_mean,sex_encoded" \
  --params '{"C": 0.001, "max_iter": 1000, "penalty": "l2", "random_state": 42, "solver": "liblinear"}' \
  --save
```

**Important**:
- Parameters must be valid JSON format
- Use single quotes around the JSON string
- Use double quotes inside the JSON

---

### Complete Example

Train a Logistic Regression model with all custom options:

```bash
python3 train.py \
  --model logistic_regression \
  --resampling random_undersampling \
  --columns "txn_count_device_percentile,time_setup_to_txn_seconds,device_ip_consistency,source_device_consistency,time_since_last_device_id_txn,purchase_deviation_from_device_mean,sex_encoded" \
  --params '{"C": 0.001, "max_iter": 1000, "penalty": "l2", "random_state": 42, "solver": "liblinear"}' \
  --save
```

**What this does**:
1. Fits and saves `FraudFeatureEngineer` (feature engineering pipeline)
2. Fits and saves `FraudDataPreprocessor` (preprocessing pipeline)
3. Trains a Logistic Regression model with:
   - Random undersampling for class imbalance
   - Only the 7 specified features
   - Custom hyperparameters (C=0.001, L2 penalty, etc.)
4. Saves all three components to the `models/` directory:
   - `models/feature_engineer.pkl`
   - `models/preprocessor.pkl`
   - `models/logistic_regression_model.pkl`

---

### Understanding Training Output

After running the training command, you'll see:

```
======================================================================
Training Logistic Regression (L1 Regularization)
Description: Best model for fraud detection (high recall, low overfitting)
======================================================================

Loading and preprocessing data...
Initializing logistic_regression model...
Resampling technique: random_undersampling
Training model...
Training complete!

Making predictions...

======================================================================
Logistic Regression (L1 Regularization) Evaluation Results
======================================================================
Accuracy:  0.9234
Precision: 0.8123
Recall:    0.7456
F1 Score:  0.7774
ROC-AUC:   0.8912
PR-AUC:    0.7623

              precision    recall  f1-score   support
           0     0.9567    0.9412    0.9489     18450
           1     0.8123    0.7456    0.7774      1850

======================================================================
Top 10 Most Important Features
======================================================================
                              feature  coefficient
  txn_count_device_percentile       2.345
  device_ip_consistency            1.892
  time_since_last_device_id_txn    1.654
  ...

Model saved to models/logistic_regression_model.pkl

======================================================================
Training Complete!
======================================================================
```

---

### Comparing Multiple Models

Compare all available models with the same resampling technique:

```bash
python train.py --compare --resampling random_undersampling
```

This will:
- Train all models (Logistic Regression, Random Forest, XGBoost)
- Compare performance metrics
- Save comparison results to `results/model_comparison.csv`

---

## Running the FastAPI Service

### 1. Navigate to Project Root

**CRITICAL**: Always run FastAPI from the project root directory:

```bash
cd /Users/han-ying/Downloads/ecommerce-fraud
```

**Why?** The API needs to import the `src` package and access saved models using relative paths.

---

### 2. Start the FastAPI Server

```bash
uvicorn api.main:app --reload
```

**Command breakdown**:
- `uvicorn` - ASGI server for running FastAPI
- `api.main:app` - Module path to the FastAPI app instance
- `--reload` - Auto-restart on code changes (development only)

---

### 3. Verify Server is Running

You should see output like:

```
INFO:     Will watch for changes in these directories: ['/Users/han-ying/Downloads/ecommerce-fraud']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

The API is now running at: **http://127.0.0.1:8000**

---

### 4. Production Configuration (Optional)

For production deployment, use different settings:

```bash
# Bind to all network interfaces, use multiple workers
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Options**:
- `--host 0.0.0.0` - Accept connections from any IP
- `--port 8000` - Custom port (default is 8000)
- `--workers 4` - Number of worker processes
- Remove `--reload` in production

---

## Testing the API

### Method 1: Interactive API Documentation (Recommended)

FastAPI automatically generates interactive documentation.

#### 1. Open in Browser

Navigate to: **http://127.0.0.1:8000/docs**

You'll see the Swagger UI with all available endpoints.

#### 2. Test the Health Endpoint

1. Find the **GET /health** endpoint
2. Click "Try it out"
3. Click "Execute"
4. You should see:
   ```json
   {
     "message": "app is running!"
   }
   ```

#### 3. Test the Predict Endpoint

1. Find the **POST /predict** endpoint
2. Click "Try it out"
3. In the request body, enter a sample transaction:

   ```json
   {
     "purchase_value": 150.50,
     "age": 35,
     "txn_count_device_percentile": 0.75,
     "time_setup_to_txn_seconds": 3600,
     "device_ip_consistency": 1,
     "source_device_consistency": 1,
     "time_since_last_device_id_txn": 86400,
     "purchase_deviation_from_device_mean": 0.2,
     "sex_encoded": 1
   }
   ```

4. Click "Execute"
5. Check the response:

   ```json
   {
     "predicted_class": 0,
     "probability": 0.1234
   }
   ```

   - `predicted_class`: 0 = legitimate, 1 = fraud
   - `probability`: Fraud probability (0.0 to 1.0)

---

### Method 2: Using curl (Command Line)

#### Test Health Endpoint

```bash
curl http://127.0.0.1:8000/health
```

**Expected response**:
```json
{"message":"app is running!"}
```

#### Test Predict Endpoint

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "purchase_value": 150.50,
    "age": 35,
    "txn_count_device_percentile": 0.75,
    "time_setup_to_txn_seconds": 3600,
    "device_ip_consistency": 1,
    "source_device_consistency": 1,
    "time_since_last_device_id_txn": 86400,
    "purchase_deviation_from_device_mean": 0.2,
    "sex_encoded": 1
  }'
```

**Expected response**:
```json
{
  "predicted_class": 0,
  "probability": 0.1234
}
```

---

### Method 3: Using Python Requests Library

Create a test script `test_api.py`:

```python
import requests
import json

# API endpoint
API_URL = "http://127.0.0.1:8000"

# Test health endpoint
def test_health():
    response = requests.get(f"{API_URL}/health")
    print(f"Health Check: {response.json()}")
    return response.status_code == 200

# Test prediction endpoint
def test_predict():
    transaction = {
        "purchase_value": 150.50,
        "age": 35,
        "txn_count_device_percentile": 0.75,
        "time_setup_to_txn_seconds": 3600,
        "device_ip_consistency": 1,
        "source_device_consistency": 1,
        "time_since_last_device_id_txn": 86400,
        "purchase_deviation_from_device_mean": 0.2,
        "sex_encoded": 1
    }

    response = requests.post(
        f"{API_URL}/predict",
        json=transaction
    )

    print(f"Prediction: {response.json()}")
    return response.status_code == 200

if __name__ == "__main__":
    print("Testing FastAPI...")

    if test_health():
        print("✓ Health check passed")

    if test_predict():
        print("✓ Prediction test passed")
```

Run the test:

```bash
python test_api.py
```

---

## Running the Transaction Demo UI

The project includes an interactive Streamlit UI (`app.py`) that simulates a customer making a transaction. Users fill out transaction details and see in real-time whether the transaction would be approved or flagged as fraudulent.

### What the UI Does

- **Simulates a checkout/payment form** where users enter:
  - Purchase amount
  - Customer age
  - Gender
  - Browser type
  - Traffic source
- **Calls the fraud detection API** when user clicks "Make Transaction"
- **Shows immediate feedback**:
  - ✅ **Transaction Approved** - if fraud probability is low
  - ❌ **Transaction Declined - Fraud Detected** - if flagged as fraudulent
  - Displays the fraud probability percentage

### Running Both Services

You need to run **both the FastAPI backend and Streamlit frontend** simultaneously.

#### Step 1: Start the FastAPI Backend

Open **Terminal 1**:

```bash
cd /Users/han-ying/Downloads/ecommerce-fraud
source ecommece_venv/bin/activate
uvicorn api.main:app --reload
```

**Expected output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

The API is now running at **http://127.0.0.1:8000**

---

#### Step 2: Start the Streamlit UI

Open **Terminal 2** (new terminal window):

```bash
cd /Users/han-ying/Downloads/ecommerce-fraud
source ecommece_venv/bin/activate
streamlit run app.py
```

**Expected output**:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

The UI will automatically open in your browser at **http://localhost:8501**

---

#### Step 3: Test a Transaction

1. **Fill in the transaction form**:
   - Enter a purchase amount (e.g., 150)
   - Select your age (e.g., 35)
   - Choose gender (M/F)
   - Select browser (Chrome, Firefox, Safari, etc.)
   - Select traffic source (SEO, Direct, Ads)

2. **Click "Make Transaction"**

3. **See the result**:
   - If legitimate: "✅ Transaction Approved! (Fraud probability: 12.3%)"
   - If fraudulent: "❌ Transaction Declined - Fraud Detected! (Fraud probability: 87.6%)"

---

### Quick Start (All-in-One)

If you want to start both services quickly:

```bash
# Navigate to project
cd /Users/han-ying/Downloads/ecommerce-fraud
source ecommece_venv/bin/activate

# Terminal 1: Start FastAPI (run this first)
uvicorn api.main:app --reload

# Terminal 2: Start Streamlit (in a new terminal)
streamlit run app.py
```

**Access**:
- FastAPI docs: http://127.0.0.1:8000/docs
- Streamlit UI: http://localhost:8501

---

### Troubleshooting the UI

#### Issue: "Error: Connection refused"

**Cause**: FastAPI backend is not running.

**Solution**: Make sure Terminal 1 with FastAPI is still running. Check for the message:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

#### Issue: "422 Unprocessable Entity"

**Cause**: Missing required fields or wrong data types.

**Solution**: Ensure all form fields are filled out before clicking "Make Transaction".

---

#### Issue: Streamlit shows blank/white screen

**Cause**: Port conflict or installation issue.

**Solution**:
```bash
# Kill existing Streamlit processes
pkill -f streamlit

# Restart Streamlit
streamlit run app.py
```

---

### Environment-Based Configuration (Optional)

For more flexible deployment, create a `.env` file:

```bash
# .env
API_URL=http://127.0.0.1:8000
ENV=development
```

Load in Streamlit:

```python
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv('API_URL', 'http://127.0.0.1:8000')
```

---

## Troubleshooting

### Issue 1: ModuleNotFoundError: No module named 'src'

**Error**:
```
ModuleNotFoundError: No module named 'src'
```

**Cause**: Running FastAPI from wrong directory or Python can't find `src` module.

**Solutions**:

**A. Run from project root (Recommended)**:
```bash
cd /Users/han-ying/Downloads/ecommerce-fraud
uvicorn api.main:app --reload
```

**B. Install project as package**:
```bash
cd /Users/han-ying/Downloads/ecommerce-fraud
pip install -e .
```

This requires a `setup.py` file in the root.

---

### Issue 2: FileNotFoundError: models/logistic_regression_model.pkl

**Error**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/logistic_regression_model.pkl'
```

**Cause**: Model hasn't been trained and saved yet.

**Solution**:

Train and save a model first:
```bash
python train.py --model logistic_regression --save
```

Verify the model file exists:
```bash
ls -l models/
```

---

### Issue 3: API Returns 500 Internal Server Error

**Cause**: Model loading or prediction failed.

**Debugging**:

1. Check FastAPI terminal output for full error traceback
2. Check model file is not corrupted:
   ```python
   import pickle
   with open('models/logistic_regression_model.pkl', 'rb') as f:
       model = pickle.load(f)
   print(model)
   ```
3. Verify input data format matches model expectations
4. Check all required features are present in the request

---

### Issue 4: Port Already in Use

**Error**:
```
ERROR:    [Errno 48] Address already in use
```

**Cause**: Another process is using port 8000.

**Solutions**:

**A. Kill the existing process**:
```bash
# Find process using port 8000
lsof -ti:8000

# Kill it
kill -9 $(lsof -ti:8000)
```

**B. Use a different port**:
```bash
uvicorn api.main:app --reload --port 8001
```

---

### Issue 5: Connection Refused When Testing

**Error**:
```
requests.exceptions.ConnectionError: Connection refused
```

**Cause**: FastAPI server is not running.

**Solution**:

1. Verify server is running:
   ```bash
   curl http://127.0.0.1:8000/health
   ```

2. Check the server terminal for errors

3. Restart the server if needed

---

### Issue 6: Invalid Input Data Format

**Error**:
```
422 Unprocessable Entity
```

**Cause**: Request data doesn't match the expected schema.

**Solution**:

1. Check `api/schemas.py` for required fields
2. Ensure all required fields are included in the request
3. Verify data types match (int, float, string)
4. Check for typos in field names

Example valid request:
```json
{
  "purchase_value": 150.50,      # float, required
  "age": 35,                      # int, required
  "sex_encoded": 1                # int, required
  // ... all other required fields
}
```

---

### Issue 7: Model Predictions Look Wrong

**Debugging Steps**:

1. **Test with known data**:
   Load a row from your test set and compare:
   - Direct model prediction (in Python)
   - API prediction (via endpoint)

   They should match exactly.

2. **Check preprocessing**:
   Ensure API preprocessing matches training preprocessing.

3. **Verify feature order**:
   Model expects features in specific order. Check:
   ```python
   print(model.feature_names_in_)  # sklearn models
   ```

4. **Check for data leakage**:
   Make sure you're not including target variable or derived features.

---

## Additional Resources

### API Documentation

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Project Documentation

- `QUICKSTART.md` - Quick start guide for the project
- `docs/` - Additional documentation
- `src/config.py` - Configuration settings

### Model Information

View available models and their configurations:

```bash
python -c "from src.config import MODEL_CONFIGS; import json; print(json.dumps(MODEL_CONFIGS, indent=2))"
```

---

## Quick Reference

### Essential Commands

```bash
# Navigate to project root
cd /Users/han-ying/Downloads/ecommerce-fraud

# Activate virtual environment
source ecommece_venv/bin/activate

# Train and save model
python train.py --model logistic_regression --save

# Run FastAPI (Terminal 1)
uvicorn api.main:app --reload

# Run Streamlit UI (Terminal 2 - opens demo at http://localhost:8501)
streamlit run app.py

# Test API health
curl http://127.0.0.1:8000/health

# View API docs
open http://127.0.0.1:8000/docs
```

### Testing Transactions via UI

1. **Start both services** (FastAPI in Terminal 1, Streamlit in Terminal 2)
2. **Open browser** to http://localhost:8501
3. **Fill in transaction details** (amount, age, gender, browser, source)
4. **Click "Make Transaction"**
5. **See result**: ✅ Approved or ❌ Declined (with fraud probability)

---

## Next Steps

1. ✅ Train your model with desired configuration
2. ✅ Save the model to `models/` directory
3. ✅ Start the FastAPI service
4. ✅ Test endpoints using /docs interface
5. ✅ Integrate with Streamlit dashboard
6. 🚀 Deploy to production (when ready)

---

## Support

If you encounter issues not covered in this guide:

1. Check FastAPI terminal output for detailed error messages
2. Review `api/inference.py` for model loading logic
3. Verify all paths are relative to project root
4. Ensure virtual environment is activated
5. Check that all dependencies are installed

For model training issues, refer to `QUICKSTART.md`.
