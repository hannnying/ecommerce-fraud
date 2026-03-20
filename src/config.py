"""
Configuration file for fraud detection project.

Contains paths, constants, and hyperparameters used throughout the project.
Only uses environment variables for settings that differ between environments.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# ============================================================================
# Environment Configuration (from env vars)
# ============================================================================
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# ============================================================================
# Project Paths (constants - work the same everywhere)
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# Data Paths (constants)
# ============================================================================
DATA_DIR = PROJECT_ROOT / "data"

# Input data files
RAW_DATA_PATH = DATA_DIR / "Fraud_Data.csv"
IP_COUNTRY_PATH = DATA_DIR / "IpAddress_to_Country.csv"
GDP_DATA_PATH = DATA_DIR / "gdp_usd.xlsx"
FRAUD_WITH_COUNTRY_PATH = DATA_DIR / "fraud_with_country.csv"

# Processed data files
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
RAW_TRAIN_PATH = DATA_DIR / "raw_train.csv"
RAW_TEST_PATH = DATA_DIR / "raw_test.csv"

# ============================================================================
# Feature Definitions
# ============================================================================

# All numerical features (continuous values)
numerical_features = [
    'device_txn_idx',
    'device_time_since_last_s',
    'device_age_hours',
    'device_txn_velocity_24h',
    'global_txn_velocity_24h',
    'country_txn_velocity_24h',
    'time_setup_to_txn_seconds',
    'purchase_hour',
    'amount_per_age'
]

# All categorical features (discrete categories)
categorical_features = [
    'source',
    'browser'
]

# All boolean features (stored as int 0/1)
boolean_features = [
    'signup_before_first_device_txn',
    'repeated_device_purchase',
    'purchase_spike',
    'identity_changed',
    'prev_is_fraud',
    'is_late_night',
    'under_18',
    'age_18_25',
    'age_26_35',
    'age_36_50',
    '50_above',
    'unknown_country'
]

# ============================================================================
# Model-Specific Feature Sets
# ============================================================================

# SEEN DEVICES MODEL
# For transactions from devices with historical data
# Includes device-specific features that require transaction history
seen_device_numerical_features = [
    'device_txn_idx',
    'device_time_since_last_s',
    'device_age_hours',
    'global_txn_velocity_24h',
    'country_txn_velocity_24h'
]

seen_device_boolean_features = [
    'repeated_device_purchase',
    'purchase_spike',
    'identity_changed',
    'prev_is_fraud'
]

seen_device_categorical_features = [
    'source',
    'browser'
]

# All features for seen device model
seen_device_features = (
    seen_device_numerical_features +
    seen_device_boolean_features +
    seen_device_categorical_features
)

# UNSEEN/NEW DEVICES MODEL
# For transactions from new devices without historical data
# Uses only transaction-level features that don't require device history
unseen_device_numerical_features = [
    'purchase_hour',
    'amount_per_age'
]

unseen_device_boolean_features = [
    'is_late_night',
    'under_18',
    'age_18_25',
    'age_26_35',
    'age_36_50',
    '50_above',
    'unknown_country'
]

unseen_device_categorical_features = [
    'source',
    'browser'
]

# All features for unseen device model
unseen_device_features = (
    unseen_device_numerical_features +
    unseen_device_boolean_features +
    unseen_device_categorical_features
)

# Combined feature list (all features)
all_features = list(set(numerical_features + categorical_features + boolean_features))


# ============================================================================
# Model Paths (constants)
# ============================================================================
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# Model files
FEATURE_ENGINEER_PATH = MODELS_DIR / "feature_engineer.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

# ============================================================================
# Results Paths (constants)
# ============================================================================
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ============================================================================
# Model Parameters (constants)
# ============================================================================
TARGET_COL = "is_fraud"
RANDOM_STATE = 0
TEST_SIZE = 0.2
CV_FOLDS = 5
OPTIMIZATION_METRIC = "recall"
THRESHOLD = 0.6

START_IDX = 2000

# ============================================================================
# API Configuration (constants with sensible defaults)
# ============================================================================
API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("API_PORT", 8000))
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")

# ============================================================================
# Redis Configuration
# ============================================================================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

DEVICE_STATE_HASH = "device_state_hash"
TRANSACTION_STREAM = "transactions_stream"
LABELS_STREAM = "labels_stream"
RESULT_STREAM = "results_stream"

REDIS_BLOCK_TIMEOUT = int(os.getenv("REDIS_BLOCK_TIMEOUT", 5000)) # milliseconds
BUCKET_SIZE_SECONDS = 60

REDIS_STATE_DIR = PROJECT_ROOT / "redis_saved_state"
DEVICE_STATE_PATH = REDIS_STATE_DIR / "device_state.pkl"
GLOBAL_BUCKETS_PATH = REDIS_STATE_DIR / "global_buckets.pkl"
IP_STATE_PATH = REDIS_STATE_DIR / "ip_state.pkl"

# ============================================================================
# Streamlit Configuration (constants)
# ============================================================================
STREAMLIT_HOST = "0.0.0.0"
STREAMLIT_PORT = 8501


# Hyperparameter grids for tuning
PARAM_GRIDS = {
    'logistic_l1': {
        'penalty': ['l1'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1],
        'solver': ['liblinear'],
        'class_weight': ['balanced'],
        'max_iter': [1000],
        'random_state': [RANDOM_STATE]
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10],
        'class_weight': ['balanced'],
        'random_state': [RANDOM_STATE],
        'n_jobs': [-1]
    },
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [5, 10, 15, 20],
        'random_state': [RANDOM_STATE],
        'n_jobs': [-1]
    }
}

# DB URL
test_db_url = "postgresql://postgres@localhost:5432/fraud_db_test"
db_url = "postgresql://postgres@localhost:5432/fraud_db"

# Evaluation metrics
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']

# Primary optimization metric for hyperparameter tuning
OPTIMIZATION_METRIC = 'recall'  # Prioritize catching fraud

# Cross-validation settings
CV_FOLDS = 5

# Plotting settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Feature importance
TOP_N_FEATURES = 20  # Number of top features to display

# Fraud rate (from EDA)
FRAUD_RATE = 0.09365  # 9.365% of transactions are fraudulent

if __name__ == "__main__":
    print("Configuration Settings")
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"\nTrain Data: {TRAIN_DATA_PATH}")
    print(f"Test Data: {TEST_DATA_PATH}")
    print(f"\nTarget Column: {TARGET_COL}")
    print(f"Optimization Metric: {OPTIMIZATION_METRIC}")
    print(f"Random State: {RANDOM_STATE}")
    print(f"Fraud Rate: {FRAUD_RATE:.4%}")
