"""
Configuration file for fraud detection project.

Contains paths, constants, and hyperparameters used throughout the project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
RAW_DATA_PATH = DATA_DIR / "Fraud_Data.csv"
IP_COUNTRY_PATH = DATA_DIR / "IpAddress_to_Country.csv"
GDP_DATA_PATH = DATA_DIR / "gdp_usd.xlsx"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Target column
TARGET_COL = "is_fraud"

# Columns to drop before modeling
DROP_COLS = [
    "user_id", "signup_time", "purchase_time",
    "device_id", "ip_address", "country",
    "sex", "age", "transactions_by_user_id"
]

# Categorical features
CATEGORICAL_FEATURES = ["source", "browser"]

# Random seed for reproducibility
RANDOM_STATE = 42

# Train/validation split
TEST_SIZE = 0.2

# Model configurations
MODEL_CONFIGS = {
    'logistic_regression': {
        'display_name': 'Logistic Regression (L1)',
        'description': 'Best model - 72% recall, minimal overfitting',
        'recommended': True
    },
    'random_forest': {
        'display_name': 'Random Forest',
        'description': 'Tree-based model, ~72% recall after tuning',
        'recommended': True
    },
    'xgboost': {
        'display_name': 'XGBoost',
        'description': 'Gradient boosting, ~73% recall after tuning',
        'recommended': True
    }
}

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
