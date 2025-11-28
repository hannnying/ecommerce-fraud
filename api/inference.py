import os
import pickle
import random
import string
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from pandera.typing import DataFrame
from api.schemas import Transaction, TransactionDTO


# determine which directory API is running from, and what model you want to use

ROOT_DIR = Path(__file__).resolve().parents[1]
FEATURE_ENGINEER_PATH = ROOT_DIR / "models" / "feature_engineer.pkl"
PREPROCESSOR_PATH = ROOT_DIR / "models" / "preprocessor.pkl"
MODEL_PATH = ROOT_DIR / "models" / "logistic_regression_model.pkl"

def load_feature_engineer():
    with open(FEATURE_ENGINEER_PATH, "rb") as f:
        feature_engineer = pickle.load(f)
    return feature_engineer

def get_feature_engineer():
    return load_feature_engineer()

def load_preprocessor():
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    return preprocessor

def get_preprocessor():
    return load_preprocessor()

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def get_model():
    return load_model()

def generate_user_id(): # not used in model, but required as per schema
    return random.randint(1, 999999)

def generate_device_id():
    upper_letters = string.ascii_uppercase
    random_letters_list = random.choices(upper_letters, k=13)
    return "".join(random_letters_list)

def generate_ip_address():
    """Generate IP address from a normal distribution with mean and standard deviation based on original dataset, bounded by max and min."""
    mean = 2.152145e+09
    std = 1.248497e+09
    ip_numeric = random.gauss(mean, std)
    return max(52093.5, min(ip_numeric, 4.294850e+09))

def generate_signup_time():
    seconds_before = random.randint(1, 365 * 24 * 60 * 60)
    signup = datetime.now() - timedelta(seconds=seconds_before)
    return signup.strftime("%Y-%m-%d %H:%M:%S")


def run_inference(
        transaction: DataFrame[Transaction],
        feature_engineer,
        preprocessor,
        model
    ):

    transaction_df = feature_engineer.transform(transaction)
    X, y = preprocessor.preprocess(transaction_df)
    X_processed = preprocessor.transform(X)

    predicted_class = model.predict(X_processed)
    probability = model.predict_proba(X_processed)[:, 1]

    return predicted_class, probability

