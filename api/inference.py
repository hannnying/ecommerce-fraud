import os
import random
import string
from datetime import datetime, timedelta
import pandas as pd
from pandera.typing import DataFrame
from api.schemas import Transaction, TransactionDTO


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

