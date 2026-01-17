# Run this script to train the initial model on the first 50,000 transactions
import argparse
import json
import pickle

import numpy as np
import pandas as pd
from src.config import (
    MODELS_DIR,
    DEVICE_STATE_PATH,
    GLOBAL_BUCKETS_PATH,
    IP_STATE_PATH,
    RAW_DATA_PATH,
    START_IDX,
    TARGET_COL
)
from src.models.models_v3 import FraudDetectionModel, FraudModelTuner
from src.preprocessing import FraudDataPreprocessor
from src.feature_engineering.engineer import TransactionFeatureEngineer
from src.models.models_v3 import FraudDetectionModel
from src.state.device_state import DeviceState
from src.state.global_bucket import GlobalVelocity
from src.state.ip_state import IPState
from src.state.prediction_store import PredictionStore
from uuid import uuid4


class Train:
    def __init__(self):
        self.device_state = DeviceState()
        self.global_velocity = GlobalVelocity()
        self.ip_state = IPState()
        self.feature_engineer = TransactionFeatureEngineer()
        self.model = FraudDetectionModel(
            resampling_type="random_undersampling",
            model_type="random_forest",
            custom_params={"n_estimators":50, "max_depth":20, "min_samples_leaf":2, "min_samples_split":5, "random_state":0}
        )

    def train_pipeline(self):
        df = pd.read_csv(RAW_DATA_PATH)
        df["signup_time"] = pd.to_datetime(df["signup_time"])
        df["purchase_time"] = pd.to_datetime(df["purchase_time"])
        df = df.sort_values(by="purchase_time")

        train = df[:START_IDX]
        train.reset_index(inplace=True)
        processed_transactions = []

        for idx, row in train.iterrows():
            transaction_id = str(uuid4)
            processed_transaction, transaction_id, device_data = self.feature_engineer.compute_features(row, training=True, transaction_id=transaction_id)
            processed_transactions.append(processed_transaction)

            device_id, txn_count, purchase_value, sex, age, first_seen, ip_addresses, sources, purchase_time, source, ip_address = device_data
            self.device_state.update_device_state(
                device_id,
                txn_count,
                purchase_value,
                sex,
                age,
                first_seen,
                ip_addresses,
                sources,
                purchase_time,
                source,
                ip_address
            )
            self.device_state.update_device_timestamp(device_id, transaction_id, purchase_time)
            self.global_velocity.update_global_bucket(purchase_time)
            self.ip_state.update_ip_txn_idx(ip_address)

        X_train = pd.DataFrame(processed_transactions)
        y_train = train[TARGET_COL]
        self.model.fit(X_train, y_train)

        # save states
        self.device_state.export_to_file(DEVICE_STATE_PATH)
        self.global_velocity.export_to_file(GLOBAL_BUCKETS_PATH)
        self.ip_state.export_to_file(IP_STATE_PATH)

        model_path = MODELS_DIR / f"{self.model.model_type}.pkl"
        with open(model_path, "wb") as m:
            pickle.dump(self.model, m)
            print(f"model saved at: {model_path}")


if __name__=="__main__":
    Train().train_pipeline()