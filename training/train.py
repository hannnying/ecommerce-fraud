# Run this script to train the initial model on the first 50,000 transactions
import argparse
import json
import pickle

import numpy as np
import pandas as pd
from src.config import (
    MODELS_DIR,
    PREPROCESSOR_PATH,
    DEVICE_STATE_PATH,
    GLOBAL_BUCKETS_PATH,
    IP_STATE_PATH,
    RAW_DATA_PATH,
    START_IDX,
    TARGET_COL,
    THRESHOLD
)
from src.feature_engineering.engineer import TransactionFeatureEngineer
from src.models.models import FraudDetectionModel
from src.models.rule_based import RuleBasedModel
from src.state.device_state import DeviceState
from src.state.global_bucket import GlobalVelocity
from src.state.ip_state import IPState
from sklearn.metrics import average_precision_score, classification_report
from uuid import uuid4


class Train:
    def __init__(self):
        self.device_state = DeviceState()
        self.global_velocity = GlobalVelocity()
        self.ip_state = IPState()
        self.feature_engineer = TransactionFeatureEngineer()
        self.model = FraudDetectionModel(
            resampling_type="smote",
            model_type="logistic_regression",
            custom_params={"penalty": "l2", "C": 2}
        )


    def train_pipeline(self, holdout=0.2):
        df = pd.read_csv(RAW_DATA_PATH)
        df["signup_time"] = pd.to_datetime(df["signup_time"])
        df["purchase_time"] = pd.to_datetime(df["purchase_time"])
        df = df.sort_values(by="purchase_time")

        INITIAL_START_IDX = int((1 - holdout) * START_IDX)
        train = df[:START_IDX]
        train.reset_index(inplace=True)

        processed_train = []
        y_train = []

        y_val = []
        y_val_model = []
        y_val_proba_model = []
        y_val_hybrid = []

        for idx, row in train.iterrows():
            transaction_id = str(uuid4)
            processed_transaction, transaction_id, device_data = self.feature_engineer.compute_features(row, training=True, transaction_id=transaction_id)
            processed_train.append(processed_transaction)

            if idx < INITIAL_START_IDX:
                y_train.append(row[TARGET_COL])

            else:
                rule_based_label = RuleBasedModel().predict(processed_transaction)
                model_proba = self.model.predict_proba(pd.DataFrame(processed_transaction, index=[0])[self.model.scaler.feature_names])[0]
                model_label = int(self.model.predict(pd.DataFrame(processed_transaction, index=[0])[self.model.scaler.feature_names])[0])
                y_val.append(row[TARGET_COL])

                if rule_based_label > -1:
                    y_val_hybrid.append(rule_based_label)
                else:
                    y_val_hybrid.append(model_label)

                y_val_model.append(model_label)
                y_val_proba_model.append(model_proba)

            # Update state after prediction
            device_id, txn_count, purchase_value, sex, age, purchase_time, signup_time, ip_address, txn_count, first_seen_signup, first_seen, identities, purchase_values = device_data

            self.device_state.update_device_state(
                device_id,
                purchase_value,
                sex,
                age,
                purchase_time,
                signup_time,
                ip_address,
                txn_count,
                first_seen_signup,
                first_seen,
                identities,
                purchase_values
            )

            self.device_state.update_device_timestamp(device_id, transaction_id, purchase_time)
            self.global_velocity.update_global_bucket(purchase_time)
            self.ip_state.update_ip_txn_idx(ip_address)

            # train model once before evaluating on holdout set
            if idx + 1 == INITIAL_START_IDX:
                X_train = pd.DataFrame(processed_train)
                y_train = np.array(y_train)
                # self.model = 
                self.model.fit(X_train, y_train, logging=False)

            if not idx % 1000:
                print(f"processed {idx} transactions")

        # evaluate
        print(f"PR-AUC of model, including transactions with rule based labels: {average_precision_score(y_val, y_val_proba_model)}")
        print(f"Classification report, LR: {classification_report(y_val, y_val_model)}")
        print(f"Classification report of hybrid model (rule-based + LR): {classification_report(y_val, y_val_hybrid)}")

        # Train on holdout
        X_train_final = pd.DataFrame(processed_train)
        y_train_final = np.concat([y_train, y_val], axis=0)

        # resample, scale, train and test on holdout
        self.model.fit(X_train_final, np.array(y_train_final), logging=True)

        # save states
        self.device_state.export_to_file(DEVICE_STATE_PATH)
        self.global_velocity.export_to_file(GLOBAL_BUCKETS_PATH)
        self.ip_state.export_to_file(IP_STATE_PATH)

        # save preprocessor and model to local directory
        with open(PREPROCESSOR_PATH, "wb") as p:
            pickle.dump(self.model.scaler, p)
            print(f"preprocessor saved at: {PREPROCESSOR_PATH}")

        model_path = MODELS_DIR / f"{self.model.model_type}.pkl"
        with open(model_path, "wb") as m:
            pickle.dump(self.model.model, m)
            print(f"model saved at: {model_path}")


if __name__=="__main__":
    Train().train_pipeline()