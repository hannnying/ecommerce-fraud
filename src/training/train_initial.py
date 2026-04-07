import argparse
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from src.config import (
    MODELS_DIR,
    PREPROCESSOR_PATH,
    DEVICE_STATE_PATH,
    GLOBAL_BUCKETS_PATH,
    IP_STATE_PATH,
    FRAUD_WITH_COUNTRY_PATH,
    START_IDX,
    TARGET_COL,
    db_url,
    categorical_features,
    numerical_features,
    boolean_features,
    seen_device_features,
    unseen_device_features
)
from src.database import session
from src.drift_monitor.drift_monitor import DriftMonitor
from src.domain_models import Prediction, Transaction
from src.feature_engineering.engineer_flexible import TransactionFeatureEngineer
from src.model.models import FraudDetectionModel
from src.model.rule_based import RuleBasedModel
from src.repositories import PredictionRepository, TransactionRepository
from src.state.device_state_flexible import DeviceState
from src.state.global_bucket import GlobalVelocity
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import cross_val_score
from sqlalchemy import Session
from uuid import uuid4


class Train:
    def __init__(self, session: Session, processed_transaction_path=None):
        self.device_state = DeviceState()
        self.global_velocity = GlobalVelocity()
        self.feature_engineer = TransactionFeatureEngineer()
        self.model = FraudDetectionModel(
            resampling_type="smote",
            model_type="logistic_regression",
            custom_params={"penalty": "l2", "C": 2}
        )
        if processed_transaction_path:
            self.processed_transaction_path = Path(processed_transaction_path)
        else:
            self.processed_transaction_path = None
        self.prediction_repository = PredictionRepository(session=session)
        self.transaction_repository = TransactionRepository(session=session)

    def process_transaction(self, raw_transaction):
        transaction_id = str(uuid4())
        processed_transaction, transaction_id, (device_id, state_to_update) = self.feature_engineer.compute_features(raw_transaction, training=True, transaction_id=transaction_id)

        # store raw values for reconstruction of redis states from dataframe
        processed_transaction["id"] = transaction_id
        processed_transaction["device_id"] = raw_transaction["device_id"]
        processed_transaction["country"] = raw_transaction["country"]
        processed_transaction["signup_time"] = raw_transaction["signup_time"]
        processed_transaction["purchase_value"] = raw_transaction["purchase_value"]
        processed_transaction["true_label"] = raw_transaction["is_fraud"]

        return processed_transaction, state_to_update
    
    def update_redis(self, state_to_update, transaction_id, device_id, purchase_time, country):
        self.device_state.update_device_state(
            device_id=device_id,
            state_updates=state_to_update
        )

        self.device_state.update_device_timestamp(device_id, transaction_id, purchase_time)
        self.global_velocity.update_bucket(transaction_id, purchase_time)
        self.global_velocity.update_bucket(transaction_id, purchase_time, country)


    def train_model(self, model_name: str, df: pd.DataFrame):
        if model_name not in ["unseen_devices", "seen_devices"]:
            raise ValueError(f"model_name must be either unseen_devices or seen_devices")
        
        model = FraudDetectionModel(
            resampling_type="smote",
            model_type="logistic_regression",
            custom_params={"penalty": "l2", "random_state": 42},
            model_name=model_name
        )

        if model_name == "unseen_devices":
            X = df[unseen_device_features]
        else:
            X = df[seen_device_features]

        y = df[TARGET_COL]
        model.fit(X, y)

        pred_df = pd.DataFrame()
        pred_df["transaction_id"] = df["transaction_id"]
        pred_df["model_used"] = model_name
        pred_df["model_version"] = 1
        pred_df["fraud_proba"] = model.predict_proba(X)

        return pred_df


    def train_pipeline(self):
        df = pd.read_csv(FRAUD_WITH_COUNTRY_PATH)
        df["signup_time"] = pd.to_datetime(df["signup_time"])
        df["purchase_time"] = pd.to_datetime(df["purchase_time"])

        # Jan - June transactions
        df = df.sort_values(by="purchase_time")
        df = df[df["purchase_time"] < datetime(2015, 7, 1)]

        print(f"\n{'='*70}")
        print(f"TRAINING PIPELINE")
        print(f"{'='*70}")
        print(f"Total transactions (Jan-Jun): {len(df):,}")
        print(f"Date range: {df['purchase_time'].min()} to {df['purchase_time'].max()}")
        print(f"{'='*70}\n")

        counter = 0

        if self.processed_transaction_path and self.processed_transaction_path.exists() and self.processed_transaction_path.is_file():
            processed_train = pd.read_csv(self.processed_transaction_path)
            processed_train["purchase_time"] = pd.to_datetime(processed_train["purchase_time"])
            processed_train["signup_time"] = pd.to_datetime(processed_train["signup_time"])
            print(f"Read full processed data from: {self.processed_transaction_path}...")

        else:
            print(f"File not found at {self.processed_transaction_path}, processing transactions from {FRAUD_WITH_COUNTRY_PATH}")
            processed_transactions = []

            for idx, row in df.iterrows():
                processed_transaction, state_to_update = self.process_transaction(row)
                processed_transactions.append(processed_transaction)

                transaction_id, = processed_transaction["id"]
                device_id = processed_transaction["device_id"]
                country = processed_transaction["country"]
                purchase_time = processed_transaction["purchase_time"]
                state_to_update["prev_is_fraud"] = processed_transaction["true_label"]

                self.update_redis(state_to_update, transaction_id, device_id, purchase_time, country)

                counter += 1

                if not counter % 1000:
                    print(f"processed {counter} transactions: {processed_transaction}")
                    print("\n")

                if counter == 30000:
                    break

            processed_transactions = pd.DataFrame(processed_transactions)

            self.transaction_repository.bulk_insert(processed_transactions)

            # train model on Mar - Jun transactions
            march_start = datetime(2015, 3, 1)
            seen_devices_df = processed_transactions[(processed_transactions["device_txn_idx"] > 1) & (processed_transactions['purchase_time'] >= march_start)]
            unknown_devices_df = processed_transactions[(processed_transactions["device_txn_idx"] == 1) & (processed_transactions['purchase_time'] >= march_start)]

            with mlflow.start_run() as run:
                run_id = run.info.run_id
                print(f"\n{'='*60}")
                print(f"Started MLflow run: {run_id}")
                print(f"{'='*60}\n")

                seen_devices_predictions = self.train_model(model_name="seen_devices", df=seen_devices_df)
                unknown_devices_predictions =  self.train_model(model_name="unseen_devices", df=unknown_devices_df)

                self.prediction_repository.bulk_insert(seen_devices_predictions)
                self.prediction_repository.bulk_insert(unknown_devices_predictions)

                # Build and log drift references (logs to current run)
                print("Building and logging drift reference distributions...")
                drift_monitor = DriftMonitor()
                drift_monitor.build_data_reference(
                    df=processed_transactions,
                    numerical_features=numerical_features,
                    categorical_features=categorical_features,
                    boolean_features=boolean_features,
                    logging=True
                )

                print(f"\n{'='*60}")
                print(f"✅ All artifacts logged to run: {run_id}")
                print(f"   - Model: fraud_detection_model")
                print(f"   - Training metadata: samples, fraud_rate, etc.")
                print(f"{'='*60}\n")

        # if using processed data from local directory, and assuming last redis states are not saved, build redis states from processed data
        if self.processed_transaction_path:
            self.device_state.build_state_from_df(processed_train)
            self.global_velocity.build_bucket_from_df(processed_train)

        self.device_state.export_to_file(DEVICE_STATE_PATH)
        self.global_velocity.export_to_file(GLOBAL_BUCKETS_PATH)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Script to train models on Jan-Jun transactions")
    parser.add_argument("--processed_train_path", help="Load processed transactions") # 'data/processed/processed_train.csv'
    args = parser.parse_args()
    train_object = Train(
        session=session,
        processed_transaction_path=args.processed_train_path
    )
    train_object.train_pipeline()

