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
    categorical_features,
    numerical_features,
    boolean_features,
    seen_device_features,
    unseen_device_features
)
from src.drift_monitor.drift_monitor import DriftMonitor
from src.feature_engineering.engineer_flexible import TransactionFeatureEngineer
from src.models.models import FraudDetectionModel
from src.models.rule_based import RuleBasedModel
from src.state.device_state_flexible import DeviceState
from src.state.global_bucket import GlobalVelocity
from src.state.ip_state import IPState
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import cross_val_score
from uuid import uuid4


class Train:
    def __init__(self, processed_transaction_path=None):
        self.device_state = DeviceState()
        self.global_velocity = GlobalVelocity()
        # self.ip_state = IPState()
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
        

    def train_unknown_devices_model_pipeline(self, unknown_devices_df):
        """
        Resample, scale and encode transactions by unknown devices.
        Train and save model.
        """
        unknown_devices_model = FraudDetectionModel(
            resampling_type="smote",
            model_type="logistic_regression",
            custom_params={"penalty": "l2", "random_state": 42},
            model_name="unseen_devices"
        )
        X, y = unknown_devices_df[unseen_device_features], unknown_devices_df[TARGET_COL]
        unknown_devices_model.fit(X, y)


    def train_seen_devices_model_pipeline(self, seen_devices_df):
        """
        Scale and encode transactions by seen devices.
        Train and save model.
        """
        seen_devices_model = FraudDetectionModel(
            model_type="logistic_regression",
            custom_params={"penalty": "l2", "random_state": 42},
            model_name="seen_devices"
        )
        X, y = seen_devices_df[seen_device_features], seen_devices_df[TARGET_COL]
        seen_devices_model.fit(X, y)


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

        if self.processed_transaction_path and self.processed_transaction_path.exists() and self.processed_transaction_path.is_file():
                processed_train = pd.read_csv(self.processed_transaction_path)
                processed_train["purchase_time"] = pd.to_datetime(processed_train["purchase_time"])
                print(f"Read full processed data from: {self.processed_transaction_path}...")

        else:
            print(f"File not found at {self.processed_transaction_path}, processing transactions from {FRAUD_WITH_COUNTRY_PATH}")
            processed_train = []

            for idx, row in df.iterrows():
                transaction_id = str(uuid4())  # ✅ Fixed: Added parentheses
                purchase_time = row["purchase_time"]
                processed_transaction, transaction_id, (device_id, state_to_update) = self.feature_engineer.compute_features(row, training=True, transaction_id=transaction_id)

                # store raw values for reconstruction of redis states from dataframe
                processed_transaction["transaction_id"] = transaction_id
                processed_transaction["device_id"] = row["device_id"]
                processed_transaction["country"] = row["country"]
                processed_transaction["signup_time"] = row["signup_time"]
                processed_transaction["purchase_value"] = row["purchase_value"]

                # assess transaction risk using rules
                processed_transaction["rule_label"] = RuleBasedModel().predict(processed_transaction)
                processed_transaction["is_fraud"] = row["is_fraud"]

                processed_train.append(processed_transaction)

                state_to_update["prev_is_fraud"] = row["is_fraud"]

                if idx % 1000 == 0:
                    print(f"Processed {idx}: {processed_transaction}...")
                    print("\n")

                self.device_state.update_device_state(
                    device_id=device_id,
                    state_updates=state_to_update
                )

                self.device_state.update_device_timestamp(device_id, transaction_id, purchase_time)
                self.global_velocity.update_bucket(transaction_id, purchase_time)
                self.global_velocity.update_bucket(transaction_id, purchase_time, row["country"])


            processed_train = pd.DataFrame(processed_train)

            print(f"\n{'='*70}")
            print(f"FEATURE ENGINEERING COMPLETE")
            print(f"{'='*70}")
            print(f"Total processed: {len(processed_train):,}")
            print(f"{'='*70}\n")

            # Save processed training data
            processed_data_dir = Path("data/processed")
            processed_data_dir.mkdir(exist_ok=True, parents=True)

            processed_train_path = processed_data_dir / "processed_train.csv"

            print(f"{'='*70}")
            print(f"SAVING PROCESSED DATA")
            print(f"{'='*70}")

            # Save full processed data (Jan-Jun)
            processed_train.to_csv(processed_train_path, index=False)
            print(f"✓ Saved full processed data: {processed_train_path}")
            print(f"  Shape: {processed_train.shape}")

            print(f"{'='*70}\n")
        
        # train model on Mar - Jun transactions
        march_start = datetime(2015, 3, 1)
        seen_devices_df = processed_train[(processed_train["device_txn_idx"] > 1) & (processed_train['purchase_time'] >= march_start)]
        unknown_devices_df = processed_train[(processed_train["device_txn_idx"] == 1) & (processed_train['purchase_time'] >= march_start)]

        print(f"{'='*70}")
        print(f"TRAINING DATA SPLIT")
        print(f"{'='*70}")
        print(f"Seen devices (Mar-Jun): {len(seen_devices_df):,}")
        print(f"Unseen devices (Mar-Jun): {len(unknown_devices_df):,}")
        print(f"{'='*70}\n")

        # Start single MLflow run for model, drift references, and metadata
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            print(f"\n{'='*60}")
            print(f"Started MLflow run: {run_id}")
            print(f"{'='*60}\n")

            self.train_seen_devices_model_pipeline(seen_devices_df)
            self.train_unknown_devices_model_pipeline(unknown_devices_df)

            # Build and log drift references (logs to current run)
            print("Building and logging drift reference distributions...")
            drift_monitor = DriftMonitor()
            drift_monitor.build_data_reference(
                df=processed_train,
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

        # save states
        if self.processed_transaction_path:
            self.device_state.build_state_from_df(processed_train)
            self.global_velocity.build_bucket_from_df(processed_train)

        self.device_state.export_to_file(DEVICE_STATE_PATH)
        self.global_velocity.export_to_file(GLOBAL_BUCKETS_PATH)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Script to train models on Jan-Jun transactions")
    parser.add_argument("--processed_train_path", help="Load processed transactions") # 'data/processed/processed_train.csv'
    args = parser.parse_args()
    Train(processed_transaction_path=args.processed_train_path).train_pipeline()

